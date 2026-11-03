"""Dense (FAISS) and sparse (BM25) indexes built from ONE shared passage list."""
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from adaptive_rag import config

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text):
    """Lowercase + word tokens. Used for BOTH passages and queries."""
    return _TOKEN_RE.findall(text.lower())


class HybridIndex:
    def __init__(self, corpus, model_name=config.EMBEDDING_MODEL,
                 index_dir=config.INDEX_DIR):
        # One fixed ordering, used by both indexes.
        self.doc_ids = sorted(corpus.keys())
        self.texts = [corpus[d] for d in self.doc_ids]
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name, device="cpu")
        self._build_or_load_dense()
        self._build_sparse()

    # ---------- dense ----------
    def _paths(self):
        tag = self.model_name.replace("/", "_")
        return (self.index_dir / f"{tag}.faiss",
                self.index_dir / f"{tag}.meta.json")

    def _build_or_load_dense(self):
        faiss_path, meta_path = self._paths()
        if faiss_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["doc_ids"] == self.doc_ids:        # stale-index guard
                self.faiss_index = faiss.read_index(str(faiss_path))
                print(f"Loaded FAISS index ({self.faiss_index.ntotal} vectors)")
                return
            print("Saved index does not match corpus; rebuilding.")

        print("Embedding corpus (first run only)...")
        t0 = time.perf_counter()
        emb = self.model.encode(
            self.texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")
        self.faiss_index = faiss.IndexFlatIP(emb.shape[1])  # IP == cosine (normalized)
        self.faiss_index.add(emb)
        faiss.write_index(self.faiss_index, str(faiss_path))
        meta_path.write_text(json.dumps({"doc_ids": self.doc_ids}))
        print(f"Built FAISS index in {time.perf_counter() - t0:.1f}s")

    # ---------- sparse ----------
    def _build_sparse(self):
        t0 = time.perf_counter()
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        print(f"Built BM25 index in {time.perf_counter() - t0:.1f}s")

    # ---------- search ----------
    def dense_search(self, query, n=config.N_CANDIDATES):
        q = self.model.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True).astype("float32")
        scores, idxs = self.faiss_index.search(q, n)
        return [(self.doc_ids[i], float(s)) for s, i in zip(scores[0], idxs[0])]

    def sparse_search(self, query, n=config.N_CANDIDATES):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argpartition(-scores, n)[:n]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    from adaptive_rag.data import load_scifact, split_queries

    corpus, queries, qrels = load_scifact()
    dev_ids, _ = split_queries(queries)
    index = HybridIndex(corpus)

    # --- Check A: structure on one query ---
    qid = dev_ids[0]
    print(f"\nQuery [{qid}]: {queries[qid]}")
    print(f"Relevant: {list(qrels[qid])}")
    dense = index.dense_search(queries[qid])
    sparse = index.sparse_search(queries[qid])
    for name, res in (("DENSE", dense), ("SPARSE", sparse)):
        s = [x[1] for x in res]
        print(f"\n{name}: n={len(res)}, sorted_desc={s == sorted(s, reverse=True)}, "
              f"all_ids_valid={all(d in corpus for d, _ in res)}")
        for rank, (d, sc) in enumerate(res[:5], 1):
            print(f"  {rank}. {d}  score={sc:.4f}  {corpus[d][:70]}...")
    print(f"\nTop-5 overlap dense/sparse: "
          f"{len({d for d,_ in dense[:5]} & {d for d,_ in sparse[:5]})}/5")

    # --- Check B: recall@50 of each stream over all dev queries ---
    hit_d = hit_s = 0
    for q in dev_ids:
        rel = set(qrels[q])
        hit_d += bool(rel & {d for d, _ in index.dense_search(queries[q])})
        hit_s += bool(rel & {d for d, _ in index.sparse_search(queries[q])})
    print(f"\nRecall@50 (>=1 relevant found) over {len(dev_ids)} dev queries:")
    print(f"  Dense : {hit_d / len(dev_ids):.3f}")
    print(f"  Sparse: {hit_s / len(dev_ids):.3f}")