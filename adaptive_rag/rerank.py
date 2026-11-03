"""Cross-encoder reranker."""
import time

from sentence_transformers import CrossEncoder

from adaptive_rag import config


class Reranker:
    def __init__(self, model_name=config.CROSS_ENCODER_MODEL,
                 max_length=config.CROSS_ENCODER_MAX_LENGTH):
        # Load ONCE. Loading inside the query loop would swamp every latency number.
        self.model = CrossEncoder(model_name, device="cpu", max_length=max_length)

    def warm_up(self, n=3):
        pairs = [("warm up query", "warm up passage text")] * 8
        for _ in range(n):
            self.model.predict(pairs, batch_size=8, show_progress_bar=False)

    def rerank(self, query, doc_ids, id_to_text, batch_size=16):
        """
        Score (query, passage) pairs and return [(doc_id, ce_score)] best-first.
        Pass passage TEXT to the model, not IDs.
        """
        if not doc_ids:
            return []
        pairs = [(query, id_to_text[d]) for d in doc_ids]
        scores = self.model.predict(pairs, batch_size=batch_size,
                                    show_progress_bar=False)
        ranked = sorted(zip(doc_ids, scores), key=lambda x: -float(x[1]))
        return [(d, float(s)) for d, s in ranked]


if __name__ == "__main__":
    import numpy as np
    from adaptive_rag.data import load_scifact, split_queries
    from adaptive_rag.fusion import rrf
    from adaptive_rag.indexes import HybridIndex

    corpus, queries, qrels = load_scifact()
    dev_ids, _ = split_queries(queries)
    index = HybridIndex(corpus)
    rr = Reranker()
    rr.warm_up()

    # --- Check A: structure + sanity on one query ---
    qid = dev_ids[0]
    q = queries[qid]
    fused = rrf([index.dense_search(q), index.sparse_search(q)])
    top20 = [d for d, _ in fused[:20]]
    out = rr.rerank(q, top20, corpus)
    scores = [s for _, s in out]
    print(f"Query [{qid}]: {q}")
    print(f"Relevant: {list(qrels[qid])}")
    print(f"n={len(out)}, sorted_desc={scores == sorted(scores, reverse=True)}, "
          f"same_ids={set(d for d, _ in out) == set(top20)}")
    for r, (d, s) in enumerate(out[:5], 1):
        print(f"  {r}. {d}  ce={s:.3f}  {corpus[d][:70]}...")

    # --- Check B: latency vs number of passages ---
    print("\nRerank latency vs. number of passages (10 dev queries each):")
    sample = dev_ids[:10]
    for n in (5, 10, 20, 30, 50):
        times = []
        for sid in sample:
            sq = queries[sid]
            cand = [d for d, _ in rrf([index.dense_search(sq),
                                       index.sparse_search(sq)])[:n]]
            t0 = time.perf_counter()
            rr.rerank(sq, cand, corpus)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  n={n:>2}: mean {np.mean(times):7.1f} ms   p95 {np.percentile(times, 95):7.1f} ms")