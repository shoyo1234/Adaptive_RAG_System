"""What would the LLM actually receive? Context size and noise, per mode. No LLM is called.

Usage (from repo root): uv run python -m evaluation.context_quality --split test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken

from adaptive_rag import config
from adaptive_rag.data import load_scifact, split_queries
from adaptive_rag.indexes import HybridIndex
from adaptive_rag.pipeline import RetrievalPipeline
from adaptive_rag.rerank import Reranker

NS = (3, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--modes", default="hybrid,static5,static10,adaptive")
    args = ap.parse_args()
    modes = args.modes.split(",")

    enc = tiktoken.get_encoding("cl100k_base")
    tok_cache = {}

    def ntok(doc_id):
        if doc_id not in tok_cache:
            tok_cache[doc_id] = len(enc.encode(corpus[doc_id]))
        return tok_cache[doc_id]

    corpus, queries, qrels = load_scifact()
    dev_ids, test_ids = split_queries(queries)
    qids = dev_ids if args.split == "dev" else test_ids

    pipe = RetrievalPipeline(HybridIndex(corpus), reranker=Reranker())
    pipe.warm_up()

    per_query, rows = [], []
    for m in modes:
        acc = {n: {"tok": [], "noise_tok": [], "prec": [], "hit": []} for n in NS}
        for i, qid in enumerate(qids, 1):
            ranked = pipe.run(m, queries[qid])["ranked"]
            for n in NS:
                top = ranked[:n]
                rel = [d in qrels[qid] for d in top]
                toks = [ntok(d) for d in top]
                acc[n]["tok"].append(sum(toks))
                acc[n]["noise_tok"].append(sum(t for t, r in zip(toks, rel) if not r))
                acc[n]["prec"].append(np.mean(rel))
                acc[n]["hit"].append(float(any(rel)))
                per_query.append({"mode": m, "n": n, "qid": qid, "tokens": sum(toks),
                                  "precision": float(np.mean(rel)), "hit": float(any(rel))})
            if i % 50 == 0:
                print(f"  [{m}] {i}/{len(qids)}")
        for n in NS:
            a = acc[n]
            rows.append({"mode": m, "N": n,
                         "context_tokens": np.mean(a["tok"]),
                         "noise_tokens": np.mean(a["noise_tok"]),
                         "noise_share": 1 - np.mean(a["prec"]),
                         "precision@N": np.mean(a["prec"]),
                         "hit@N": np.mean(a["hit"])})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\nLLM context quality on {args.split} split ({len(qids)} queries)")
    print(df.round(3).to_string(index=False))

    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    df.to_csv(Path(config.RESULTS_DIR) / f"context_quality_{args.split}.csv", index=False)
    pd.DataFrame(per_query).to_csv(
        Path(config.RESULTS_DIR) / f"context_quality_perquery_{args.split}.csv", index=False)


if __name__ == "__main__":
    main()