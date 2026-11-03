"""Does reranking help, and can a confidence signal predict when it won't?

Usage (from repo root, after Step 1 is done):
    uv run python -m evaluation.diagnose_gate --ce-max-length 256
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from adaptive_rag import config
from adaptive_rag.data import load_scifact, split_queries
from adaptive_rag.indexes import HybridIndex
from adaptive_rag.pipeline import RetrievalPipeline
from adaptive_rag.rerank import Reranker
from evaluation.metrics import ndcg_at_k

M = 10      # number of top fused scores the signals look at


def signals(fused_scores, dense_ids, sparse_ids):
    s = np.array(fused_scores[:M], dtype=float)
    p = s / s.sum()
    return {
        "rel_gap": (s[0] - s[1]) / s[0],                     # high = clear winner
        "std_norm": float(np.std(s / s[0])),                 # high = peaked list
        "entropy": float(-(p * np.log(p)).sum() / np.log(len(p))),   # high = flat/unsure
        "agreement": len(set(dense_ids[:5]) & set(sparse_ids[:5])) / 5,
    }


def auc(x, y):
    """P(signal of a 'not helped' query > signal of a 'helped' query). 0.5 = useless."""
    r = pd.Series(x).rank()                                  # average ranks for ties
    pos = np.array(y, dtype=bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce-max-length", type=int,
                    default=config.CROSS_ENCODER_MAX_LENGTH)
    ap.add_argument("--rerank-k", type=int, default=10)
    args = ap.parse_args()

    corpus, queries, qrels = load_scifact()
    dev_ids, _ = split_queries(queries)
    index = HybridIndex(corpus)
    pipe = RetrievalPipeline(index, reranker=Reranker(max_length=args.ce_max_length))
    pipe.warm_up()

    rows = []
    for i, qid in enumerate(dev_ids, 1):
        q = queries[qid]
        hyb = pipe.run_hybrid(q)
        sta = pipe.run_static(q, args.rerank_k)
        dense = [d for d, _ in index.dense_search(q)]
        sparse = [d for d, _ in index.sparse_search(q)]
        row = {"qid": qid,
               "ndcg_hybrid": ndcg_at_k(hyb["ranked"], qrels[qid], 10),
               "ndcg_static": ndcg_at_k(sta["ranked"], qrels[qid], 10)}
        row.update(signals([s for _, s in hyb["pool"]], dense, sparse))
        rows.append(row)
        if i % 25 == 0:
            print(f"  {i}/{len(dev_ids)}")

    df = pd.DataFrame(rows)
    df["delta"] = df["ndcg_static"] - df["ndcg_hybrid"]
    df["group"] = np.where(df["delta"] > 0.01, "helped",
                   np.where(df["delta"] < -0.01, "hurt", "unchanged"))
    df["not_helped"] = df["group"] != "helped"

    print(f"\nReranking top-{args.rerank_k} vs hybrid, per query (n={len(df)}):")
    print(df["group"].value_counts().to_string())
    print(f"\nMean NDCG@10: hybrid {df.ndcg_hybrid.mean():.4f}, "
          f"static {df.ndcg_static.mean():.4f}")
    print(f"Queries where reranking did NOT help: {df.not_helped.mean():.1%}")

    sig = ["rel_gap", "std_norm", "entropy", "agreement"]
    print("\nMean signal value per group:")
    print(df.groupby("group")[sig].mean().round(4).to_string())

    print("\nAUC for predicting 'reranking did not help' (0.5 = no information;")
    print("rel_gap/std_norm/agreement should be > 0.5, entropy < 0.5 if informative):")
    for s in sig:
        print(f"  {s:10s} AUC = {auc(df[s], df['not_helped']):.3f}")

    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    out = Path(config.RESULTS_DIR) / "diagnose_gate_dev.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()