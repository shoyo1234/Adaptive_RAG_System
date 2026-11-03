"""Where does the gate (skip if dense & BM25 agree on top-1, else rerank 5) go wrong?
No models are run.  Usage: uv run python -m evaluation.gate_errors
"""
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_rag import config

R = Path(config.RESULTS_DIR)
k = pd.read_csv(R / "kcurve_dev.csv", dtype={"qid": str})
g = pd.read_csv(R / "diagnose_gate_dev.csv", dtype={"qid": str})[["qid", "group"]]
df = k.merge(g, on="qid")                       # group = static10 vs hybrid outcome
df["decision"] = np.where(df["top1_agree"] == 1, "skip", "rerank")

print("Gate decision vs what reranking top-10 did to that query:")
print(pd.crosstab(df["decision"], df["group"], margins=True).to_string())

print("\nMean NDCG@10 by branch (hybrid ranking vs rerank-top-5 ranking):")
for d, sub in df.groupby("decision"):
    print(f"  {d:6s} n={len(sub):3d}  hybrid {sub.ndcg_hybrid.mean():.4f}  "
          f"rerank5 {sub.ndcg_k5.mean():.4f}  "
          f"(rerank5 - hybrid = {sub.ndcg_k5.mean() - sub.ndcg_hybrid.mean():+.4f})")

print("\nWhere is the first relevant passage in the hybrid ranking, per branch?")
bins = pd.cut(df["first_rel_rank"], [0, 1, 3, 5, 10, 10_000],
              labels=["1", "2-3", "4-5", "6-10", ">10"])
print(pd.crosstab(df["decision"], bins).to_string())

missed = df[(df.decision == "skip") & (df.ndcg_k5 > df.ndcg_hybrid + 0.01)]
print(f"\nSkipped but rerank-5 would have helped: {len(missed)} queries")
cols = ["qid", "first_rel_rank", "ndcg_hybrid", "ndcg_k5", "dense_gap", "bm25_rel_gap"]
print(missed[cols].round(3).to_string(index=False))