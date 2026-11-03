"""Offline gate simulation from saved per-query results. No models are run.

Usage (from repo root): uv run python -m evaluation.simulate_gate
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_rag import config
from evaluation.diagnose_gate_v2 import auc, auc_ci

SIGNALS = ["rel_gap", "bm25_rel_gap", "bm25_z", "dense_gap", "dense_z",
           "top1_agree", "overlap_top3"]


def simulate(df, skip, hybrid_ms, rerank_ms):
    """Skipped queries keep the hybrid ranking; the rest take the static10 ranking."""
    ndcg = float(np.where(skip, df["ndcg_hybrid"], df["ndcg_static"]).mean())
    rate = float(np.mean(skip))
    return rate, ndcg, hybrid_ms + (1 - rate) * rerank_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-ms", type=float, default=71.0)
    ap.add_argument("--rerank-ms", type=float, default=1088.0)  # static10 mean 1159 - hybrid 71
    args = ap.parse_args()

    df = pd.read_csv(Path(config.RESULTS_DIR) / "diagnose_gate_v2_dev.csv",
                     dtype={"qid": str})
    n = len(df)
    top1_ok = (df["first_rel_rank"] == 1).to_numpy()
    print(f"Hybrid top-1 already relevant: {top1_ok.sum()}/{n} ({top1_ok.mean():.1%})")

    print("\nAUC for predicting 'hybrid top-1 is already relevant' "
          "(higher signal = safer to skip):")
    res = []
    for s in SIGNALS:
        a = auc(df[s], top1_ok)
        lo, hi = auc_ci(df[s], top1_ok)
        res.append((a, s, lo, hi))
    for a, s, lo, hi in sorted(res, reverse=True):
        print(f"  {s:14s} AUC={a:.3f}  [{lo:.3f}, {hi:.3f}]")

    agree = df["top1_agree"].to_numpy() == 1
    rules = [("never skip (= static10)", np.zeros(n, bool)),
             ("always skip (= hybrid)", np.ones(n, bool)),
             ("ORACLE: skip when not helped", df["not_helped"].to_numpy()),
             ("top1_agree", agree)]
    for col in ("bm25_rel_gap", "dense_gap"):
        for q in (0.3, 0.5):
            t = df[col].quantile(q)
            high = (df[col] >= t).to_numpy()
            rules.append((f"{col} >= p{int(q * 100)} ({t:.3f})", high))
            rules.append((f"top1_agree & {col} >= p{int(q * 100)}", agree & high))

    base = df["ndcg_static"].mean()
    rows = []
    for name, skip in rules:
        rate, ndcg, lat = simulate(df, np.asarray(skip, bool),
                                   args.hybrid_ms, args.rerank_ms)
        rows.append({"rule": name, "skip_rate": rate, "ndcg@10": ndcg,
                     "vs_static10": ndcg - base, "est_lat_ms": lat,
                     "speedup_x": (args.hybrid_ms + args.rerank_ms) / lat})
    out = pd.DataFrame(rows)
    print("\nSimulated gates on dev (skipped queries keep the hybrid ranking):")
    print(out.round(4).to_string(index=False))
    print("\nNOTE: latency here is ESTIMATED from mean stage costs, and thresholds are")
    print("chosen on the same dev queries, so treat gains as optimistic. Real")
    print("measurements come in Task 2.6; the test split is the final check on Day 3.")

    out.to_csv(Path(config.RESULTS_DIR) / "simulate_gate_dev.csv", index=False)


if __name__ == "__main__":
    main()