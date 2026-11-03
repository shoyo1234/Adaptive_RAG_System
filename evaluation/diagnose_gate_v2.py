"""Richer gate signals from RAW retrieval scores, tested against the saved
helped/hurt/unchanged labels. No reranking is run, so this is fast.

Usage (from repo root): uv run python -m evaluation.diagnose_gate_v2
"""
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_rag import config
from adaptive_rag.data import load_scifact, split_queries
from adaptive_rag.fusion import rrf
from adaptive_rag.indexes import HybridIndex


def z_top1(scores):
    s = np.array(scores, dtype=float)
    return float((s[0] - s.mean()) / (s.std() + 1e-9))


def features(index, query):
    dense = index.dense_search(query)
    sparse = index.sparse_search(query)
    fused = rrf([dense, sparse])
    d = np.array([s for _, s in dense])
    s = np.array([x for _, x in sparse])
    d_ids = [i for i, _ in dense]
    s_ids = [i for i, _ in sparse]
    f_ids = [i for i, _ in fused]
    feats = {
        "dense_top1": d[0],
        "dense_gap": d[0] - d[1],
        "dense_z": z_top1(d),
        "bm25_top1": s[0],
        "bm25_rel_gap": (s[0] - s[1]) / s[0],
        "bm25_z": z_top1(s),
        "top1_agree": float(d_ids[0] == s_ids[0]),
        "overlap_top3": len(set(d_ids[:3]) & set(s_ids[:3])) / 3,
        "overlap_top10": len(set(d_ids[:10]) & set(s_ids[:10])) / 10,
        "fused_top1_in_both": float(f_ids[0] in d_ids[:10] and f_ids[0] in s_ids[:10]),
    }
    return feats, f_ids


def auc(x, y):
    """P(signal of a 'not helped' query > signal of a 'helped' query). 0.5 = useless."""
    r = pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()
    pos = np.asarray(y, dtype=bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auc_ci(x, y, n_boot=300, seed=config.SEED):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=bool)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        a = auc(x[i], y[i])
        if not np.isnan(a):
            vals.append(a)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    corpus, queries, qrels = load_scifact()
    index = HybridIndex(corpus)

    labels = pd.read_csv(Path(config.RESULTS_DIR) / "diagnose_gate_dev.csv",
                         dtype={"qid": str})
    rows = []
    for qid in labels["qid"]:
        feats, f_ids = features(index, queries[qid])
        rel = set(qrels[qid])
        first = next((r for r, d in enumerate(f_ids, 1) if d in rel), 999)
        rows.append({"qid": qid, "first_rel_rank": first, **feats})
    df = labels.merge(pd.DataFrame(rows), on="qid")

    sig = ["rel_gap", "std_norm", "entropy", "agreement", "dense_top1", "dense_gap",
           "dense_z", "bm25_top1", "bm25_rel_gap", "bm25_z", "top1_agree",
           "overlap_top3", "overlap_top10", "fused_top1_in_both"]

    print("AUC for predicting 'reranking did NOT help' (>0.5: higher value = safer to skip;")
    print("<0.5: higher value = more likely to need reranking). 95% bootstrap CI shown.\n")
    out = []
    for s in sig:
        a = auc(df[s], df["not_helped"])
        lo, hi = auc_ci(df[s], df["not_helped"])
        out.append((abs(a - 0.5), s, a, lo, hi))
    for _, s, a, lo, hi in sorted(out, reverse=True):
        flag = "  <- CI excludes 0.5" if (lo > 0.5 or hi < 0.5) else ""
        print(f"  {s:20s} AUC={a:.3f}  [{lo:.3f}, {hi:.3f}]{flag}")

    print("\nWhere is the first relevant doc in the HYBRID ranking, by group?")
    bins = pd.cut(df["first_rel_rank"], [0, 1, 3, 5, 10, 10_000],
                  labels=["1", "2-3", "4-5", "6-10", ">10"])
    print(pd.crosstab(df["group"], bins).to_string())

    out_path = Path(config.RESULTS_DIR) / "diagnose_gate_v2_dev.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()