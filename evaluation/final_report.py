"""Final report per PROTOCOL.md. No models are run.

Usage (from repo root): uv run python -m evaluation.final_report
"""
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_rag import config

R = Path(config.RESULTS_DIR)
N_BOOT = 2000
NDCG_COL = {0: "ndcg_hybrid", 3: "ndcg_k3", 5: "ndcg_k5", 10: "ndcg_k10", 20: "ndcg_k20"}


def boot_ci(diff, seed=config.SEED):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
    return np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])


def dev_thresholds(dev):
    agree = dev["top1_agree"] == 1
    return {
        "dg_p30": dev["dense_gap"].quantile(0.30),
        "dg_p50": dev["dense_gap"].quantile(0.50),
        "bg_p50": dev["bm25_rel_gap"].quantile(0.50),
        "dg_conf": dev.loc[~agree, "dense_gap"].median(),
    }


def rules(df, t):
    """depth per query: 0 = skip (keep hybrid ranking), else rerank that many."""
    n = len(df)
    agree = df["top1_agree"].to_numpy() == 1
    dg, bg = df["dense_gap"].to_numpy(), df["bm25_rel_gap"].to_numpy()
    ov = df["overlap_top3"].to_numpy()
    conf = dg >= t["dg_conf"]
    return {
        "hybrid only": np.zeros(n, int),
        "static k=3": np.full(n, 3),
        "static k=5": np.full(n, 5),
        "static k=10": np.full(n, 10),
        "static k=20": np.full(n, 20),
        "PRIMARY: agree -> skip, else k=5": np.where(agree, 0, 5),
        "(a) agree & dense_gap>=p30 -> skip, else 5": np.where(agree & (dg >= t["dg_p30"]), 0, 5),
        "(b) agree -> skip; else 5 if conf else 10": np.where(agree, 0, np.where(conf, 5, 10)),
        "(c1) dense_gap>=p50 -> skip, else 5": np.where(dg >= t["dg_p50"], 0, 5),
        "(c2) bm25_rel_gap>=p50 -> skip, else 5": np.where(bg >= t["bg_p50"], 0, 5),
        "(c3) overlap_top3>=2/3 -> skip, else 5": np.where(ov >= 0.66, 0, 5),
    }


def per_query_ndcg(df, depth):
    out = np.empty(len(df))
    for k, col in NDCG_COL.items():
        m = depth == k
        out[m] = df.loc[m, col].to_numpy()
    return out


def report(df, t, title, only=None):
    rs = rules(df, t)
    if only:
        rs = {k: v for k, v in rs.items() if k in only}
    ref = {name: per_query_ndcg(df, rs[name]) for name in ("hybrid only", "static k=5", "static k=10")
           if name in rs or not only}
    ref = {n: per_query_ndcg(df, rules(df, t)[n]) for n in ("hybrid only", "static k=5", "static k=10")}
    rows = []
    for name, depth in rs.items():
        nd = per_query_ndcg(df, depth)
        row = {"rule": name, "skip_rate": float((depth == 0).mean()),
               "mean_k": float(depth.mean()), "ndcg@10": nd.mean()}
        for short, key in (("hyb", "hybrid only"), ("s5", "static k=5"), ("s10", "static k=10")):
            lo, hi = boot_ci(nd - ref[key])
            row[f"vs_{short}"] = f"{nd.mean() - ref[key].mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
        rows.append(row)
    if "hybrid only" in rs:                       # upper bound that uses the labels
        orc = np.maximum(df["ndcg_hybrid"], df["ndcg_k5"]).to_numpy()
        rows.append({"rule": "ORACLE (best of hybrid / k=5; uses labels)",
                     "skip_rate": float((df["ndcg_hybrid"] >= df["ndcg_k5"]).mean()),
                     "mean_k": np.nan, "ndcg@10": orc.mean(),
                     **{f"vs_{s}": f"{orc.mean() - ref[k].mean():+.3f}"
                        for s, k in (("hyb", "hybrid only"), ("s5", "static k=5"), ("s10", "static k=10"))}})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(f"\n=== {title} (n={len(df)}) ===")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main():
    dev = pd.read_csv(R / "kcurve_dev.csv", dtype={"qid": str})
    test = pd.read_csv(R / "kcurve_test.csv", dtype={"qid": str})
    t = dev_thresholds(dev)
    print("Dev-derived thresholds:", {k: round(float(v), 4) for k, v in t.items()})

    report(test, t, "TEST split (clean result)")

    pooled = pd.concat([dev, test], ignore_index=True)
    report(pooled, t, "POOLED dev+test (descriptive; gate numbers are optimistic)",
           only={"hybrid only", "static k=5", "static k=10", "PRIMARY: agree -> skip, else k=5"})

    print("\nCI = 95% paired bootstrap over queries (2000 resamples). A difference counts only")
    print("if its interval excludes 0. Latency is measured separately in Task 3.4.")


if __name__ == "__main__":
    main()