"""Two-lever gate simulation (skip + rerank depth) from results/kcurve_dev.csv.
No models are run.  Usage: uv run python -m evaluation.simulate_gate_v2
"""
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_rag import config

# Mean total latency (ms) per rerank depth, MEASURED on dev earlier (0 = hybrid only).
COST = {0: 71.0, 3: 374.0, 5: 652.0, 10: 1159.0, 20: 2587.0}


def boot_ci(diff, n_boot=2000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def evaluate(df, depth):
    """depth[i] = rerank depth for query i (0 = skip, keep hybrid ranking)."""
    ndcg = np.empty(len(df))
    for k in (0, 3, 5, 10, 20):
        m = depth == k
        col = "ndcg_hybrid" if k == 0 else f"ndcg_k{k}"
        ndcg[m] = df.loc[m, col].to_numpy()
    lat = np.array([COST[int(d)] for d in depth])
    return ndcg, lat


def main():
    df = pd.read_csv(Path(config.RESULTS_DIR) / "kcurve_dev.csv", dtype={"qid": str})
    n = len(df)
    agree = df["top1_agree"].to_numpy() == 1
    dg = df["dense_gap"].to_numpy()
    t30 = np.quantile(dg, 0.30)
    med_unskipped = np.median(dg[~agree])

    def fixed(k):
        return np.full(n, k)

    def skip_then(skip, k):
        return np.where(skip, 0, k)

    def skip_dynamic(skip, k_conf, k_unsure):
        conf = dg >= med_unskipped          # larger raw dense gap = more confident
        return np.where(skip, 0, np.where(conf, k_conf, k_unsure))

    rules = [
        ("static5", fixed(5)),
        ("static10", fixed(10)),
        ("static20", fixed(20)),
        ("agree -> skip, else k=5", skip_then(agree, 5)),
        ("agree -> skip, else k=10", skip_then(agree, 10)),
        ("agree -> skip, else k=20", skip_then(agree, 20)),
        (f"agree & dense_gap>=p30 -> skip, else k=5", skip_then(agree & (dg >= t30), 5)),
        (f"agree & dense_gap>=p30 -> skip, else k=10", skip_then(agree & (dg >= t30), 10)),
        ("agree -> skip; else k=5 if confident else 10", skip_dynamic(agree, 5, 10)),
        ("agree -> skip; else k=10 if confident else 20", skip_dynamic(agree, 10, 20)),
    ]

    s5, _ = evaluate(df, fixed(5))
    s10, _ = evaluate(df, fixed(10))
    rows = []
    for name, depth in rules:
        ndcg, lat = evaluate(df, depth)
        lo5, hi5 = boot_ci(ndcg - s5)
        lo10, hi10 = boot_ci(ndcg - s10)
        rows.append({
            "rule": name,
            "skip_rate": float((depth == 0).mean()),
            "ndcg@10": ndcg.mean(),
            "vs_s5": ndcg.mean() - s5.mean(), "vs_s5_ci": f"[{lo5:+.3f},{hi5:+.3f}]",
            "vs_s10": ndcg.mean() - s10.mean(), "vs_s10_ci": f"[{lo10:+.3f},{hi10:+.3f}]",
            "est_lat_ms": lat.mean(),
            "speedup_vs_s10": COST[10] / lat.mean(),
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    print("\nCI = 95% paired bootstrap over queries. A difference only counts if its")
    print("interval excludes 0. Latency is estimated from measured per-depth means;")
    print("the dense_gap threshold is chosen on these same queries (optimistic).")
    out.to_csv(Path(config.RESULTS_DIR) / "simulate_gate_v2_dev.csv", index=False)


if __name__ == "__main__":
    main()