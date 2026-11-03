"""Timed run: every mode on every query, interleaved, repeated.

Rotating the mode order per query spreads background-load drift evenly across modes.
Rankings from pass 1 are also scored, as a consistency check against final_report.

Usage (from repo root): uv run python -m evaluation.timing --split test --repeats 3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_rag import config
from adaptive_rag.data import load_scifact, split_queries
from adaptive_rag.indexes import HybridIndex
from adaptive_rag.pipeline import RetrievalPipeline
from adaptive_rag.rerank import Reranker
from evaluation.metrics import evaluate_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--modes", default="hybrid,static5,static10,adaptive")
    ap.add_argument("--ce-max-length", type=int, default=config.CROSS_ENCODER_MAX_LENGTH)
    ap.add_argument("--tag", default="timing")
    args = ap.parse_args()
    modes = args.modes.split(",")

    corpus, queries, qrels = load_scifact()
    dev_ids, test_ids = split_queries(queries)
    qids = dev_ids if args.split == "dev" else test_ids

    pipe = RetrievalPipeline(HybridIndex(corpus),
                             reranker=Reranker(max_length=args.ce_max_length))
    pipe.warm_up()

    lat = {m: [[] for _ in range(args.repeats)] for m in modes}
    rank = {m: {} for m in modes}
    skips = []
    for r in range(args.repeats):
        for i, qid in enumerate(qids):
            s = i % len(modes)
            for m in modes[s:] + modes[:s]:             # rotating order
                out = pipe.run(m, queries[qid])
                lat[m][r].append(out["timings"]["total_ms"])
                if r == 0:
                    rank[m][qid] = out["ranked"]
                    if m == "adaptive":
                        skips.append(out["gate"]["skip"])
        print(f"  pass {r + 1}/{args.repeats} done")

    rows = []
    for m in modes:
        allv = np.concatenate([np.array(x) for x in lat[m]])
        met = evaluate_run(rank[m], qrels, 10)
        rows.append({
            "mode": m, "ndcg@10": met["ndcg@10"], "mrr@10": met["mrr@10"],
            "lat_mean_ms": allv.mean(), "lat_median_ms": float(np.median(allv)),
            "lat_p95_ms": float(np.percentile(allv, 95)),
            "mean_by_pass": "/".join(f"{np.mean(x):.0f}" for x in lat[m]),
            "skip_rate": float(np.mean(skips)) if m == "adaptive" else np.nan,
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"\nTimed run on {args.split} split ({len(qids)} queries, {args.repeats} passes)")
    print(df.round(4).to_string(index=False))

    mean = df.set_index("mode")["lat_mean_ms"]
    if "adaptive" in mean:
        for ref in ("static5", "static10", "static20", "hybrid"):
            if ref in mean:
                print(f"  adaptive vs {ref}: {mean[ref] / mean['adaptive']:.2f}x "
                      f"{'faster' if mean[ref] > mean['adaptive'] else 'slower'} (mean latency)")

    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    out_path = Path(config.RESULTS_DIR) / f"{args.tag}_{args.split}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()