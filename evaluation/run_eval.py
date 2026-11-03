"""Evaluate retrieval modes on a query split.

Usage (from repo root):
    uv run python -m evaluation.run_eval --split dev --mode all --tag three_way
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

ALL_MODES = ["dense", "sparse", "hybrid", "static5", "static10", "static20", "adaptive"]


def evaluate_mode(pipeline, mode, qids, queries, qrels, k):
    run, latencies, skips, depths = {}, [], [], []
    for i, qid in enumerate(qids, 1):
        out = pipeline.run(mode, queries[qid])
        run[qid] = out["ranked"]
        latencies.append(out["timings"]["total_ms"])
        if "gate" in out:
            skips.append(out["gate"]["skip"])
            depths.append(out["gate"]["k"])
        if i % 25 == 0:
            print(f"  [{mode}] {i}/{len(qids)}")

    m = evaluate_run(run, qrels, k)
    row = {
        "mode": mode,
        "n_queries": len(qids),
        f"ndcg@{k}": m[f"ndcg@{k}"],
        f"mrr@{k}": m[f"mrr@{k}"],
        "lat_mean_ms": float(np.mean(latencies)),
        "lat_median_ms": float(np.median(latencies)),
        "lat_p95_ms": float(np.percentile(latencies, 95)),
        "skip_rate": float(np.mean(skips)) if skips else np.nan,
        "mean_k": float(np.mean(depths)) if depths else np.nan,
    }
    if mode.startswith("static"):
        row["skip_rate"], row["mean_k"] = 0.0, float(mode[len("static"):])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--mode", default="all",
                    help="dense | sparse | hybrid | staticN | adaptive | all")
    ap.add_argument("--k", type=int, default=config.FINAL_TOP_K)
    ap.add_argument("--ce-max-length", type=int,
                    default=config.CROSS_ENCODER_MAX_LENGTH)
    ap.add_argument("--tag", default="run", help="used in the output CSV name")
    args = ap.parse_args()

    if args.split == "test":
        print("NOTE: test split is for FINAL reporting only. Do not tune on it.\n")

    corpus, queries, qrels = load_scifact()
    dev_ids, test_ids = split_queries(queries)
    qids = dev_ids if args.split == "dev" else test_ids

    index = HybridIndex(corpus)
    reranker = Reranker(max_length=args.ce_max_length)
    pipeline = RetrievalPipeline(index, reranker=reranker)
    pipeline.warm_up()

    modes = ALL_MODES if args.mode == "all" else [args.mode]
    rows = [evaluate_mode(pipeline, m, qids, queries, qrels, args.k) for m in modes]

    df = pd.DataFrame(rows)
    df.insert(1, "ce_max_len", args.ce_max_length)
    pd.set_option("display.width", 250)
    print(f"\nResults on {args.split} split ({len(qids)} queries), k={args.k}")
    print(df.round(4).to_string(index=False))

    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    out_path = Path(config.RESULTS_DIR) / f"day2_{args.tag}_{args.split}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()