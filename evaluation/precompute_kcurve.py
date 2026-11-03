"""Per-query NDCG for every rerank depth from ONE cross-encoder pass.

The cross-encoder scores each (query, passage) pair independently, so scoring the
fused top-20 once gives the ranking for every depth k <= 20:
    ranking(k) = sort(fused[:k] by CE score) + fused[k:]

Usage (from repo root): uv run python -m evaluation.precompute_kcurve
"""
import argparse
from pathlib import Path

import pandas as pd

from adaptive_rag import config
from adaptive_rag.data import load_scifact, split_queries
from adaptive_rag.indexes import HybridIndex
from adaptive_rag.rerank import Reranker
from evaluation.diagnose_gate_v2 import features
from evaluation.metrics import ndcg_at_k

KS = (3, 5, 10, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--ce-max-length", type=int,
                    default=config.CROSS_ENCODER_MAX_LENGTH)
    args = ap.parse_args()
    if args.split == "test":
        print("NOTE: test split is for FINAL reporting only. Do not tune on it.\n")

    corpus, queries, qrels = load_scifact()
    dev_ids, test_ids = split_queries(queries)
    qids = dev_ids if args.split == "dev" else test_ids

    index = HybridIndex(corpus)
    rr = Reranker(max_length=args.ce_max_length)
    rr.warm_up()

    rows = []
    for i, qid in enumerate(qids, 1):
        q = queries[qid]
        feats, f_ids = features(index, q)          # gate signals + fused ranking
        ce = dict(rr.rerank(q, f_ids[:max(KS)], corpus))   # {doc_id: ce_score}

        row = {"qid": qid, **feats,
               "first_rel_rank": next((r for r, d in enumerate(f_ids, 1)
                                       if d in qrels[qid]), 999),
               "ndcg_hybrid": ndcg_at_k(f_ids, qrels[qid], 10)}
        for k in KS:
            ranked = sorted(f_ids[:k], key=lambda d: -ce[d]) + f_ids[k:]
            row[f"ndcg_k{k}"] = ndcg_at_k(ranked, qrels[qid], 10)
        rows.append(row)
        if i % 25 == 0:
            print(f"  {i}/{len(qids)}")

    df = pd.DataFrame(rows)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    out = Path(config.RESULTS_DIR) / f"kcurve_{args.split}.csv"
    df.to_csv(out, index=False)

    if args.split == "dev":
        print("\nSanity check, mean NDCG@10 (should match your earlier static runs):")
        print(f"  hybrid {df.ndcg_hybrid.mean():.4f}   (earlier 0.6449)")
        for k, ref in zip(KS, (0.6504, 0.6664, 0.6689, 0.6753)):
            print(f"  k={k:<2}   {df[f'ndcg_k{k}'].mean():.4f}   (earlier {ref})")
    else:
        print("\nTest-split means (report only; do not tune on these):")
        print(f"  hybrid {df.ndcg_hybrid.mean():.4f}")
        for k in KS:
            print(f"  k={k:<2}   {df[f'ndcg_k{k}'].mean():.4f}")
    print(f"\nSaved: {out}")



if __name__ == "__main__":
    main()