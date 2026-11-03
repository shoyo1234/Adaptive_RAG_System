"""Ranking metrics: NDCG@K and MRR@K."""
import math
from statistics import mean


def _dcg(gains):
    # position i (0-based) is rank i+1, discount = log2(rank + 1)
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids, qrel, k=10):
    """
    ranked_ids: doc IDs, best first.
    qrel: {doc_id: relevance grade} for this query (grade > 0 = relevant).
    """
    gains = [qrel.get(d, 0) for d in ranked_ids[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_ids, qrel, k=10):
    """Reciprocal rank of the first relevant doc within the top k (0 if none)."""
    for rank, d in enumerate(ranked_ids[:k], start=1):
        if qrel.get(d, 0) > 0:
            return 1.0 / rank
    return 0.0


def evaluate_run(run, qrels, k=10):
    """
    run: {query_id: [ranked doc_ids]}
    Returns the mean NDCG@k and MRR@k over the queries in `run`.
    """
    ndcgs = [ndcg_at_k(ids, qrels[q], k) for q, ids in run.items()]
    mrrs = [mrr_at_k(ids, qrels[q], k) for q, ids in run.items()]
    return {f"ndcg@{k}": mean(ndcgs), f"mrr@{k}": mean(mrrs)}