"""Reciprocal Rank Fusion."""
from adaptive_rag import config


def rrf(ranked_lists, k=config.RRF_K):
    """
    Fuse several ranked lists into one.

    ranked_lists: list of lists of (doc_id, raw_score), each already sorted
                  best-first. Raw scores are IGNORED; only rank matters.
    Returns: list of (doc_id, rrf_score), sorted best-first.
    """
    fused = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):   # rank starts at 1
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by score descending; tie-break on doc_id so results are deterministic.
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


if __name__ == "__main__":
    from adaptive_rag.data import load_scifact, split_queries
    from adaptive_rag.indexes import HybridIndex

    corpus, queries, qrels = load_scifact()
    dev_ids, _ = split_queries(queries)
    index = HybridIndex(corpus)

    qid = dev_ids[0]
    dense = index.dense_search(queries[qid])
    sparse = index.sparse_search(queries[qid])
    fused = rrf([dense, sparse])

    print(f"\nQuery [{qid}]: {queries[qid]}")
    print(f"Relevant: {list(qrels[qid])}")
    print(f"Fused pool size: {len(fused)} (max possible {len(dense) + len(sparse)})")
    for rank, (d, s) in enumerate(fused[:5], 1):
        print(f"  {rank}. {d}  rrf={s:.5f}  {corpus[d][:70]}...")

    # Candidate-pool recall: reranking can only fix what is in the pool.
    def pool_recall(pool_fn):
        return sum(bool(set(qrels[q]) & {d for d, _ in pool_fn(q)}) for q in dev_ids) / len(dev_ids)

    print(f"\nPool recall over {len(dev_ids)} dev queries:")
    print(f"  Dense only : {pool_recall(lambda q: index.dense_search(queries[q])):.3f}")
    print(f"  Sparse only: {pool_recall(lambda q: index.sparse_search(queries[q])):.3f}")
    print(f"  Fused pool : {pool_recall(lambda q: rrf([index.dense_search(queries[q]), index.sparse_search(queries[q])])):.3f}")