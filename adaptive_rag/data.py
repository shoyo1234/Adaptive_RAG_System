"""Load BEIR SciFact and create a fixed dev/test split of queries."""
import random
from datasets import load_dataset
from adaptive_rag import config


def load_scifact():
    corpus_ds = load_dataset("BeIR/scifact", "corpus")["corpus"]
    queries_ds = load_dataset("BeIR/scifact", "queries")["queries"]
    qrels_ds = load_dataset("BeIR/scifact-qrels")["test"]

    # doc_id -> passage text (title + body)
    corpus = {}
    for row in corpus_ds:
        title = (row["title"] or "").strip()
        text = (row["text"] or "").strip()
        corpus[str(row["_id"])] = f"{title}. {text}" if title else text

    # qid -> {doc_id: relevance grade}; keep only positive labels
    qrels = {}
    for row in qrels_ds:
        if row["score"] > 0:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])

    # only queries that have test labels (SciFact has ~300 of these)
    queries = {
        str(row["_id"]): row["text"]
        for row in queries_ds
        if str(row["_id"]) in qrels
    }
    return corpus, queries, qrels


def split_queries(queries, seed=config.SEED, dev_fraction=config.DEV_FRACTION):
    """Deterministic dev/test split of query IDs."""
    qids = sorted(queries.keys())
    random.Random(seed).shuffle(qids)
    cut = int(len(qids) * dev_fraction)
    return qids[:cut], qids[cut:]      # dev_ids, test_ids


if __name__ == "__main__":
    corpus, queries, qrels = load_scifact()
    dev_ids, test_ids = split_queries(queries)

    print(f"Corpus passages : {len(corpus)}")
    print(f"Queries w/ labels: {len(queries)}")
    print(f"Dev / Test       : {len(dev_ids)} / {len(test_ids)}")
    print(f"Overlap (must be 0): {len(set(dev_ids) & set(test_ids))}")

    qid = dev_ids[0]
    print(f"\nSample query [{qid}]: {queries[qid]}")
    print(f"Relevant doc IDs : {list(qrels[qid].keys())}")
    first_rel = next(iter(qrels[qid]))
    print(f"First relevant passage: {corpus[first_rel][:200]}...")