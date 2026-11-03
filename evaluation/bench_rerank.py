"""Find a fast, fair cross-encoder setting. Run from repo root."""
import time
import numpy as np
import torch

from adaptive_rag.data import load_scifact, split_queries
from adaptive_rag.fusion import rrf
from adaptive_rag.indexes import HybridIndex
from adaptive_rag.rerank import Reranker

corpus, queries, qrels = load_scifact()
dev_ids, _ = split_queries(queries)
index = HybridIndex(corpus)
sample = dev_ids[:10]

# candidate lists prepared once, so only reranking is timed
cands = {}
for sid in sample:
    q = queries[sid]
    cands[sid] = [d for d, _ in rrf([index.dense_search(q), index.sparse_search(q)])]

print(f"torch threads (default): {torch.get_num_threads()}")
tok = Reranker(max_length=512).model.tokenizer
lens = [len(tok(corpus[d])["input_ids"]) for d in cands[sample[0]][:50]]
print(f"passage token lengths (top-50, 1 query): median {int(np.median(lens))}, "
      f"max {max(lens)}\n")

for max_len in (512, 256):
    for threads in (torch.get_num_threads(), 4):
        torch.set_num_threads(threads)
        rr = Reranker(max_length=max_len)
        rr.warm_up()
        times = []
        for sid in sample:
            t0 = time.perf_counter()
            rr.rerank(queries[sid], cands[sid][:10], corpus)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"max_length={max_len}  threads={threads}: "
              f"n=10 mean {np.mean(times):7.1f} ms")