# Evaluation protocol (frozen before the first test-split run)

- Data: BEIR SciFact, 300 labeled queries, split 150 dev / 150 test (seed 42).
  All design choices were made on dev only. The test split is used for reporting.
- Candidate pool: top-50 dense (FAISS, all-MiniLM-L6-v2) + top-50 BM25, fused by RRF (k=60).
- Cross-encoder: ms-marco-MiniLM-L-6-v2, inputs truncated to 256 tokens
  (dev NDCG@10: 0.6689 at 256 vs 0.6667 at 512).
- PRIMARY adaptive gate (declared before test): skip reranking if the dense and BM25
  retrievers rank the same passage first (raw scores, before fusion); otherwise rerank
  the fused top-5. No tuned parameters.
- Baselines: hybrid only; static reranking at k = 5, 10, 20.
- Ablations (all reported; none selected on test):
  (a) agreement AND dense top1-top2 gap >= dev 30th percentile (threshold computed from dev);
  (b) dynamic depth (k=5 if the dense gap is confident, else 10);
  (c) single signals: dense gap only, BM25 relative gap only, top-3 overlap only.
- Metrics: NDCG@10, MRR@10, paired bootstrap 95% CI over queries (2000 resamples);
  latency (mean / median / p95) measured with all modes run back-to-back in one session.
- Known limitations: 150 test queries give CIs of roughly +/-0.02 NDCG; single dataset;
  cross-encoder cannot help when the relevant passage is outside the candidate depth.


## Addendum (written after the aggregate test means were seen, before any gate-level test result)

- Aggregate test-split NDCG@10 was observed: hybrid 0.7314, rerank k=3/5/10/20 =
  0.7213/0.7140/0.7054/0.6985. Reranking did not improve on hybrid on the test split.
  The primary gate and the ablations below are NOT changed in response.
- Thresholds for the single-signal ablations, all computed from DEV only:
  (c1) skip if dense gap >= dev median, else rerank top-5;
  (c2) skip if BM25 relative gap >= dev median, else rerank top-5;
  (c3) skip if top-3 overlap between dense and BM25 >= 2/3, else rerank top-5.
- Ablation (a) uses the dev 30th percentile of the dense gap; ablation (b) uses the dev
  median dense gap among non-agreeing queries as its "confident" cutoff.
- A pooled dev+test (300 query) table is reported as a secondary, descriptive result.
  Because the gate design used dev, the pooled gate numbers are optimistic; the test-only
  table is the clean one.
- If a second dataset is added, it will be reported alongside SciFact, not instead of it,
  and the reason (reranking showed no benefit on SciFact test) will be stated.