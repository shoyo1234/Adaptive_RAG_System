"""Retrieval pipeline modes with per-stage latency measurement."""
import time

from adaptive_rag.fusion import rrf
from adaptive_rag.gate import GateParams, decide


def _ms(t0):
    return (time.perf_counter() - t0) * 1000.0


class RetrievalPipeline:
    def __init__(self, index, reranker=None):
        self.index = index
        self.reranker = reranker
        self.id_to_text = dict(zip(index.doc_ids, index.texts))

    def warm_up(self):
        """First calls are slower (lazy init). Run once before any timing."""
        for _ in range(3):
            self.index.dense_search("warm up query")
            self.index.sparse_search("warm up query")
        if self.reranker is not None:
            self.reranker.warm_up()

    # ---------- single-stream references ----------
    def run_dense(self, query):
        t0 = time.perf_counter()
        res = self.index.dense_search(query)
        total = _ms(t0)
        return {"ranked": [d for d, _ in res], "pool": res,
                "timings": {"dense_ms": total, "total_ms": total}}

    def run_sparse(self, query):
        t0 = time.perf_counter()
        res = self.index.sparse_search(query)
        total = _ms(t0)
        return {"ranked": [d for d, _ in res], "pool": res,
                "timings": {"sparse_ms": total, "total_ms": total}}

    # ---------- Baseline 1: hybrid ----------
    def run_hybrid(self, query):
        t_all = time.perf_counter()

        t0 = time.perf_counter()
        dense = self.index.dense_search(query)
        dense_ms = _ms(t0)

        t0 = time.perf_counter()
        sparse = self.index.sparse_search(query)
        sparse_ms = _ms(t0)

        t0 = time.perf_counter()
        fused = rrf([dense, sparse])
        fusion_ms = _ms(t0)

        return {
            "ranked": [d for d, _ in fused],
            "pool": fused,                    # [(doc_id, rrf_score)]
            "dense": dense,                   # raw lists: the gate reads these
            "sparse": sparse,
            "timings": {"dense_ms": dense_ms, "sparse_ms": sparse_ms,
                        "fusion_ms": fusion_ms, "total_ms": _ms(t_all)},
        }

    # ---------- shared helper ----------
    def _rerank_head(self, query, fused, k):
        """Rerank the top-k fused passages; the rest keep their fused order."""
        head = [d for d, _ in fused[:k]]
        tail = [d for d, _ in fused[k:]]
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query, head, self.id_to_text)
        rerank_ms = _ms(t0)
        return [d for d, _ in reranked] + tail, reranked, rerank_ms

    # ---------- Baseline 2: static reranking ----------
    def run_static(self, query, rerank_k):
        """Hybrid, then ALWAYS rerank the top rerank_k fused passages."""
        t_all = time.perf_counter()
        out = self.run_hybrid(query)

        ranked, reranked, rerank_ms = self._rerank_head(query, out["pool"], rerank_k)

        timings = dict(out["timings"])
        timings["rerank_ms"] = rerank_ms
        timings["total_ms"] = _ms(t_all)
        return {"ranked": ranked, "pool": out["pool"],
                "reranked": reranked, "timings": timings}

    # ---------- Adaptive gating ----------
    def run_adaptive(self, query, params=None):
        """Hybrid, then the gate decides: skip reranking, or rerank the top-k."""
        params = params or GateParams()
        t_all = time.perf_counter()
        out = self.run_hybrid(query)
        fused = out["pool"]

        t0 = time.perf_counter()
        decision = decide(out["dense"], out["sparse"], params)
        gate_ms = _ms(t0)

        timings = dict(out["timings"])
        timings["gate_ms"] = gate_ms
        if decision.skip:
            ranked, reranked = [d for d, _ in fused], None
            timings["rerank_ms"] = 0.0
        else:
            ranked, reranked, timings["rerank_ms"] = self._rerank_head(
                query, fused, decision.k)
        timings["total_ms"] = _ms(t_all)

        return {"ranked": ranked, "pool": fused, "reranked": reranked,
                "timings": timings,
                "gate": {"skip": decision.skip, "k": decision.k, **decision.signals}}

    # ---------- dispatcher ----------
    def run(self, mode, query):
        if mode.startswith("static"):                    # "static5", "static10", ...
            return self.run_static(query, int(mode[len("static"):]))
        if mode == "adaptive":
            return self.run_adaptive(query)
        return {"dense": self.run_dense,
                "sparse": self.run_sparse,
                "hybrid": self.run_hybrid}[mode](query)