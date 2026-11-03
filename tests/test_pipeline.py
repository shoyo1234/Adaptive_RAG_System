import pytest

from adaptive_rag.data import load_scifact, split_queries
from adaptive_rag.gate import GateParams
from adaptive_rag.indexes import HybridIndex
from adaptive_rag.pipeline import RetrievalPipeline
from adaptive_rag.rerank import Reranker


@pytest.fixture(scope="module")
def setup():
    corpus, queries, _ = load_scifact()
    dev_ids, _ = split_queries(queries)
    pipe = RetrievalPipeline(HybridIndex(corpus), reranker=Reranker())
    pipe.warm_up()
    return pipe, [queries[q] for q in dev_ids[:8]]


def test_forced_skip_equals_hybrid(setup):
    pipe, qs = setup
    for q in qs:
        out = pipe.run_adaptive(q, GateParams(force="skip"))
        assert out["ranked"] == pipe.run_hybrid(q)["ranked"]


@pytest.mark.parametrize("k", [5, 10])
def test_forced_rerank_equals_static(setup, k):
    pipe, qs = setup
    for q in qs:
        out = pipe.run_adaptive(q, GateParams(force="rerank", k_rerank=k))
        assert out["ranked"] == pipe.run_static(q, k)["ranked"]


def test_default_gate_matches_its_own_decision(setup):
    """Whatever the gate decides, the ranking must equal the matching baseline."""
    pipe, qs = setup
    for q in qs:
        out = pipe.run_adaptive(q)
        g = out["gate"]
        expected = (pipe.run_hybrid(q)["ranked"] if g["skip"]
                    else pipe.run_static(q, g["k"])["ranked"])
        assert out["ranked"] == expected


def test_skip_spends_no_rerank_time(setup):
    pipe, qs = setup
    out = pipe.run_adaptive(qs[0], GateParams(force="skip"))
    assert out["timings"]["rerank_ms"] == 0.0