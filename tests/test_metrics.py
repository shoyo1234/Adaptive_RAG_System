import math
import pytest
from evaluation.metrics import ndcg_at_k, mrr_at_k, evaluate_run


def test_perfect_ranking():
    qrel = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "x"], qrel, 10) == pytest.approx(1.0)
    assert mrr_at_k(["a", "b", "x"], qrel, 10) == pytest.approx(1.0)


def test_nothing_relevant_retrieved():
    qrel = {"a": 1}
    assert ndcg_at_k(["x", "y", "z"], qrel, 10) == 0.0
    assert mrr_at_k(["x", "y", "z"], qrel, 10) == 0.0


def test_single_relevant_at_rank_3():
    qrel = {"a": 1}
    # DCG = 1/log2(4) = 0.5, IDCG = 1  ->  0.5
    assert ndcg_at_k(["x", "y", "a"], qrel, 10) == pytest.approx(0.5)
    assert mrr_at_k(["x", "y", "a"], qrel, 10) == pytest.approx(1 / 3)


def test_two_relevant_split_ranks():
    qrel = {"a": 1, "b": 1}
    # ranking [a, x, b]: DCG = 1 + 1/log2(4) = 1.5
    # ideal  [a, b]   : IDCG = 1 + 1/log2(3)
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], qrel, 10) == pytest.approx(expected)


def test_graded_relevance():
    qrel = {"a": 2, "b": 1}
    # ranking [b, a]: DCG = 1 + 2/log2(3);  ideal [a, b]: IDCG = 2 + 1/log2(3)
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], qrel, 10) == pytest.approx(expected)


def test_cutoff_is_respected():
    qrel = {"a": 1}
    ranked = ["x"] * 10 + ["a"]           # relevant doc sits at rank 11
    assert ndcg_at_k(ranked, qrel, 10) == 0.0
    assert mrr_at_k(ranked, qrel, 10) == 0.0
    assert mrr_at_k(ranked, qrel, 11) == pytest.approx(1 / 11)


def test_no_relevant_docs_in_qrel():
    assert ndcg_at_k(["a", "b"], {}, 10) == 0.0


def test_evaluate_run_averages():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": ["a", "x"], "q2": ["x", "b"]}       # MRR: 1.0 and 0.5
    out = evaluate_run(run, qrels, k=10)
    assert out["mrr@10"] == pytest.approx(0.75)