import pytest
from adaptive_rag.gate import GateParams, compute_signals, decide

DENSE = [("a", 0.80), ("b", 0.60), ("c", 0.50)]          # dense gap = 0.20
SPARSE_AGREE = [("a", 10.0), ("c", 5.0), ("b", 4.0)]     # BM25 rel. gap = (10-5)/10 = 0.5
SPARSE_DISAGREE = [("c", 10.0), ("a", 9.0), ("b", 4.0)]


def test_signals_hand_computed():
    s = compute_signals(DENSE, SPARSE_AGREE)
    assert s["top1_agree"] is True
    assert s["dense_gap"] == pytest.approx(0.20)
    assert s["bm25_rel_gap"] == pytest.approx(0.5)


def test_disagreement_detected():
    assert compute_signals(DENSE, SPARSE_DISAGREE)["top1_agree"] is False


def test_skip_when_top1_agree():
    d = decide(DENSE, SPARSE_AGREE)
    assert d.skip and d.k == 0


def test_rerank_top5_when_disagree():
    d = decide(DENSE, SPARSE_DISAGREE)
    assert not d.skip and d.k == 5


def test_dense_gap_condition():
    assert not decide(DENSE, SPARSE_AGREE, GateParams(dense_gap_min=0.30)).skip  # 0.20 < 0.30
    assert decide(DENSE, SPARSE_AGREE, GateParams(dense_gap_min=0.10)).skip      # 0.20 >= 0.10


def test_force_overrides():
    assert decide(DENSE, SPARSE_DISAGREE, GateParams(force="skip")).skip
    d = decide(DENSE, SPARSE_AGREE, GateParams(force="rerank", k_rerank=10))
    assert not d.skip and d.k == 10


def test_dynamic_depth():
    d = decide(DENSE, SPARSE_DISAGREE, GateParams(conf_gap=0.30, k_unsure=10))
    assert not d.skip and d.k == 10                       # gap 0.20 < 0.30: unsure
    d = decide(DENSE, SPARSE_DISAGREE, GateParams(conf_gap=0.10, k_unsure=10))
    assert d.k == 5                                       # gap 0.20 >= 0.10: confident


def test_short_lists_do_not_crash():
    assert decide([("a", 0.5)], [("a", 3.0)]).skip        # one result each, agree
    assert decide([], []).skip is False                   # nothing to trust: rerank