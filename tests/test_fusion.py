import pytest
from adaptive_rag.fusion import rrf

A = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
B = [("b", 0.9), ("c", 0.8), ("d", 0.7)]


def test_hand_computed_scores():
    result = dict(rrf([A, B], k=60))
    assert result["a"] == pytest.approx(1 / 61)
    assert result["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert result["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert result["d"] == pytest.approx(1 / 63)


def test_order():
    assert [d for d, _ in rrf([A, B], k=60)] == ["b", "c", "a", "d"]


def test_doc_in_one_list_still_scores():
    result = dict(rrf([A, B], k=60))
    assert "a" in result and "d" in result          # each appears in only one list


def test_no_duplicates():
    ids = [d for d, _ in rrf([A, B])]
    assert len(ids) == len(set(ids)) == 4


def test_raw_scores_are_ignored():
    scaled = [(d, s * 1000) for d, s in A]
    assert rrf([A, B]) == rrf([scaled, B])


def test_deterministic_tie_break():
    # x and y have identical RRF scores (rank 1 in one list each)
    out1 = rrf([[("y", 1.0)], [("x", 1.0)]])
    out2 = rrf([[("x", 1.0)], [("y", 1.0)]])
    assert [d for d, _ in out1] == [d for d, _ in out2] == ["x", "y"]


def test_empty_input():
    assert rrf([]) == []
    assert rrf([[], []]) == []