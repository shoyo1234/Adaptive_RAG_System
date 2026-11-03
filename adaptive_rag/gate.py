"""Adaptive gate: decide whether to rerank, and how deep, from RAW retrieval scores.

Why raw scores: RRF keeps only ranks, which flattens the confidence information
(see evaluation/diagnose_gate_v2.py), so the gate reads the dense and BM25 result
lists BEFORE fusion. Signal definitions match those used in the diagnostics.
"""
from dataclasses import dataclass, field

from adaptive_rag import config


@dataclass(frozen=True)
class GateParams:
    require_agreement: bool = True       # skip only if dense and BM25 pick the same top-1
    dense_gap_min: float = 0.0           # extra condition: dense top1-top2 gap >= this
    k_rerank: int = config.GATE_K_RERANK  # rerank depth when not skipped
    conf_gap: float | None = None        # optional dynamic depth (ablation): if the dense
    k_unsure: int | None = None          #   gap is below conf_gap, rerank k_unsure instead
    force: str | None = None             # None | "skip" | "rerank" (used by parity tests)


@dataclass(frozen=True)
class GateDecision:
    skip: bool
    k: int                               # rerank depth; 0 when skipped
    signals: dict = field(default_factory=dict)


def compute_signals(dense, sparse):
    """dense / sparse: [(doc_id, raw_score)] best-first, BEFORE fusion."""
    d = [s for _, s in dense]
    b = [s for _, s in sparse]
    dense_gap = d[0] - d[1] if len(d) > 1 else 0.0
    bm25_rel_gap = (b[0] - b[1]) / b[0] if len(b) > 1 and b[0] > 0 else 0.0
    top1_agree = bool(dense and sparse and dense[0][0] == sparse[0][0])
    return {"top1_agree": top1_agree,
            "dense_gap": float(dense_gap),
            "bm25_rel_gap": float(bm25_rel_gap)}


def decide(dense, sparse, params=GateParams()):
    sig = compute_signals(dense, sparse)

    if params.force == "skip":
        return GateDecision(True, 0, sig)
    if params.force == "rerank":
        return GateDecision(False, params.k_rerank, sig)

    confident = ((sig["top1_agree"] or not params.require_agreement)
                 and sig["dense_gap"] >= params.dense_gap_min)
    if confident:
        return GateDecision(True, 0, sig)

    k = params.k_rerank
    if (params.conf_gap is not None and params.k_unsure is not None
            and sig["dense_gap"] < params.conf_gap):
        k = params.k_unsure
    return GateDecision(False, k, sig)  