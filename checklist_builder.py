"""Checklist builder — assembles the final ChecklistItem with deterministic statuses.

Status assignment is always based on float scores, never on LLM text.
Verifier results can override LLM-derived statuses.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Optional

import config
from models import (
    ChecklistItem,
    ComparisonResult,
    ImprovedFormulation,
    ItemTrace,
    VerificationResult,
    Violation,
    ViolationContext,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _score_to_status(
    score: Optional[float],
    lo: float,
    hi: float,
    labels: tuple = ("insufficient", "unclear", "sufficient"),
) -> str:
    """Map a float score to a status label using deterministic thresholds.

    score < lo  → labels[0]   (default: 'insufficient')
    lo ≤ score < hi → labels[1]  (default: 'unclear')
    score ≥ hi  → labels[2]  (default: 'sufficient')

    Never reads LLM text. Handles None gracefully.
    """
    if score is None:
        return "unknown"
    if score < lo:
        return labels[0]
    if score < hi:
        return labels[1]
    return labels[2]


def _compute_confidence(
    violation: Violation,
    comparisons: Dict[str, List[ComparisonResult]],
) -> float:
    """Compute confidence score as weighted average of all component scores.

    Returns a value in [0, 1]. Handles None scores gracefully.
    """
    w = config.EVIDENCE_WEIGHTS

    evidence = violation.evidence_score if violation.evidence_score is not None else 0.0
    legal = violation.legal_score if violation.legal_score is not None else 0.0
    actionability = violation.actionability_score if violation.actionability_score is not None else 0.0

    all_comps = [c for comps in comparisons.values() for c in comps]
    if all_comps:
        similarity = sum(c.similarity_score for c in all_comps) / len(all_comps)
    else:
        similarity = 0.0

    confidence = (
        w["evidence"] * evidence
        + w["legal"] * legal
        + w["actionability"] * actionability
        + w["similarity"] * similarity
    )
    return min(1.0, max(0.0, confidence))


def _apply_verifier_overrides(
    item: ChecklistItem,
    verif_notes: List[VerificationResult],
) -> ChecklistItem:
    """Apply verifier overrides — verifier beats LLM when evidence is insufficient."""
    for note in verif_notes:
        if note.axis == "evidence" and note.status == "unconfirmed":
            if note.detail == "insufficient":
                logger.info(
                    '{"event": "VERIFIER_OVERRIDE", "violation_id": "%s", "axis": "evidence", '
                    '"from": "%s", "to": "insufficient"}',
                    item.violation_id, item.evidence_status,
                )
                item = replace(
                    item,
                    evidence_status="insufficient",
                    possibly_not_a_violation=True,
                )
    return item


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_checklist_item(
    ctx: ViolationContext,
    scored_violation: Violation,
    comparisons: Dict[str, List[ComparisonResult]],
    improvement: ImprovedFormulation,
    agent_outputs: Dict[str, str],
) -> ChecklistItem:
    """Assemble the final ChecklistItem from all evaluated components.

    Statuses are determined deterministically from float scores.
    Verifier overrides are applied last.
    """
    assert scored_violation.evidence_score is not None, (
        f"scored_violation not evaluated for {scored_violation.id}. "
        "ensure evaluate_violation() ran before build_checklist_item()"
    )

    v = scored_violation
    violation_id = v.id

    # Deterministic status assignment from scores
    evidence_status = _score_to_status(
        v.evidence_score,
        lo=config.EVIDENCE_SUFFICIENT_THRESHOLD,
        hi=config.EVIDENCE_SUFFICIENT_THRESHOLD + 0.2,
    )
    legal_status = _score_to_status(
        v.legal_score,
        lo=config.LEGAL_CORRECT_THRESHOLD,
        hi=config.LEGAL_CORRECT_THRESHOLD + 0.2,
    )
    actionability_status = _score_to_status(
        v.actionability_score,
        lo=config.ACTIONABILITY_THRESHOLD,
        hi=config.ACTIONABILITY_THRESHOLD + 0.2,
    )

    confidence = _compute_confidence(v, comparisons)

    possibly_not_a_violation = (
        v.possibly_not_a_violation
        or (v.evidence_score < config.EVIDENCE_SUFFICIENT_THRESHOLD)
    )

    # Verifier notes from context
    from verifier import verify_checklist_item
    verif_notes = verify_checklist_item(v, ctx.ref_results)

    # Build trace
    trace = ItemTrace(
        violation_id=violation_id,
        used_chunk_ids=ctx.used_chunk_ids,
        retrieval_queries=ctx.retrieval_queries,
        evidence_sources=[
            c.match_id for c in comparisons.get("typical", [])
        ],
        norm_sources=[
            c.match_id for c in comparisons.get("norms", [])
        ],
        typical_sources=[
            c.match_id for c in comparisons.get("typical", [])
        ],
        verifier_notes=[
            f"{n.axis}:{n.status}:{n.detail}" for n in verif_notes
        ],
    )

    item = ChecklistItem(
        violation_id=violation_id,
        raw_text=v.raw_text,
        description=v.description,
        subject=v.subject,
        law_ref=v.law_ref,
        source_document=v.source_document,
        page=v.page,
        section=v.section,
        evidence_score=v.evidence_score,
        legal_score=v.legal_score,
        actionability_score=v.actionability_score,
        confidence_score=confidence,
        evidence_status=evidence_status,
        legal_status=legal_status,
        actionability_status=actionability_status,
        evidence_comment=v.evidence_comment,
        legal_comment=v.legal_comment,
        actionability_comment=v.actionability_comment,
        improved_formulation=improvement.improved_text,
        legal_qualification=improvement.legal_qualification,
        legal_qualification_grounded=improvement.legal_qualification_grounded,
        justification=improvement.justification,
        recommendation=improvement.recommendation,
        comparisons=comparisons,
        verification_notes=verif_notes,
        agent_outputs=agent_outputs,
        possibly_not_a_violation=possibly_not_a_violation,
        trace=trace,
    )

    item = _apply_verifier_overrides(item, verif_notes)

    logger.info(
        '{"event": "CHECKLIST_ITEM_BUILT", "violation_id": "%s", "confidence": %.2f, '
        '"possibly_violation": %s}',
        violation_id, confidence, str(not possibly_not_a_violation).lower(),
    )

    return item
