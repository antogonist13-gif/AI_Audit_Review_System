"""Violation evaluator — three parallel LLM evaluations using pre-fetched context.

ARCHITECTURAL INVARIANT: This module NEVER imports act_retrieval.
All data arrives via ViolationContext.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import List, Tuple

import config
from models import RetrievalResult, Violation, ViolationContext
from prompts import (
    ACTIONABILITY_EVAL_TEMPLATE,
    EVIDENCE_EVAL_TEMPLATE,
    LEGAL_EVAL_TEMPLATE,
    parse_scored_response,
)
from rag_pipeline import get_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_context(results: List[RetrievalResult]) -> str:
    """Format retrieval results as a numbered source list."""
    if not results:
        return "(нет данных)"
    lines = []
    for i, r in enumerate(results, start=1):
        law = r.metadata.get("law", "")
        law_str = f" [{law}]" if law else ""
        lines.append(f"{i}. {r.text[:300].strip()}{law_str}")
    return "\n".join(lines)


def _evaluate_evidence(
    violation: Violation,
    corpus_context: str,
    norm_context: str,
) -> Tuple[float, str]:
    """Evaluate evidence sufficiency via LLM."""
    prompt = EVIDENCE_EVAL_TEMPLATE.format(
        violation_description=violation.description,
        subject=violation.subject,
        norm_context=norm_context,
        corpus_context=corpus_context,
    )
    try:
        pipeline = get_pipeline()
        response = pipeline.analyze(prompt)
        parsed = parse_scored_response(response)
        score = float(parsed["score"])
        comment = parsed["comment"]
        used_fallback = (score == 0.0)
        logger.info(
            "SCORE_PARSE_RESULT violation_id=%s axis=evidence score=%.3f used_fallback=%s",
            violation.id, score, used_fallback,
        )
        return score, comment
    except Exception as exc:
        logger.error("Evidence evaluation failed for %s: %s", violation.id, exc)
        return 0.0, str(exc)


def _evaluate_legal(
    violation: Violation,
    norm_context: str,
) -> Tuple[float, str]:
    """Evaluate legal correctness via LLM."""
    prompt = LEGAL_EVAL_TEMPLATE.format(
        violation_description=violation.description,
        law_ref=violation.law_ref,
        norm_context=norm_context,
    )
    try:
        pipeline = get_pipeline()
        response = pipeline.analyze(prompt)
        parsed = parse_scored_response(response)
        score = float(parsed["score"])
        comment = parsed["comment"]
        used_fallback = (score == 0.0)
        logger.info(
            "SCORE_PARSE_RESULT violation_id=%s axis=legal score=%.3f used_fallback=%s",
            violation.id, score, used_fallback,
        )
        return score, comment
    except Exception as exc:
        logger.error("Legal evaluation failed for %s: %s", violation.id, exc)
        return 0.0, str(exc)


def _evaluate_actionability(
    violation: Violation,
) -> Tuple[float, str]:
    """Evaluate actionability via LLM."""
    prompt = ACTIONABILITY_EVAL_TEMPLATE.format(
        violation_description=violation.description,
        subject=violation.subject,
        law_ref=violation.law_ref,
    )
    try:
        pipeline = get_pipeline()
        response = pipeline.analyze(prompt)
        parsed = parse_scored_response(response)
        score = float(parsed["score"])
        comment = parsed["comment"]
        used_fallback = (score == 0.0)
        logger.info(
            "SCORE_PARSE_RESULT violation_id=%s axis=actionability score=%.3f used_fallback=%s",
            violation.id, score, used_fallback,
        )
        return score, comment
    except Exception as exc:
        logger.error("Actionability evaluation failed for %s: %s", violation.id, exc)
        return 0.0, str(exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_violation(ctx: ViolationContext) -> Violation:
    """Run three parallel LLM evaluations and return a scored Violation.

    Receives ctx — NEVER calls retrieval.
    """
    violation = ctx.violation

    norm_context = _build_context(ctx.ref_results.get("norms", []))
    corpus_context = _build_context(ctx.ref_results.get("typical", []))

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_evidence = executor.submit(_evaluate_evidence, violation, corpus_context, norm_context)
        future_legal = executor.submit(_evaluate_legal, violation, norm_context)
        future_actionability = executor.submit(_evaluate_actionability, violation)

        evidence_score, evidence_comment = future_evidence.result()
        legal_score, legal_comment = future_legal.result()
        actionability_score, actionability_comment = future_actionability.result()

    # Clamp all scores to [0, 1]
    evidence_score = min(1.0, max(0.0, evidence_score))
    legal_score = min(1.0, max(0.0, legal_score))
    actionability_score = min(1.0, max(0.0, actionability_score))

    logger.info(
        '{"event": "EVALUATION_DONE", "violation_id": "%s", "evidence": %.2f, "legal": %.2f, "actionability": %.2f}',
        violation.id, evidence_score, legal_score, actionability_score,
    )

    return replace(
        violation,
        evidence_score=evidence_score,
        legal_score=legal_score,
        actionability_score=actionability_score,
        evidence_comment=evidence_comment,
        legal_comment=legal_comment,
        actionability_comment=actionability_comment,
    )
