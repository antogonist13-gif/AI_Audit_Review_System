"""Formulation improver — LLM-based improvement with deterministic grounding check.

ARCHITECTURAL INVARIANT: This module NEVER imports act_retrieval.
All data arrives via ViolationContext.
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

import config
from llm_cache import cached_llm_call
from models import ImprovedFormulation, RetrievalResult, ViolationContext
from prompts import (
    FORMULATION_TEMPLATE,
    RECOMMENDATION_TEMPLATE,
    parse_improvement_response,
)
from rag_pipeline import get_pipeline
from verifier import _fuzzy_ratio

logger = logging.getLogger(__name__)

# Law reference patterns in qualification text
_LAW_REF_PATTERNS = [
    re.compile(r"ФЗ[-\s]?\d+", re.IGNORECASE),
    re.compile(r"Федеральный закон\s+(?:от\s+[\d.]+\s+)?(?:№\s*)?\d+[-\w]*", re.IGNORECASE),
    re.compile(r"ст(?:атья|\.?)[\s.]*\d+[\w.]*", re.IGNORECASE),
    re.compile(r"п(?:ункт|\.?)[\s.]*\d+[\w.]*", re.IGNORECASE),
    re.compile(r"КоАП\s*РФ", re.IGNORECASE),
    re.compile(r"ГК\s*РФ", re.IGNORECASE),
    re.compile(r"ТК\s*РФ", re.IGNORECASE),
    re.compile(r"Постановление\s+(?:Правительства|РФ)[^,;]{0,50}", re.IGNORECASE),
    re.compile(r"приказ\s+[А-ЯЁ\w]+\s+(?:№\s*)?\d+", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_law_refs_from_text(law: str) -> List[str]:
    """Extract all law reference strings from a qualification text."""
    refs: List[str] = []
    for pattern in _LAW_REF_PATTERNS:
        for m in pattern.finditer(law):
            refs.append(m.group(0).strip())
    return list(set(refs))


def _verify_law_grounding(
    law: str,
    norm_results: List[RetrievalResult],
) -> Tuple[bool, List[str]]:
    """Check if law qualification is grounded in retrieved norm chunks.

    Returns (grounded, matched_chunk_ids).
    """
    if not law.strip():
        return False, []

    refs = _extract_law_refs_from_text(law)
    if not refs:
        return False, []

    if not norm_results:
        return False, []

    matched_ids: List[str] = []
    for ref in refs:
        for r in norm_results:
            ratio = _fuzzy_ratio(ref.lower(), r.text.lower()[:200])
            if ratio >= config.LAW_GROUNDING_FUZZY_MIN:
                chunk_id = r.metadata.get("chunk_id", r.chunk_id)
                matched_ids.append(chunk_id)

    matched_ids = list(set(matched_ids))
    grounded = len(matched_ids) > 0

    logger.info(
        '{"event": "GROUNDING_CHECK", "grounded": %s, "matched_ids": %s}',
        str(grounded).lower(), matched_ids,
    )

    return grounded, matched_ids


def _generate_recommendation(
    ctx: ViolationContext,
    legal_qualification: str,
    norm_context: str,
) -> str:
    """Generate a recommendation for remediation."""
    violation = ctx.violation
    prompt = RECOMMENDATION_TEMPLATE.format(
        violation_description=violation.description,
        subject=violation.subject,
        legal_qualification=legal_qualification,
        norm_context=norm_context,
    )
    try:
        pipeline = get_pipeline()
        response = pipeline.analyze(prompt)
        parsed = parse_improvement_response(response)
        return parsed.get("recommendation", response.strip()[:300])
    except Exception as exc:
        logger.error("Recommendation generation failed: %s", exc)
        return ""


def _build_norm_context(results: List[RetrievalResult]) -> str:
    if not results:
        return "(нет нормативного контекста)"
    return "\n".join(f"{i+1}. {r.text[:250]}" for i, r in enumerate(results))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def improve_formulation(ctx: ViolationContext) -> ImprovedFormulation:
    """Generate an improved formulation for the violation using LLM + grounding check.

    Receives ctx — NEVER calls retrieval.
    """
    violation = ctx.violation
    norm_results = ctx.ref_results.get("norms", [])
    norm_context = _build_norm_context(norm_results)

    prompt = FORMULATION_TEMPLATE.format(
        raw_text=violation.raw_text,
        subject=violation.subject,
        law_ref=violation.law_ref,
        norm_context=norm_context,
    )

    try:
        pipeline = get_pipeline()
        raw_response = pipeline.analyze(prompt)
    except Exception as exc:
        logger.error("Formulation improvement failed for %s: %s", violation.id, exc)
        return ImprovedFormulation(
            improved_text=violation.raw_text,
            legal_qualification=violation.law_ref,
            justification="",
            recommendation="",
            agent_output_raw="",
        )

    parsed = parse_improvement_response(raw_response)

    improved_text = parsed.get("improved_text") or violation.raw_text
    legal_qualification = parsed.get("legal_qualification") or violation.law_ref
    justification = parsed.get("justification", "")

    # Deterministic grounding check
    grounded, matched_ids = _verify_law_grounding(legal_qualification, norm_results)

    if not grounded:
        legal_qualification = legal_qualification + " [не подтверждена]"
        logger.info(
            "GROUNDING_CHECK violation_id=%s grounded=False — marking as unconfirmed",
            violation.id,
        )

    recommendation = _generate_recommendation(ctx, legal_qualification, norm_context)

    return ImprovedFormulation(
        improved_text=improved_text,
        legal_qualification=legal_qualification,
        justification=justification,
        recommendation=recommendation,
        legal_qualification_grounded=grounded,
        norm_source_ids=matched_ids,
        agent_output_raw=raw_response,
    )
