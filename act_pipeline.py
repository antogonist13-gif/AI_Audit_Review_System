"""Act pipeline — main orchestrator.

This is the ONLY module that calls fetch_violation_context.
Each violation is processed exactly once in _process_violation.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from act_retrieval import fetch_violation_context
from act_preprocessor import preprocess_act, violations_with_context_to_violations
from checklist_builder import build_checklist_item
from comparison_engine import compare_violation
from formulation_improver import improve_formulation
from loader import load_document
from models import ChecklistItem, Violation, ViolationContext
from violation_evaluator import evaluate_violation
from violation_normalizer import normalize_violation

logger = logging.getLogger(__name__)

# Pipeline run statistics
pipeline_stats: Dict[str, object] = {
    "violations_total": 0,
    "violations_processed": 0,
    "violations_failed": 0,
    "retrieval_calls": 0,
    "llm_cache_hits": 0,
    "llm_cache_misses": 0,
    "parser_llm_fallbacks": 0,
    "grounding_failures": 0,
    "verifier_overrides": 0,
    "possibly_not_violations": 0,
    "avg_confidence": 0.0,
}


def _reset_stats() -> None:
    for key in pipeline_stats:
        pipeline_stats[key] = 0 if key != "avg_confidence" else 0.0


def _update_avg_confidence(items: List[ChecklistItem]) -> None:
    if items:
        pipeline_stats["avg_confidence"] = sum(i.confidence_score for i in items) / len(items)


# ---------------------------------------------------------------------------
# Core processing unit
# ---------------------------------------------------------------------------


def _process_violation(violation: Violation) -> Optional[ChecklistItem]:
    """Process a single violation through the full pipeline.

    This is the ONLY place that calls fetch_violation_context.
    Returns None if any step fails (does not crash the pipeline).
    """
    try:
        # STEP 1: Single retrieval — the only fetch_violation_context call
        ctx: ViolationContext = fetch_violation_context(violation)
        pipeline_stats["retrieval_calls"] = int(pipeline_stats["retrieval_calls"]) + 1

        # STEP 2: Parallel evaluation + comparison (both use ctx, no retrieval)
        with ThreadPoolExecutor(max_workers=2) as inner:
            f_eval = inner.submit(evaluate_violation, ctx)
            f_compare = inner.submit(compare_violation, ctx)

        scored_violation = f_eval.result()
        comparisons = f_compare.result()

        # STEP 3: Formulation improvement (uses ctx, no retrieval)
        improvement = improve_formulation(ctx)

        if not improvement.legal_qualification_grounded:
            pipeline_stats["grounding_failures"] = int(pipeline_stats["grounding_failures"]) + 1

        # STEP 4: Assembly
        item = build_checklist_item(
            ctx=ctx,
            scored_violation=scored_violation,
            comparisons=comparisons,
            improvement=improvement,
            agent_outputs={
                "evidence": scored_violation.evidence_comment,
                "legal": scored_violation.legal_comment,
                "actionability": scored_violation.actionability_comment,
                "formulation": improvement.agent_output_raw,
            },
        )

        if item.possibly_not_a_violation:
            pipeline_stats["possibly_not_violations"] = int(pipeline_stats["possibly_not_violations"]) + 1

        # Count verifier overrides
        for note in item.verification_notes:
            if note.axis == "evidence" and note.status == "unconfirmed" and note.detail == "insufficient":
                pipeline_stats["verifier_overrides"] = int(pipeline_stats["verifier_overrides"]) + 1

        pipeline_stats["violations_processed"] = int(pipeline_stats["violations_processed"]) + 1
        return item

    except Exception as exc:
        logger.error(
            '{"event": "VIOLATION_PROCESSING_FAILED", "violation_id": "%s", "error": "%s"}',
            violation.id, str(exc),
        )
        pipeline_stats["violations_failed"] = int(pipeline_stats["violations_failed"]) + 1
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_act(
    file_path: str,
    index_act: bool = True,
    max_workers: int = 4,
) -> List[ChecklistItem]:
    """Extract violations from an act, evaluate them, and return checklist items.

    Args:
        file_path: Path to the act document (PDF or DOCX).
        index_act: Whether to index the act document in ChromaDB before analysis.
        max_workers: Thread pool size for parallel violation processing.

    Returns:
        List of ChecklistItem, one per successfully processed violation.
    """
    _reset_stats()

    logger.info(
        '{"event": "PIPELINE_START", "file": "%s"}',
        Path(file_path).name,
    )

    # Load and parse
    raw_text = load_document(file_path)

    # Optionally index the act itself for self-referential retrieval
    if index_act:
        try:
            from multi_indexer import index_historical_checklists
            index_historical_checklists([file_path])
        except Exception as exc:
            logger.warning("Act indexing failed (non-fatal): %s", exc)

    preprocessed = preprocess_act(raw_text, source_doc=Path(file_path).name, file_path=file_path)
    violations = violations_with_context_to_violations(preprocessed)
    violations = [normalize_violation(v) for v in violations]

    pipeline_stats["violations_total"] = len(violations)
    logger.info(
        '{"event": "VIOLATIONS_EXTRACTED", "count": %d, "file": "%s"}',
        len(violations), Path(file_path).name,
    )

    if not violations:
        return []

    # Parallel processing — each violation processed independently
    items: List[ChecklistItem] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_process_violation, v): v
            for v in violations
        }
        for future in as_completed(future_map):
            result = future.result()
            if result is not None:
                items.append(result)

    _update_avg_confidence(items)

    logger.info(
        '{"event": "PIPELINE_DONE", "total": %d, "processed": %d, "failed": %d, '
        '"avg_confidence": %.2f}',
        pipeline_stats["violations_total"],
        pipeline_stats["violations_processed"],
        pipeline_stats["violations_failed"],
        pipeline_stats["avg_confidence"],
    )

    return items
