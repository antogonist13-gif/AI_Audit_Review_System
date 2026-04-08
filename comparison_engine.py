"""Comparison engine — O(n) conversion of retrieval results to ComparisonResult.

ARCHITECTURAL INVARIANT: This module NEVER imports act_retrieval.
All data arrives via ViolationContext.
"""
from __future__ import annotations

from typing import Dict, List

from models import ComparisonResult, RetrievalResult, ViolationContext


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_law_ref(metadata: dict) -> str:
    """Extract law reference from chunk metadata."""
    return metadata.get("law", "")


def _results_to_comparisons(
    results: List[RetrievalResult],
    source_type: str,
) -> List[ComparisonResult]:
    """Convert RetrievalResult list to ComparisonResult list — O(n), no IO."""
    comparisons: List[ComparisonResult] = []
    for r in results:
        comparisons.append(
            ComparisonResult(
                match_id=r.metadata.get("chunk_id", r.chunk_id),
                source_type=source_type,
                similarity_score=r.score,
                matched_text=r.text[:300],
                law_ref=_extract_law_ref(r.metadata),
                metadata=r.metadata,
            )
        )
    return comparisons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_violation(ctx: ViolationContext) -> Dict[str, List[ComparisonResult]]:
    """Convert all retrieval results in ctx to ComparisonResult lists.

    Pure conversion — O(n) with no IO and no retrieval calls.
    """
    return {
        kind: _results_to_comparisons(results, kind)
        for kind, results in ctx.ref_results.items()
    }
