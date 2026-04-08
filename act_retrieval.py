"""Act retrieval — the ONLY module that performs vector/BM25 search.

Architectural invariant: fetch_violation_context must be called exactly once
per violation, exclusively from act_pipeline._process_violation.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import config
from bm25_cache import bm25_search
from models import RetrievalResult, Violation, ViolationContext

logger = logging.getLogger(__name__)
_retrieval_logger = logging.getLogger("act_retrieval.calls")

# Debug-mode call counter — tracks per-violation call count
_call_counter: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Pure scoring functions
# ---------------------------------------------------------------------------


def _law_boost(chunk_text: str, law_ref: str) -> float:
    """Return 1.0 if law_ref appears in chunk_text, else 0.0."""
    if not law_ref:
        return 0.0
    return 1.0 if law_ref.lower() in chunk_text.lower() else 0.0


def _hybrid_score(cosine: float, bm25: float, law_boost: float) -> float:
    """Weighted combination of cosine, BM25, and law boost scores."""
    w = config.HYBRID_SCORE_WEIGHTS
    return (
        w["cosine"] * cosine
        + w["bm25"] * bm25
        + w["law_boost"] * law_boost
    )


# ---------------------------------------------------------------------------
# Collection search helpers
# ---------------------------------------------------------------------------


def _search_collection(
    kind: str,
    query: str,
    law_ref: str,
    top_k: int = config.RETRIEVAL_TOP_K,
) -> List[RetrievalResult]:
    """Search a ChromaDB collection with cosine + BM25 hybrid scoring."""
    try:
        from embeddings import embed_query
        from multi_indexer import get_ref_collection
    except ImportError as exc:
        logger.error("Missing dependency for retrieval: %s", exc)
        return []

    try:
        col = get_ref_collection(kind)
        if col.count() == 0:
            logger.warning("Collection %s is empty — no results", kind)
            return []

        query_embedding = embed_query(query)
        chroma_results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, col.count()),
            include=["documents", "distances", "metadatas"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed for %s: %s", kind, exc)
        return []

    documents = chroma_results["documents"][0]
    distances = chroma_results["distances"][0]
    metadatas = chroma_results["metadatas"][0]

    # Convert cosine distance to similarity
    cosine_scores = [max(0.0, 1.0 - d) for d in distances]

    # BM25 scores (normalized)
    bm25_raw = bm25_search(kind, query, top_k=top_k)
    bm25_max = max((s for _, s in bm25_raw), default=1.0) or 1.0
    bm25_map: Dict[int, float] = {idx: score / bm25_max for idx, score in bm25_raw}

    results: List[RetrievalResult] = []
    for i, (doc, cos, meta) in enumerate(zip(documents, cosine_scores, metadatas)):
        bm25_score = bm25_map.get(i, 0.0)
        boost = _law_boost(doc, law_ref)
        final = _hybrid_score(cosine=cos, bm25=bm25_score, law_boost=boost)
        results.append(
            RetrievalResult(
                chunk_id=meta.get("chunk_id", f"{kind}_{i}"),
                text=doc,
                score=final,
                source=kind,
                metadata=meta,
            )
        )

    return results


def _rerank_results(
    results: List[RetrievalResult],
    top_k: int = config.RERANK_TOP_K,
) -> List[RetrievalResult]:
    """Sort by score descending and take top_k."""
    return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_violation_context(violation: Violation) -> ViolationContext:
    """Retrieve all reference context for a violation.

    This is the ONLY place that performs search. Must be called exactly once
    per violation from act_pipeline._process_violation.
    """
    if not violation.normalized_text:
        raise ValueError(
            f"Violation {violation.id} not normalized before retrieval. "
            "Call normalize_violation() first."
        )

    # Debug double-retrieval detection
    if logger.isEnabledFor(logging.DEBUG):
        _call_counter[violation.id] = _call_counter.get(violation.id, 0) + 1
        if _call_counter[violation.id] > 1:
            logger.critical(
                "DOUBLE_RETRIEVAL_DETECTED violation_id=%s call_count=%d",
                violation.id,
                _call_counter[violation.id],
            )

    _retrieval_logger.info(
        "RETRIEVAL_CALLED violation_id=%s normalized_text_len=%d",
        violation.id,
        len(violation.normalized_text),
    )
    logger.info(
        '{"event": "RETRIEVAL_CALLED", "violation_id": "%s", "queries": {"norms": "%s"}}',
        violation.id,
        violation.normalized_text[:80],
    )

    query = violation.normalized_text
    law_ref = violation.law_ref

    retrieval_queries = {
        "norms": query,
        "typical": query,
        "historical": query,
    }

    ref_results: Dict[str, List[RetrievalResult]] = {}
    for kind in ("norms", "typical", "historical"):
        raw = _search_collection(kind, query, law_ref=law_ref)
        ref_results[kind] = _rerank_results(raw)

    ctx = ViolationContext(
        violation=violation,
        ref_results=ref_results,
        retrieval_queries=retrieval_queries,
    )

    norms_found = len(ref_results.get("norms", []))
    typical_found = len(ref_results.get("typical", []))
    logger.info(
        '{"event": "RETRIEVAL_DONE", "violation_id": "%s", "norms_found": %d, "typical_found": %d}',
        violation.id, norms_found, typical_found,
    )

    return ctx
