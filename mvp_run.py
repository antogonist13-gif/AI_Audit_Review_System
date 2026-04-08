"""MVP runner — minimal end-to-end smoke test (no LLM evaluation, no ChromaDB required).

Usage:
    python3 mvp_run.py [path_to_act]

After Phase 3 this demonstrates the foundational pipeline:
    load → extract → normalize → (mock) retrieve
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mvp_run")


def run_mvp(act_path: str) -> None:
    from act_preprocessor import preprocess_act, violations_with_context_to_violations
    from loader import load_document
    from models import RetrievalResult, ViolationContext
    from violation_normalizer import normalize_violation

    logger.info("Loading document: %s", act_path)
    raw = load_document(act_path)
    logger.info("Document loaded — %d characters", len(raw))

    preprocessed = preprocess_act(raw, source_doc=Path(act_path).name, file_path=act_path)
    violations = violations_with_context_to_violations(preprocessed)
    logger.info("Extracted %d violations", len(violations))

    if not violations:
        logger.warning("No violations found — check the document format")
        return

    normalized = [normalize_violation(v) for v in violations]

    def _mock_fetch(violation):
        """Stub retrieval — returns empty context for smoke-testing."""
        return ViolationContext(
            violation=violation,
            ref_results={"norms": [], "typical": [], "historical": []},
            retrieval_queries={"norms": violation.normalized_text},
        )

    print("\n" + "=" * 70)
    print("MVP PIPELINE RESULTS")
    print("=" * 70)

    for v in normalized:
        ctx = _mock_fetch(v)
        print(f"\nViolation ID : {v.id[:8]}...")
        print(f"Description  : {v.description[:80]}")
        print(f"Subject      : {v.subject or '(not detected)'}")
        print(f"Law ref      : {v.law_ref or '(not detected)'}")
        print(f"Norm text    : {v.normalized_text[:60]}")
        print(f"Keywords     : {', '.join(v.keywords[:5])}")
        print(f"Weak flag    : {v.possibly_not_a_violation}")
        print(f"Norms found  : {len(ctx.ref_results['norms'])}")
        print(f"Typical found: {len(ctx.ref_results['typical'])}")
        print(f"Chunk IDs    : {ctx.used_chunk_ids}")

    print("\n" + "=" * 70)
    print(f"Total violations: {len(normalized)}")
    print("Phase 3 MVP smoke test PASSED")
    print("=" * 70)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample_act.txt"
    if not Path(path).exists():
        logger.error("File not found: %s", path)
        sys.exit(1)
    run_mvp(path)
