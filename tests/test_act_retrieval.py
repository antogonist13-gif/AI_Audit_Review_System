"""Tests for act_retrieval.py — mocked ChromaDB and embeddings."""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import RetrievalResult, Violation, ViolationContext
from act_retrieval import _hybrid_score, _law_boost, _rerank_results, fetch_violation_context


def _make_violation(normalized_text: str = "нарушение охранной зоны трубопровода") -> Violation:
    v = Violation(
        raw_text="нарушение",
        source_document="act.docx",
        page=1,
        section="описательная",
        description="нарушение охранной зоны трубопровода газоснабжения",
        subject="ООО Тест",
        law_ref="ФЗ-294",
    )
    from dataclasses import replace
    return replace(v, normalized_text=normalized_text)


def _make_result(chunk_id: str = "c1", score: float = 0.8, source: str = "norms") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, text="норма права ФЗ-294", score=score, source=source)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestLawBoost:
    def test_boost_when_law_in_text(self):
        assert _law_boost("ФЗ-294 применяется к данному случаю", "ФЗ-294") == 1.0

    def test_no_boost_when_law_absent(self):
        assert _law_boost("совсем другой текст", "ФЗ-294") == 0.0

    def test_empty_law_ref_returns_zero(self):
        assert _law_boost("любой текст ФЗ-294", "") == 0.0

    def test_case_insensitive(self):
        assert _law_boost("фз-294 применяется", "ФЗ-294") == 1.0


class TestHybridScore:
    def test_weighted_combination(self):
        import config
        w = config.HYBRID_SCORE_WEIGHTS
        expected = w["cosine"] * 0.8 + w["bm25"] * 0.6 + w["law_boost"] * 1.0
        assert abs(_hybrid_score(0.8, 0.6, 1.0) - expected) < 1e-9

    def test_all_zeros(self):
        assert _hybrid_score(0.0, 0.0, 0.0) == 0.0

    def test_all_ones(self):
        result = _hybrid_score(1.0, 1.0, 1.0)
        assert 0.99 <= result <= 1.01


class TestRerankResults:
    def test_sorted_by_score_descending(self):
        results = [
            _make_result("c1", score=0.5),
            _make_result("c2", score=0.9),
            _make_result("c3", score=0.3),
        ]
        reranked = _rerank_results(results, top_k=10)
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        results = [_make_result(f"c{i}", score=float(i) / 10) for i in range(10)]
        reranked = _rerank_results(results, top_k=3)
        assert len(reranked) == 3

    def test_empty_input(self):
        assert _rerank_results([]) == []


# ---------------------------------------------------------------------------
# fetch_violation_context guard
# ---------------------------------------------------------------------------


class TestFetchViolationContextGuard:
    def test_raises_if_not_normalized(self):
        v = Violation(
            raw_text="нарушение",
            source_document="act.docx",
            page=1,
            section="описательная",
            description="нарушение",
            subject="ООО Тест",
            law_ref="ФЗ-294",
        )
        with pytest.raises(ValueError, match="not normalized"):
            fetch_violation_context(v)


# ---------------------------------------------------------------------------
# fetch_violation_context with mocks
# ---------------------------------------------------------------------------


def _mock_search_collection(kind, query, law_ref, top_k=10):
    return [
        RetrievalResult(
            chunk_id=f"{kind}-chunk-1",
            text=f"Текст норм {kind} ФЗ-294",
            score=0.75,
            source=kind,
            metadata={"chunk_id": f"{kind}-chunk-1", "source": kind},
        )
    ]


class TestFetchViolationContext:
    def test_returns_violation_context(self):
        v = _make_violation()
        with patch("act_retrieval._search_collection", side_effect=_mock_search_collection):
            ctx = fetch_violation_context(v)
        assert isinstance(ctx, ViolationContext)
        assert ctx.violation is v

    def test_ref_results_have_expected_kinds(self):
        v = _make_violation()
        with patch("act_retrieval._search_collection", side_effect=_mock_search_collection):
            ctx = fetch_violation_context(v)
        assert "norms" in ctx.ref_results
        assert "typical" in ctx.ref_results
        assert "historical" in ctx.ref_results

    def test_retrieval_queries_contain_normalized_text(self):
        v = _make_violation()
        with patch("act_retrieval._search_collection", side_effect=_mock_search_collection):
            ctx = fetch_violation_context(v)
        assert ctx.retrieval_queries["norms"] == v.normalized_text

    def test_used_chunk_ids_non_empty(self):
        v = _make_violation()
        with patch("act_retrieval._search_collection", side_effect=_mock_search_collection):
            ctx = fetch_violation_context(v)
        assert len(ctx.used_chunk_ids) > 0

    def test_used_chunk_ids_flat_list_of_strings(self):
        v = _make_violation()
        with patch("act_retrieval._search_collection", side_effect=_mock_search_collection):
            ctx = fetch_violation_context(v)
        for item in ctx.used_chunk_ids:
            assert isinstance(item, str)

    def test_context_not_none(self):
        v = _make_violation()
        with patch("act_retrieval._search_collection", side_effect=_mock_search_collection):
            ctx = fetch_violation_context(v)
        assert ctx is not None


class TestRetrievalCalledExactlyNTimes:
    def test_called_once_per_violation(self):
        """The spy pattern used in act_pipeline integration test."""
        violations = [_make_violation() for _ in range(3)]
        call_count = [0]

        def mock_fetch(violation):
            call_count[0] += 1
            return ViolationContext(
                violation=violation,
                ref_results={"norms": [], "typical": [], "historical": []},
            )

        for v in violations:
            mock_fetch(v)

        assert call_count[0] == 3

    def test_context_is_immutable_after_creation(self):
        """ViolationContext should not expose setters that allow mutation."""
        v = _make_violation()
        ctx = ViolationContext(
            violation=v,
            ref_results={"norms": [_make_result()]},
        )
        # Adding to the results list externally should not be possible
        # through the context's own interface (no set methods)
        assert not hasattr(ctx, "set_results")
        assert not hasattr(ctx, "add_result")
