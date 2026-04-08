"""Tests for act_pipeline.py — mocked retrieval and LLM, spy on fetch_violation_context."""
from __future__ import annotations

import sys
import os
from dataclasses import replace
from typing import List
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import (
    ChecklistItem,
    ComparisonResult,
    ImprovedFormulation,
    RetrievalResult,
    Violation,
    ViolationContext,
)


# ---------------------------------------------------------------------------
# Shared test factories
# ---------------------------------------------------------------------------


def _make_violation(description: str = "нарушение охранной зоны трубопровода") -> Violation:
    v = Violation(
        raw_text=description,
        source_document="act.docx",
        page=1,
        section="описательная",
        description=description,
        subject="ООО Тест",
        law_ref="ФЗ-294",
    )
    from violation_normalizer import normalize_violation
    return normalize_violation(v)


def _make_context(violation: Violation) -> ViolationContext:
    return ViolationContext(
        violation=violation,
        ref_results={"norms": [], "typical": [], "historical": []},
        retrieval_queries={"norms": violation.normalized_text},
    )


def _make_scored(violation: Violation) -> Violation:
    return replace(
        violation,
        evidence_score=0.7,
        legal_score=0.6,
        actionability_score=0.55,
        evidence_comment="test evidence",
        legal_comment="test legal",
        actionability_comment="test actionability",
    )


def _make_improvement() -> ImprovedFormulation:
    return ImprovedFormulation(
        improved_text="улучшенный текст",
        legal_qualification="ст. 20 ФЗ-294",
        justification="обоснование",
        recommendation="рекомендация",
        legal_qualification_grounded=True,
    )


def _make_comparisons() -> dict:
    return {"norms": [], "typical": [], "historical": []}


def _make_checklist_item(v: Violation) -> ChecklistItem:
    return ChecklistItem(
        violation_id=v.id,
        raw_text=v.raw_text,
        description=v.description,
        subject=v.subject,
        law_ref=v.law_ref,
        source_document=v.source_document,
        page=v.page,
        section=v.section,
        evidence_score=0.7,
        confidence_score=0.65,
    )


# ---------------------------------------------------------------------------
# _process_violation
# ---------------------------------------------------------------------------


class TestProcessViolation:
    def _run(self, violation: Violation):
        from act_pipeline import _process_violation
        ctx = _make_context(violation)
        scored = _make_scored(violation)

        with (
            patch("act_pipeline.fetch_violation_context", return_value=ctx) as mock_fetch,
            patch("act_pipeline.evaluate_violation", return_value=scored),
            patch("act_pipeline.compare_violation", return_value=_make_comparisons()),
            patch("act_pipeline.improve_formulation", return_value=_make_improvement()),
        ):
            result = _process_violation(violation)

        return result, mock_fetch

    def test_returns_checklist_item(self):
        v = _make_violation()
        result, _ = self._run(v)
        assert isinstance(result, ChecklistItem)

    def test_returns_none_on_exception(self):
        """Exception in any step → None, pipeline does not crash."""
        from act_pipeline import _process_violation
        v = _make_violation()
        with patch("act_pipeline.fetch_violation_context", side_effect=RuntimeError("oops")):
            result = _process_violation(v)
        assert result is None

    def test_fetch_called_exactly_once(self):
        v = _make_violation()
        _, mock_fetch = self._run(v)
        assert mock_fetch.call_count == 1

    def test_fetch_called_with_correct_violation(self):
        v = _make_violation()
        _, mock_fetch = self._run(v)
        mock_fetch.assert_called_once_with(v)


# ---------------------------------------------------------------------------
# analyze_act — spy on fetch_violation_context
# ---------------------------------------------------------------------------


class TestAnalyzeAct:
    def _run_analyze(self, n_violations: int = 3):
        """Run analyze_act with n mocked violations."""
        from act_pipeline import analyze_act

        violations = [_make_violation(f"нарушение #{i} охранной зоны трубопровода газоснабжения") for i in range(n_violations)]

        def mock_normalize(v):
            return v

        def mock_fetch(violation):
            return _make_context(violation)

        def mock_evaluate(ctx):
            return _make_scored(ctx.violation)

        def mock_compare(ctx):
            return _make_comparisons()

        def mock_improve(ctx):
            return _make_improvement()

        with (
            patch("act_pipeline.load_document", return_value="mock text"),
            patch("act_pipeline.preprocess_act", return_value=[]),
            patch("act_pipeline.violations_with_context_to_violations", side_effect=lambda _: violations),
            patch("act_pipeline.normalize_violation", side_effect=mock_normalize),
            patch("act_pipeline.fetch_violation_context", side_effect=mock_fetch) as spy,
            patch("act_pipeline.evaluate_violation", side_effect=mock_evaluate),
            patch("act_pipeline.compare_violation", side_effect=mock_compare),
            patch("act_pipeline.improve_formulation", side_effect=mock_improve),
            patch("multi_indexer.index_historical_checklists"),
        ):
            items = analyze_act("mock_act.txt", index_act=True)

        return items, spy

    def test_returns_list_of_checklist_items(self):
        items, _ = self._run_analyze(3)
        assert len(items) == 3
        for item in items:
            assert isinstance(item, ChecklistItem)

    def test_fetch_called_exactly_n_times(self):
        """Architectural invariant: exactly one retrieval call per violation."""
        items, spy = self._run_analyze(3)
        assert spy.call_count == 3

    def test_fetch_count_equals_items_count(self):
        items, spy = self._run_analyze(5)
        assert spy.call_count == len(items)

    def test_none_violations_filtered_out(self):
        """If a violation fails processing, it should not appear in results."""
        from act_pipeline import analyze_act, _process_violation

        violations = [_make_violation() for _ in range(3)]

        call_count = [0]

        def mock_fetch(violation):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("simulated failure")
            return _make_context(violation)

        with (
            patch("act_pipeline.load_document", return_value="mock text"),
            patch("act_pipeline.preprocess_act", return_value=[]),
            patch("act_pipeline.violations_with_context_to_violations", side_effect=lambda _: violations),
            patch("act_pipeline.normalize_violation", side_effect=lambda v: v),
            patch("act_pipeline.fetch_violation_context", side_effect=mock_fetch),
            patch("act_pipeline.evaluate_violation", side_effect=lambda ctx: _make_scored(ctx.violation)),
            patch("act_pipeline.compare_violation", return_value=_make_comparisons()),
            patch("act_pipeline.improve_formulation", return_value=_make_improvement()),
            patch("multi_indexer.index_historical_checklists"),
        ):
            items = analyze_act("mock.txt", index_act=True)

        assert len(items) == 2

    def test_empty_violation_list_returns_empty(self):
        from act_pipeline import analyze_act

        with (
            patch("act_pipeline.load_document", return_value="empty"),
            patch("act_pipeline.preprocess_act", return_value=[]),
            patch("act_pipeline.violations_with_context_to_violations", return_value=[]),
            patch("act_pipeline.normalize_violation", side_effect=lambda v: v),
            patch("multi_indexer.index_historical_checklists"),
        ):
            items = analyze_act("empty.txt", index_act=True)

        assert items == []

    def test_pipeline_stats_updated(self):
        from act_pipeline import analyze_act, pipeline_stats

        self._run_analyze(2)
        assert pipeline_stats["violations_total"] == 2
        assert pipeline_stats["violations_processed"] == 2
        assert pipeline_stats["violations_failed"] == 0
