"""Integration tests — spy on fetch_violation_context, assert call_count == len(items).

These tests exercise the full pipeline with mocked LLM and ChromaDB.
They do NOT require Ollama or a running ChromaDB instance.
"""
from __future__ import annotations

import sys
import os
from dataclasses import replace
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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
# Shared fixtures
# ---------------------------------------------------------------------------


SAMPLE_ACT_TEXT = """
АКТ ПРОВЕРКИ №456 от 15.03.2024

ОПИСАТЕЛЬНАЯ ЧАСТЬ

1. ООО «ТехПром» не определены границы охранной зоны газопровода высокого давления
   в соответствии с требованиями ФЗ-69 ст. 28. Акт согласования №12 от 01.01.2024 отсутствует.
   Штраф составляет 200 000 рублей.

2. ООО «ТехПром» не проведена техническая диагностика оборудования в установленные сроки
   согласно ФЗ-116 ст. 7. Последнее освидетельствование проводилось 01.05.2021.
   Документ №А-45 не представлен. Ответственный Петров П.П.

3. Отсутствует план мероприятий по предотвращению аварий на производственном объекте,
   требуемый приказом Ростехнадзора №520. Дата последней проверки 20.02.2024.
"""


def _make_mock_context(violation: Violation) -> ViolationContext:
    return ViolationContext(
        violation=violation,
        ref_results={
            "norms": [
                RetrievalResult(
                    chunk_id="norm-1",
                    text="ФЗ-294 устанавливает требования к проведению проверок",
                    score=0.8,
                    source="norms",
                    metadata={"chunk_id": "norm-1", "law": "ФЗ-294"},
                )
            ],
            "typical": [],
            "historical": [],
        },
        retrieval_queries={"norms": violation.normalized_text},
    )


def _make_scored(violation: Violation) -> Violation:
    return replace(
        violation,
        evidence_score=0.7,
        legal_score=0.65,
        actionability_score=0.6,
        evidence_comment="Evidence found",
        legal_comment="Legal reference confirmed",
        actionability_comment="Actionable",
    )


def _make_improvement() -> ImprovedFormulation:
    return ImprovedFormulation(
        improved_text="Улучшенная формулировка нарушения",
        legal_qualification="ст. 20 ФЗ-294",
        justification="Обоснование квалификации",
        recommendation="Устранить нарушение в течение 30 дней",
        legal_qualification_grounded=True,
    )


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    def _run_pipeline(self, act_text: str = SAMPLE_ACT_TEXT):
        """Run the full pipeline with all external dependencies mocked."""
        from act_pipeline import analyze_act

        with (
            patch("act_pipeline.load_document", return_value=act_text),
            patch("act_pipeline.evaluate_violation", side_effect=lambda ctx: _make_scored(ctx.violation)),
            patch("act_pipeline.compare_violation", return_value={"norms": [], "typical": [], "historical": []}),
            patch("act_pipeline.improve_formulation", return_value=_make_improvement()),
            patch("multi_indexer.index_historical_checklists"),
            patch("act_pipeline.fetch_violation_context", wraps=_make_mock_context) as spy,
        ):
            items = analyze_act(act_text, index_act=True, max_workers=1)

        return items, spy

    def test_returns_checklist_items(self):
        items, _ = self._run_pipeline()
        assert len(items) >= 1
        for item in items:
            assert isinstance(item, ChecklistItem)

    def test_fetch_violation_context_called_exactly_n_times(self):
        """Core invariant: retrieval called exactly once per violation."""
        items, spy = self._run_pipeline()
        assert spy.call_count == len(items), (
            f"Expected {len(items)} retrieval calls, got {spy.call_count}"
        )

    def test_confidence_in_unit_interval(self):
        items, _ = self._run_pipeline()
        for item in items:
            assert 0.0 <= item.confidence_score <= 1.0, (
                f"confidence_score {item.confidence_score} out of [0,1]"
            )

    def test_trace_not_none(self):
        items, _ = self._run_pipeline()
        for item in items:
            assert item.trace is not None

    def test_used_chunk_ids_present(self):
        items, _ = self._run_pipeline()
        for item in items:
            assert item.trace is not None
            # used_chunk_ids may be empty if retrieval returned nothing,
            # but the list itself should exist
            assert isinstance(item.trace.used_chunk_ids, list)

    def test_no_double_retrieval(self):
        """Each violation ID should appear at most once in spy call args."""
        items, spy = self._run_pipeline()
        called_ids = [call_args[0][0].id for call_args in spy.call_args_list]
        assert len(called_ids) == len(set(called_ids)), (
            f"Duplicate violation IDs in retrieval calls: {called_ids}"
        )

    def test_grounded_items_marked_correctly(self):
        items, _ = self._run_pipeline()
        for item in items:
            # The mock improvement sets grounded=True, so all should be grounded
            assert item.legal_qualification_grounded is True

    def test_pipeline_processes_all_violations(self):
        """All extracted violations should be in the output (no silent drops)."""
        from act_parser import extract_violations
        from violation_normalizer import normalize_violation

        violations = extract_violations(SAMPLE_ACT_TEXT, "test.txt")
        violations = [normalize_violation(v) for v in violations]
        n_expected = len(violations)

        items, spy = self._run_pipeline()

        assert len(items) == n_expected, (
            f"Expected {n_expected} items, got {len(items)}"
        )
        assert spy.call_count == n_expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_act_returns_empty_list(self):
        from act_pipeline import analyze_act

        with (
            patch("act_pipeline.load_document", return_value=""),
            patch("multi_indexer.index_historical_checklists"),
        ):
            items = analyze_act("empty.txt", index_act=True)

        assert items == []

    def test_single_violation_act(self):
        single_act = """
        ОПИСАТЕЛЬНАЯ ЧАСТЬ
        1. ООО «Пример» нарушены требования безопасности согласно ФЗ-294 ст.20.
           Штраф 50 000 руб. Акт №33 от 10.03.2024. Ответственный Иванов И.И.
        """
        from act_pipeline import analyze_act

        with (
            patch("act_pipeline.load_document", return_value=single_act),
            patch("act_pipeline.evaluate_violation", side_effect=lambda ctx: _make_scored(ctx.violation)),
            patch("act_pipeline.compare_violation", return_value={"norms": [], "typical": [], "historical": []}),
            patch("act_pipeline.improve_formulation", return_value=_make_improvement()),
            patch("multi_indexer.index_historical_checklists"),
            patch("act_pipeline.fetch_violation_context", wraps=_make_mock_context) as spy,
        ):
            items = analyze_act(single_act, index_act=True, max_workers=1)

        assert spy.call_count == len(items)
        assert len(items) == 1
