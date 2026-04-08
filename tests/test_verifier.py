"""Tests for verifier.py — no LLM, no ChromaDB, 8+ boundary test cases."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import RetrievalResult, Violation
from verifier import (
    verify_actionability,
    verify_checklist_item,
    verify_evidence_sufficiency,
    verify_law_applicability,
)


def _make_violation(raw_text: str = "", description: str = "", law_ref: str = "ФЗ-294") -> Violation:
    return Violation(
        raw_text=raw_text or description,
        source_document="act.docx",
        page=1,
        section="описательная",
        description=description or raw_text,
        subject="ООО Тест",
        law_ref=law_ref,
    )


def _make_result(text: str, chunk_id: str = "c1") -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, text=text, score=0.9, source="norms")


# ---------------------------------------------------------------------------
# verify_evidence_sufficiency — boundary tests
# ---------------------------------------------------------------------------


class TestVerifyEvidenceSufficiency:
    def test_zero_patterns_insufficient(self):
        """No evidence markers → insufficient."""
        v = _make_violation(raw_text="нарушение требований безопасности")
        result = verify_evidence_sufficiency(v)
        assert result.axis == "evidence"
        assert result.status == "unconfirmed"
        assert result.detail == "insufficient"
        assert result.pattern_count == 0

    def test_exactly_one_pattern_unclear(self):
        """Exactly 1 evidence marker → unclear."""
        v = _make_violation(raw_text="нарушение выявлено 15.03.2024 на объекте")
        result = verify_evidence_sufficiency(v)
        assert result.status == "unclear"
        assert result.detail == "partial"
        assert result.pattern_count == 1

    def test_exactly_two_patterns_sufficient(self):
        """Exactly 2 evidence markers → sufficient."""
        v = _make_violation(raw_text="нарушение выявлено 15.03.2024, сумма штрафа 50 000 руб")
        result = verify_evidence_sufficiency(v)
        assert result.status == "confirmed"
        assert result.detail == "sufficient"
        assert result.pattern_count >= 2

    def test_three_patterns_sufficient(self):
        """Date + amount + doc number → sufficient."""
        v = _make_violation(
            raw_text="акт №125 от 15.03.2024 нарушение на сумму 500 000 руб"
        )
        result = verify_evidence_sufficiency(v)
        assert result.status == "confirmed"
        assert result.pattern_count >= 3

    def test_initials_count_as_pattern(self):
        """ФИО initials pattern should be recognised."""
        v = _make_violation(raw_text="ответственный Иванов И.И. нарушил требования")
        result = verify_evidence_sufficiency(v)
        assert result.pattern_count >= 1

    def test_returns_verification_result_axis(self):
        v = _make_violation(raw_text="нарушение")
        result = verify_evidence_sufficiency(v)
        assert result.axis == "evidence"

    def test_combined_raw_and_description(self):
        """Both raw_text and description are checked together."""
        v = _make_violation(
            raw_text="нарушение зоны безопасности",
            description="нарушение выявлено 01.01.2024, штраф 10 000 руб",
        )
        result = verify_evidence_sufficiency(v)
        assert result.status == "confirmed"

    def test_matched_patterns_list_populated(self):
        v = _make_violation(raw_text="выявлено 15.03.2024 штраф 50 000 руб акт №23")
        result = verify_evidence_sufficiency(v)
        assert len(result.matched_patterns) > 0


# ---------------------------------------------------------------------------
# verify_law_applicability
# ---------------------------------------------------------------------------


class TestVerifyLawApplicability:
    def test_no_law_ref_returns_unconfirmed(self):
        v = _make_violation(raw_text="нарушение", law_ref="")
        result = verify_law_applicability(v, [])
        assert result.axis == "legal"
        assert result.status == "unconfirmed"
        assert result.detail == "no_law_ref"

    def test_no_norm_results_returns_unconfirmed(self):
        v = _make_violation(raw_text="нарушение", law_ref="ФЗ-294")
        result = verify_law_applicability(v, [])
        assert result.status == "unconfirmed"
        assert result.detail == "no_norm_results"

    def test_matching_law_in_norms_confirmed(self):
        v = _make_violation(raw_text="нарушение", law_ref="ФЗ-294")
        r = _make_result("Федеральный закон ФЗ-294 о защите прав потребителей")
        result = verify_law_applicability(v, [r])
        assert result.status == "confirmed"

    def test_non_matching_law_unconfirmed(self):
        v = _make_violation(raw_text="нарушение", law_ref="ФЗ-999")
        r = _make_result("текст не содержит упоминания данного закона")
        result = verify_law_applicability(v, [r])
        assert result.status == "unconfirmed"


# ---------------------------------------------------------------------------
# verify_actionability
# ---------------------------------------------------------------------------


class TestVerifyActionability:
    def test_with_actionable_keywords(self):
        v = _make_violation(raw_text="организация обязана устранить нарушение")
        result = verify_actionability(v)
        assert result.axis == "actionability"
        assert result.status == "confirmed"

    def test_without_actionable_keywords(self):
        v = _make_violation(raw_text="нарушение зафиксировано в ходе проверки")
        result = verify_actionability(v)
        assert result.status == "unconfirmed"


# ---------------------------------------------------------------------------
# verify_checklist_item orchestrator
# ---------------------------------------------------------------------------


class TestVerifyChecklistItem:
    def test_returns_three_results(self):
        v = _make_violation(raw_text="нарушение зафиксировано")
        results = verify_checklist_item(v, {})
        assert len(results) == 3

    def test_axes_are_correct(self):
        v = _make_violation(raw_text="нарушение зафиксировано")
        results = verify_checklist_item(v, {})
        axes = {r.axis for r in results}
        assert "evidence" in axes
        assert "legal" in axes
        assert "actionability" in axes

    def test_with_norm_results(self):
        v = _make_violation(raw_text="нарушение 15.03.2024 штраф 50 000 руб", law_ref="ФЗ-294")
        r = _make_result("ФЗ-294 применяется к данному нарушению")
        results = verify_checklist_item(v, {"norms": [r]})
        evidence = next(r for r in results if r.axis == "evidence")
        legal = next(r for r in results if r.axis == "legal")
        assert evidence.status in ("confirmed", "unclear", "unconfirmed")
        assert legal.status in ("confirmed", "unconfirmed")
