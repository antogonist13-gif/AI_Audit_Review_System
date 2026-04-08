"""Tests for checklist_builder.py — no LLM, no ChromaDB."""
from __future__ import annotations

import sys
import os
from dataclasses import replace

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
from checklist_builder import (
    _apply_verifier_overrides,
    _compute_confidence,
    _score_to_status,
    build_checklist_item,
)
from models import VerificationResult


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_violation(**kwargs) -> Violation:
    defaults = dict(
        raw_text="нарушение охранной зоны 15.03.2024 штраф 50 000 руб",
        source_document="act.docx",
        page=1,
        section="описательная",
        description="нарушение охранной зоны трубопровода газоснабжения",
        subject="ООО Тест",
        law_ref="ФЗ-294",
        evidence_score=0.7,
        legal_score=0.6,
        actionability_score=0.55,
    )
    defaults.update(kwargs)
    return Violation(**defaults)


def _make_ctx(violation: Violation = None) -> ViolationContext:
    if violation is None:
        violation = _make_violation()
    return ViolationContext(
        violation=violation,
        ref_results={"norms": [], "typical": [], "historical": []},
        retrieval_queries={"norms": "тест"},
    )


def _make_improvement(**kwargs) -> ImprovedFormulation:
    defaults = dict(
        improved_text="улучшенный текст",
        legal_qualification="ст. 20 ФЗ-294",
        justification="обоснование",
        recommendation="рекомендация",
        legal_qualification_grounded=True,
    )
    defaults.update(kwargs)
    return ImprovedFormulation(**defaults)


def _make_comparisons(**kwargs) -> dict:
    return {"norms": [], "typical": [], "historical": []}


# ---------------------------------------------------------------------------
# _score_to_status — pure function tests
# ---------------------------------------------------------------------------


class TestScoreToStatus:
    def test_below_lo_returns_first_label(self):
        assert _score_to_status(0.2, lo=0.4, hi=0.6) == "insufficient"

    def test_between_lo_and_hi_returns_second_label(self):
        assert _score_to_status(0.5, lo=0.4, hi=0.6) == "unclear"

    def test_at_hi_or_above_returns_third_label(self):
        assert _score_to_status(0.8, lo=0.4, hi=0.6) == "sufficient"

    def test_exactly_at_lo_returns_second_label(self):
        assert _score_to_status(0.4, lo=0.4, hi=0.6) == "unclear"

    def test_none_returns_unknown(self):
        assert _score_to_status(None, lo=0.4, hi=0.6) == "unknown"

    def test_custom_labels(self):
        assert _score_to_status(0.1, lo=0.3, hi=0.7, labels=("bad", "ok", "good")) == "bad"
        assert _score_to_status(0.5, lo=0.3, hi=0.7, labels=("bad", "ok", "good")) == "ok"
        assert _score_to_status(0.9, lo=0.3, hi=0.7, labels=("bad", "ok", "good")) == "good"

    def test_zero_score(self):
        assert _score_to_status(0.0, lo=0.4, hi=0.6) == "insufficient"

    def test_one_score(self):
        assert _score_to_status(1.0, lo=0.4, hi=0.6) == "sufficient"


# ---------------------------------------------------------------------------
# _compute_confidence — pure function tests
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_all_scores_present(self):
        v = _make_violation(evidence_score=0.8, legal_score=0.7, actionability_score=0.6)
        conf = _compute_confidence(v, {})
        assert 0.0 <= conf <= 1.0

    def test_none_scores_treated_as_zero(self):
        v = _make_violation(evidence_score=None, legal_score=None, actionability_score=None)
        conf = _compute_confidence(v, {})
        assert conf == 0.0

    def test_with_comparisons(self):
        v = _make_violation(evidence_score=0.8, legal_score=0.8, actionability_score=0.8)
        comps = {"norms": [
            ComparisonResult(match_id="c1", source_type="norms", similarity_score=0.9, matched_text="t"),
        ]}
        conf = _compute_confidence(v, comps)
        assert conf > 0.0
        assert conf <= 1.0

    def test_clamped_to_one(self):
        v = _make_violation(evidence_score=1.0, legal_score=1.0, actionability_score=1.0)
        comps = {"norms": [
            ComparisonResult(match_id="c1", source_type="norms", similarity_score=1.0, matched_text="t"),
        ]}
        conf = _compute_confidence(v, comps)
        assert conf <= 1.0

    def test_result_in_unit_interval_for_any_inputs(self):
        for es, ls, acs in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (None, None, None)]:
            v = _make_violation(evidence_score=es, legal_score=ls, actionability_score=acs)
            conf = _compute_confidence(v, {})
            assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# _apply_verifier_overrides
# ---------------------------------------------------------------------------


class TestApplyVerifierOverrides:
    def _make_item(self, evidence_status="sufficient") -> ChecklistItem:
        return ChecklistItem(
            violation_id="v-1",
            raw_text="raw",
            description="desc",
            subject="ООО",
            law_ref="ФЗ",
            source_document="act.docx",
            page=1,
            section="описательная",
            evidence_status=evidence_status,
        )

    def test_insufficient_evidence_overrides_status(self):
        item = self._make_item(evidence_status="sufficient")
        notes = [VerificationResult(axis="evidence", status="unconfirmed", detail="insufficient")]
        result = _apply_verifier_overrides(item, notes)
        assert result.evidence_status == "insufficient"

    def test_insufficient_evidence_sets_possibly_not_violation(self):
        item = self._make_item()
        notes = [VerificationResult(axis="evidence", status="unconfirmed", detail="insufficient")]
        result = _apply_verifier_overrides(item, notes)
        assert result.possibly_not_a_violation is True

    def test_partial_evidence_does_not_override(self):
        item = self._make_item(evidence_status="sufficient")
        notes = [VerificationResult(axis="evidence", status="unclear", detail="partial")]
        result = _apply_verifier_overrides(item, notes)
        assert result.evidence_status == "sufficient"

    def test_non_evidence_axis_does_not_override(self):
        item = self._make_item(evidence_status="sufficient")
        notes = [VerificationResult(axis="legal", status="unconfirmed", detail="insufficient")]
        result = _apply_verifier_overrides(item, notes)
        assert result.evidence_status == "sufficient"

    def test_returns_new_object(self):
        item = self._make_item()
        notes = [VerificationResult(axis="evidence", status="unconfirmed", detail="insufficient")]
        result = _apply_verifier_overrides(item, notes)
        assert result is not item


# ---------------------------------------------------------------------------
# build_checklist_item
# ---------------------------------------------------------------------------


class TestBuildChecklistItem:
    def test_basic_construction(self):
        v = _make_violation()
        ctx = _make_ctx(v)
        item = build_checklist_item(
            ctx=ctx,
            scored_violation=v,
            comparisons=_make_comparisons(),
            improvement=_make_improvement(),
            agent_outputs={},
        )
        assert isinstance(item, ChecklistItem)
        assert item.violation_id == v.id

    def test_raises_if_evidence_score_none(self):
        v = _make_violation(evidence_score=None)
        ctx = _make_ctx(v)
        with pytest.raises(AssertionError, match="not evaluated"):
            build_checklist_item(
                ctx=ctx,
                scored_violation=v,
                comparisons=_make_comparisons(),
                improvement=_make_improvement(),
                agent_outputs={},
            )

    def test_confidence_in_unit_interval(self):
        v = _make_violation()
        ctx = _make_ctx(v)
        item = build_checklist_item(
            ctx=ctx,
            scored_violation=v,
            comparisons=_make_comparisons(),
            improvement=_make_improvement(),
            agent_outputs={},
        )
        assert 0.0 <= item.confidence_score <= 1.0

    def test_trace_is_not_none(self):
        v = _make_violation()
        ctx = _make_ctx(v)
        item = build_checklist_item(
            ctx=ctx,
            scored_violation=v,
            comparisons=_make_comparisons(),
            improvement=_make_improvement(),
            agent_outputs={},
        )
        assert item.trace is not None

    def test_possibly_not_violation_when_low_evidence_score(self):
        """evidence_score < 0.4 → possibly_not_a_violation=True."""
        v = _make_violation(evidence_score=0.2)
        ctx = _make_ctx(v)
        item = build_checklist_item(
            ctx=ctx,
            scored_violation=v,
            comparisons=_make_comparisons(),
            improvement=_make_improvement(),
            agent_outputs={},
        )
        assert item.possibly_not_a_violation is True

    def test_llm_cannot_set_status_directly(self):
        """Status is determined from score < 0.4, not from LLM text."""
        v = _make_violation(evidence_score=0.2)
        ctx = _make_ctx(v)
        item = build_checklist_item(
            ctx=ctx,
            scored_violation=v,
            comparisons=_make_comparisons(),
            improvement=_make_improvement(),
            agent_outputs={"evidence": "СТАТУС: sufficient"},
        )
        assert item.evidence_status == "insufficient"

    def test_verifier_overrides_llm_when_insufficient(self):
        """Verifier insufficient detection overrides any LLM-assigned status."""
        v = _make_violation(evidence_score=0.5, raw_text="нарушение без деталей")
        ctx = _make_ctx(v)
        item = build_checklist_item(
            ctx=ctx,
            scored_violation=v,
            comparisons=_make_comparisons(),
            improvement=_make_improvement(),
            agent_outputs={},
        )
        # If verifier flagged insufficient, item status should be insufficient
        evidence_verif = next(
            (n for n in item.verification_notes if n.axis == "evidence"), None
        )
        if evidence_verif and evidence_verif.status == "unconfirmed" and evidence_verif.detail == "insufficient":
            assert item.evidence_status == "insufficient"
            assert item.possibly_not_a_violation is True

    def test_legal_qualification_grounding_preserved(self):
        v = _make_violation()
        ctx = _make_ctx(v)
        improvement = _make_improvement(legal_qualification_grounded=False)
        item = build_checklist_item(
            ctx=ctx,
            scored_violation=v,
            comparisons=_make_comparisons(),
            improvement=improvement,
            agent_outputs={},
        )
        assert item.legal_qualification_grounded is False

    def test_agent_outputs_stored(self):
        v = _make_violation()
        ctx = _make_ctx(v)
        outputs = {"evidence": "БАЛЛ: 7", "legal": "БАЛЛ: 8"}
        item = build_checklist_item(
            ctx=ctx,
            scored_violation=v,
            comparisons=_make_comparisons(),
            improvement=_make_improvement(),
            agent_outputs=outputs,
        )
        assert item.agent_outputs == outputs
