"""Tests for models.py — all dataclasses, guards, and auto-generation."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import (
    ChecklistItem,
    ComparisonResult,
    ImprovedFormulation,
    ItemTrace,
    RetrievalResult,
    Violation,
    VerificationResult,
    ViolationContext,
)


# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------


def _make_violation(**kwargs) -> Violation:
    defaults = dict(
        raw_text="нарушение охранной зоны",
        source_document="act.docx",
        page=1,
        section="описательная",
        description="не определены границы охранной зоны",
        subject="ООО Тест",
        law_ref="ФЗ-294",
    )
    defaults.update(kwargs)
    return Violation(**defaults)


class TestViolation:
    def test_instantiation(self):
        v = _make_violation()
        assert v.raw_text == "нарушение охранной зоны"
        assert v.source_document == "act.docx"

    def test_id_auto_generated(self):
        v = _make_violation()
        assert v.id != ""
        assert len(v.id) == 36  # UUID4 canonical form

    def test_id_unique_per_instance(self):
        v1 = _make_violation()
        v2 = _make_violation()
        assert v1.id != v2.id

    def test_id_preserved_when_provided(self):
        v = _make_violation(id="custom-id-123")
        assert v.id == "custom-id-123"

    def test_mutable_defaults_independent(self):
        v1 = _make_violation()
        v2 = _make_violation()
        v1.keywords.append("test")
        assert "test" not in v2.keywords

    def test_optional_scores_default_none(self):
        v = _make_violation()
        assert v.evidence_score is None
        assert v.legal_score is None
        assert v.actionability_score is None

    def test_possibly_not_a_violation_default_false(self):
        v = _make_violation()
        assert v.possibly_not_a_violation is False

    def test_empty_string_fields_default(self):
        v = _make_violation()
        assert v.normalized_text == ""
        assert v.evidence_comment == ""
        assert v.legal_comment == ""
        assert v.actionability_comment == ""


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_instantiation(self):
        r = RetrievalResult(chunk_id="c1", text="норма", score=0.85, source="norms")
        assert r.chunk_id == "c1"
        assert r.score == 0.85

    def test_metadata_default_empty_dict(self):
        r = RetrievalResult(chunk_id="c1", text="t", score=0.5, source="norms")
        assert r.metadata == {}

    def test_metadata_independent_between_instances(self):
        r1 = RetrievalResult(chunk_id="c1", text="t", score=0.5, source="norms")
        r2 = RetrievalResult(chunk_id="c2", text="t", score=0.5, source="norms")
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata


# ---------------------------------------------------------------------------
# ViolationContext
# ---------------------------------------------------------------------------


def _make_retrieval(chunk_id: str, score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, text="text", score=score, source="norms")


class TestViolationContext:
    def test_instantiation_empty(self):
        v = _make_violation()
        ctx = ViolationContext(violation=v)
        assert ctx.violation is v
        assert ctx.ref_results == {}
        assert ctx.used_chunk_ids == []

    def test_used_chunk_ids_auto_flattened(self):
        v = _make_violation()
        r1 = _make_retrieval("chunk-1")
        r2 = _make_retrieval("chunk-2")
        r3 = _make_retrieval("chunk-3")
        ctx = ViolationContext(
            violation=v,
            ref_results={"norms": [r1, r2], "typical": [r3]},
        )
        assert set(ctx.used_chunk_ids) == {"chunk-1", "chunk-2", "chunk-3"}

    def test_used_chunk_ids_not_overwritten_if_provided(self):
        v = _make_violation()
        r1 = _make_retrieval("chunk-1")
        ctx = ViolationContext(
            violation=v,
            ref_results={"norms": [r1]},
            used_chunk_ids=["explicit-id"],
        )
        assert ctx.used_chunk_ids == ["explicit-id"]

    def test_used_chunk_ids_no_duplicates_structure(self):
        """used_chunk_ids is a flat list — test it has no structural nesting."""
        v = _make_violation()
        r1 = _make_retrieval("c-a")
        r2 = _make_retrieval("c-b")
        ctx = ViolationContext(violation=v, ref_results={"norms": [r1, r2]})
        for item in ctx.used_chunk_ids:
            assert isinstance(item, str)


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------


class TestComparisonResult:
    def test_instantiation(self):
        cr = ComparisonResult(
            match_id="m1",
            source_type="norms",
            similarity_score=0.75,
            matched_text="текст нормы",
        )
        assert cr.match_id == "m1"
        assert cr.similarity_score == 0.75
        assert cr.law_ref == ""

    def test_mutable_metadata_default(self):
        cr1 = ComparisonResult(match_id="m1", source_type="norms", similarity_score=0.5, matched_text="t")
        cr2 = ComparisonResult(match_id="m2", source_type="norms", similarity_score=0.5, matched_text="t")
        cr1.metadata["x"] = 1
        assert "x" not in cr2.metadata


# ---------------------------------------------------------------------------
# ImprovedFormulation
# ---------------------------------------------------------------------------


class TestImprovedFormulation:
    def test_instantiation(self):
        imf = ImprovedFormulation(
            improved_text="улучшенный текст",
            legal_qualification="ст. 20 ФЗ-294",
            justification="обоснование",
            recommendation="рекомендация",
        )
        assert imf.legal_qualification_grounded is False
        assert imf.norm_source_ids == []
        assert imf.agent_output_raw == ""

    def test_norm_source_ids_independent(self):
        imf1 = ImprovedFormulation(improved_text="t", legal_qualification="q", justification="j", recommendation="r")
        imf2 = ImprovedFormulation(improved_text="t", legal_qualification="q", justification="j", recommendation="r")
        imf1.norm_source_ids.append("id1")
        assert "id1" not in imf2.norm_source_ids


# ---------------------------------------------------------------------------
# ItemTrace
# ---------------------------------------------------------------------------


class TestItemTrace:
    def test_instantiation(self):
        trace = ItemTrace(violation_id="v-1")
        assert trace.violation_id == "v-1"
        assert trace.used_chunk_ids == []
        assert trace.retrieval_queries == {}

    def test_list_fields_independent(self):
        t1 = ItemTrace(violation_id="v-1")
        t2 = ItemTrace(violation_id="v-2")
        t1.used_chunk_ids.append("chunk-a")
        assert "chunk-a" not in t2.used_chunk_ids


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_instantiation(self):
        vr = VerificationResult(axis="evidence", status="sufficient", detail="ok")
        assert vr.axis == "evidence"
        assert vr.pattern_count == 0
        assert vr.matched_patterns == []


# ---------------------------------------------------------------------------
# ChecklistItem
# ---------------------------------------------------------------------------


def _make_checklist_item(**kwargs) -> ChecklistItem:
    defaults = dict(
        violation_id="v-1",
        raw_text="raw",
        description="description",
        subject="ООО Тест",
        law_ref="ФЗ-294",
        source_document="act.docx",
        page=1,
        section="описательная",
    )
    defaults.update(kwargs)
    return ChecklistItem(**defaults)


class TestChecklistItem:
    def test_instantiation(self):
        item = _make_checklist_item()
        assert item.violation_id == "v-1"
        assert item.confidence_score == 0.0
        assert item.evidence_status == "unknown"
        assert item.legal_status == "unknown"
        assert item.actionability_status == "unknown"

    def test_optional_scores_default_none(self):
        item = _make_checklist_item()
        assert item.evidence_score is None
        assert item.legal_score is None
        assert item.actionability_score is None

    def test_mutable_defaults_independent(self):
        item1 = _make_checklist_item()
        item2 = _make_checklist_item()
        item1.verification_notes.append(
            VerificationResult(axis="evidence", status="ok", detail="")
        )
        assert len(item2.verification_notes) == 0

    def test_comparisons_dict_independent(self):
        item1 = _make_checklist_item()
        item2 = _make_checklist_item()
        item1.comparisons["norms"] = []
        assert "norms" not in item2.comparisons

    def test_trace_default_none(self):
        item = _make_checklist_item()
        assert item.trace is None

    def test_possibly_not_a_violation_default(self):
        item = _make_checklist_item()
        assert item.possibly_not_a_violation is False

    def test_grounding_default_false(self):
        item = _make_checklist_item()
        assert item.legal_qualification_grounded is False
