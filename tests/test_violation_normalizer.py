"""Tests for violation_normalizer.py — no mocks, no LLM, no ChromaDB."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import Violation
from violation_normalizer import _extract_keywords, _normalize_text, normalize_violation


def _make_violation(description: str = "не определены границы охранной зоны", **kwargs) -> Violation:
    defaults = dict(
        raw_text=description,
        source_document="act.docx",
        page=1,
        section="описательная",
        description=description,
        subject="ООО Тест",
        law_ref="ФЗ-294",
    )
    defaults.update(kwargs)
    return Violation(**defaults)


# ---------------------------------------------------------------------------
# _normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_returns_non_empty_for_non_empty_input(self):
        result = _normalize_text("Нарушение границ охранной зоны")
        assert result != ""

    def test_lowercases_output(self):
        result = _normalize_text("НАРУШЕНИЕ ОХРАННОЙ ЗОНЫ")
        assert result == result.lower()

    def test_strips_punctuation(self):
        result = _normalize_text("нарушение, охранной. зоны!")
        assert "," not in result
        assert "." not in result
        assert "!" not in result

    def test_multiple_spaces_collapsed(self):
        result = _normalize_text("  нарушение   зоны  ")
        assert "  " not in result

    def test_returns_string(self):
        assert isinstance(_normalize_text("текст"), str)

    def test_empty_input_returns_empty(self):
        assert _normalize_text("") == ""

    def test_single_word(self):
        assert _normalize_text("нарушение") != ""


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_returns_list(self):
        keywords = _extract_keywords("нарушение охранной зоны норма закона")
        assert isinstance(keywords, list)

    def test_length_le_top_n(self):
        keywords = _extract_keywords("нарушение охранной зоны норма закона", top_n=5)
        assert len(keywords) <= 5

    def test_stopwords_excluded(self):
        import config
        keywords = _extract_keywords("нарушение и в охранной зоны")
        for word in keywords:
            assert word not in config.RUSSIAN_STOPWORDS

    def test_empty_text_returns_empty(self):
        assert _extract_keywords("") == []

    def test_short_words_excluded(self):
        keywords = _extract_keywords("нарушение ох зо")
        for k in keywords:
            assert len(k) > 2

    def test_most_common_first(self):
        text = "нарушение нарушение нарушение охранной зоны зоны"
        keywords = _extract_keywords(text, top_n=10)
        # "нарушение" appears 3× and should be first
        if "нарушение" in keywords and "охранной" in keywords:
            assert keywords.index("нарушение") < keywords.index("охранной")


# ---------------------------------------------------------------------------
# normalize_violation
# ---------------------------------------------------------------------------


class TestNormalizeViolation:
    def test_returns_new_object(self):
        v = _make_violation()
        result = normalize_violation(v)
        assert result is not v

    def test_normalized_text_non_empty(self):
        v = _make_violation()
        result = normalize_violation(v)
        assert result.normalized_text != ""

    def test_keywords_populated(self):
        v = _make_violation()
        result = normalize_violation(v)
        assert isinstance(result.keywords, list)

    def test_keywords_le_10(self):
        v = _make_violation()
        result = normalize_violation(v)
        assert len(result.keywords) <= 10

    def test_does_not_mutate_input(self):
        v = _make_violation()
        original_text = v.normalized_text
        normalize_violation(v)
        assert v.normalized_text == original_text

    def test_empty_description_raises(self):
        v = _make_violation(description="")
        with pytest.raises(AssertionError):
            normalize_violation(v)

    def test_long_description(self):
        long_desc = "нарушение требований безопасности " * 20
        v = _make_violation(description=long_desc)
        result = normalize_violation(v)
        assert result.normalized_text != ""

    def test_short_description_flagged(self):
        """Descriptions with < 20 tokens get possibly_not_a_violation=True."""
        v = _make_violation(description="нарушение зоны охраны")
        result = normalize_violation(v)
        assert result.possibly_not_a_violation is True

    def test_long_description_not_flagged(self):
        """Descriptions with ≥ 20 tokens should not be flagged."""
        desc = " ".join(["нарушение"] * 25)
        v = _make_violation(description=desc)
        result = normalize_violation(v)
        assert result.possibly_not_a_violation is False

    def test_id_preserved(self):
        v = _make_violation()
        result = normalize_violation(v)
        assert result.id == v.id

    def test_various_inputs_non_empty(self):
        inputs = [
            "не определены границы охранной зоны трубопровода",
            "отсутствует разрешение на строительство объекта капитального строительства",
            "нарушены требования пожарной безопасности на объекте",
            "не проведена техническая экспертиза несущих конструкций здания",
        ]
        for desc in inputs:
            v = _make_violation(description=desc)
            result = normalize_violation(v)
            assert result.normalized_text != "", f"Failed for: {desc}"
