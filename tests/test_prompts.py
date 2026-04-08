"""Tests for prompts.py — 12+ score formats, parsers, fallback behaviour."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from prompts import (
    _parse_score,
    parse_extraction_blocks,
    parse_improvement_response,
    parse_scored_response,
)


# ---------------------------------------------------------------------------
# _parse_score — 12+ format variants
# ---------------------------------------------------------------------------


class TestParseScore:
    # -- numeric /10 formats --
    def test_integer_over_10(self):
        assert abs(_parse_score("БАЛЛ: 7/10") - 0.7) < 0.001

    def test_integer_over_10_with_spaces(self):
        assert abs(_parse_score("БАЛЛ: 7 / 10") - 0.7) < 0.001

    def test_float_over_10(self):
        assert abs(_parse_score("БАЛЛ: 7.5/10") - 0.75) < 0.001

    def test_float_over_10_comma(self):
        assert abs(_parse_score("БАЛЛ: 7,5/10") - 0.75) < 0.001

    # -- plain decimal [0,1] --
    def test_decimal_already_normalised(self):
        assert abs(_parse_score("БАЛЛ: 0.7") - 0.7) < 0.001

    def test_decimal_comma(self):
        assert abs(_parse_score("БАЛЛ: 0,7") - 0.7) < 0.001

    # -- plain integer (treated as /10) --
    def test_plain_integer_seven(self):
        assert abs(_parse_score("БАЛЛ: 7") - 0.7) < 0.001

    def test_plain_integer_ten(self):
        assert abs(_parse_score("БАЛЛ: 10") - 1.0) < 0.001

    def test_plain_integer_zero(self):
        assert _parse_score("БАЛЛ: 0") == 0.0

    # -- word forms --
    def test_word_семь(self):
        assert abs(_parse_score("БАЛЛ: семь") - 0.7) < 0.001

    def test_word_десять(self):
        assert abs(_parse_score("БАЛЛ: десять") - 1.0) < 0.001

    def test_word_один(self):
        assert abs(_parse_score("БАЛЛ: один") - 0.1) < 0.001

    def test_word_ноль(self):
        assert _parse_score("БАЛЛ: ноль") == 0.0

    # -- ОЦЕНКА keyword --
    def test_оценка_keyword(self):
        assert abs(_parse_score("ОЦЕНКА: 8") - 0.8) < 0.001

    # -- SCORE keyword (English) --
    def test_score_english_keyword(self):
        assert abs(_parse_score("SCORE: 6/10") - 0.6) < 0.001

    # -- fallback --
    def test_no_score_returns_zero(self):
        assert _parse_score("Это текст без оценки") == 0.0

    def test_empty_string_returns_zero(self):
        assert _parse_score("") == 0.0

    def test_fallback_never_raises(self):
        for text in ["БАЛЛ: abc", "БАЛЛ:", "БАЛЛ: -5", "БАЛЛ: 999"]:
            result = _parse_score(text)
            assert isinstance(result, float)

    # -- clamping --
    def test_clamped_to_max_1(self):
        assert _parse_score("БАЛЛ: 15") <= 1.0

    def test_clamped_to_min_0(self):
        assert _parse_score("БАЛЛ: 0") >= 0.0


# ---------------------------------------------------------------------------
# parse_scored_response
# ---------------------------------------------------------------------------


class TestParseScoredResponse:
    def test_returns_score_and_comment(self):
        text = "БАЛЛ: 7\nКОММЕНТАРИЙ: доказательства имеются"
        result = parse_scored_response(text)
        assert "score" in result
        assert "comment" in result
        assert abs(float(result["score"]) - 0.7) < 0.001
        assert "доказательства" in result["comment"]

    def test_missing_score_returns_zero(self):
        result = parse_scored_response("КОММЕНТАРИЙ: нет оценки")
        assert float(result["score"]) == 0.0

    def test_comment_fallback_to_full_text(self):
        result = parse_scored_response("просто текст без маркеров")
        assert result["comment"] != ""

    def test_never_raises(self):
        for text in ["", "БАЛЛ: abc", "---", "нет ничего"]:
            r = parse_scored_response(text)
            assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# parse_extraction_blocks
# ---------------------------------------------------------------------------


class TestParseExtractionBlocks:
    def test_single_block(self):
        text = "НАРУШЕНИЕ: не определены границы\nСУБЪЕКТ: ООО Тест\nНОРМА: ФЗ-294\nРАЗДЕЛ: описательная"
        blocks = parse_extraction_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["violation"] == "не определены границы"
        assert blocks[0]["subject"] == "ООО Тест"
        assert blocks[0]["law_ref"] == "ФЗ-294"

    def test_multiple_blocks_separated_by_dashes(self):
        text = (
            "НАРУШЕНИЕ: первое нарушение\nСУБЪЕКТ: ООО А\n"
            "\n---\n"
            "НАРУШЕНИЕ: второе нарушение\nСУБЪЕКТ: ООО Б\n"
        )
        blocks = parse_extraction_blocks(text)
        assert len(blocks) == 2

    def test_empty_block_skipped(self):
        text = "\n---\n\n---\nНАРУШЕНИЕ: реальное\nСУБЪЕКТ: ООО В\n"
        blocks = parse_extraction_blocks(text)
        assert len(blocks) == 1

    def test_block_without_violation_field_skipped(self):
        text = "СУБЪЕКТ: ООО Д\nНОРМА: ФЗ-294\n"
        blocks = parse_extraction_blocks(text)
        assert len(blocks) == 0

    def test_empty_text_returns_empty_list(self):
        assert parse_extraction_blocks("") == []

    def test_missing_optional_fields_returned_as_empty(self):
        text = "НАРУШЕНИЕ: нарушение без нормы\n"
        blocks = parse_extraction_blocks(text)
        assert blocks[0]["law_ref"] == ""
        assert blocks[0]["section"] == ""


# ---------------------------------------------------------------------------
# parse_improvement_response
# ---------------------------------------------------------------------------


class TestParseImprovementResponse:
    def test_all_fields_present(self):
        text = (
            "УЛУЧШЕННАЯ_ФОРМУЛИРОВКА: новый текст нарушения\n"
            "ПРАВОВАЯ_КВАЛИФИКАЦИЯ: ст. 20 ФЗ-294\n"
            "ОБОСНОВАНИЕ: подтверждено нормой\n"
            "РЕКОМЕНДАЦИЯ: устранить в течение 30 дней\n"
        )
        result = parse_improvement_response(text)
        assert result["improved_text"] == "новый текст нарушения"
        assert result["legal_qualification"] == "ст. 20 ФЗ-294"
        assert result["justification"] == "подтверждено нормой"
        assert result["recommendation"] == "устранить в течение 30 дней"

    def test_missing_fields_return_empty_string(self):
        result = parse_improvement_response("УЛУЧШЕННАЯ_ФОРМУЛИРОВКА: только это\n")
        assert result["legal_qualification"] == ""
        assert result["justification"] == ""

    def test_empty_text_all_empty(self):
        result = parse_improvement_response("")
        for v in result.values():
            assert v == ""

    def test_never_raises(self):
        for text in ["", "БАЛЛ: 7", "случайный текст"]:
            r = parse_improvement_response(text)
            assert isinstance(r, dict)
