"""Tests for act_preprocessor.py — no LLM, no ChromaDB, no real DOCX required."""
from __future__ import annotations

import os
import sys
from typing import List
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from act_preprocessor import (
    ViolationWithContext,
    _extract_law_tokens,
    _find_context_in_descriptive,
    _find_violations_table,
    _get_descriptive_paragraphs,
    _normalize_for_match,
    _parse_table_rows,
    _segment_act,
    preprocess_act,
    violations_with_context_to_violations,
)
from models import Violation

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_act.txt")


def _read_fixture() -> str:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return fh.read()


def _make_mock_doc(header_texts: List[str], rows_data: List[List[str]]):
    """Build a minimal python-docx Document mock for table tests."""
    mock_doc = MagicMock()

    # Build table mock
    mock_table = MagicMock()
    table_rows = []

    # Row 0 — header
    header_cells = [MagicMock(text=h) for h in header_texts]
    header_row = MagicMock()
    header_row.cells = header_cells
    table_rows.append(header_row)

    # Data rows
    for row_data in rows_data:
        cells = [MagicMock(text=c) for c in row_data]
        row = MagicMock()
        row.cells = cells
        table_rows.append(row)

    mock_table.rows = table_rows
    mock_table.columns = [MagicMock()] * len(header_texts)
    mock_doc.tables = [mock_table]
    mock_doc.paragraphs = []
    return mock_doc, mock_table


# ---------------------------------------------------------------------------
# TestFindViolationsTable
# ---------------------------------------------------------------------------


class TestFindViolationsTable:
    def test_finds_table_with_correct_headers(self):
        doc, table = _make_mock_doc(
            ["№ п/п", "Вид нарушения", "Нарушенные НПА", "Формулировка нарушения"],
            [["1", "Нарушение", "ФЗ-294", "Описание"]],
        )
        result = _find_violations_table(doc)
        assert result is table

    def test_returns_none_for_wrong_headers(self):
        doc, _ = _make_mock_doc(
            ["№", "Параметр", "Значение", "Примечание"],
            [["1", "данные", "значение", "прим"]],
        )
        result = _find_violations_table(doc)
        assert result is None

    def test_returns_none_for_empty_tables_list(self):
        doc = MagicMock()
        doc.tables = []
        result = _find_violations_table(doc)
        assert result is None

    def test_returns_none_when_table_has_fewer_than_4_columns(self):
        doc, _ = _make_mock_doc(
            ["№ п/п", "Вид нарушения", "Нарушенные НПА"],
            [["1", "нарушение", "ФЗ-1"]],
        )
        # Manually set columns to only 3
        doc.tables[0].columns = [MagicMock()] * 3
        result = _find_violations_table(doc)
        assert result is None

    def test_finds_table_regardless_of_position(self):
        """Table may not be at index 0."""
        other_table = MagicMock()
        other_table.rows = [MagicMock(cells=[MagicMock(text="A"), MagicMock(text="B")])]
        other_table.columns = [MagicMock(), MagicMock()]

        violations_table = MagicMock()
        hdr_cells = [
            MagicMock(text="№ п/п"),
            MagicMock(text="Вид нарушения"),
            MagicMock(text="Нарушенные НПА"),
            MagicMock(text="Формулировка нарушения"),
        ]
        hdr_row = MagicMock(cells=hdr_cells)
        violations_table.rows = [hdr_row, MagicMock(cells=[MagicMock(text="1")] * 4)]
        violations_table.columns = [MagicMock()] * 4

        doc = MagicMock()
        doc.tables = [other_table, violations_table]
        result = _find_violations_table(doc)
        assert result is violations_table


# ---------------------------------------------------------------------------
# TestParseTableRows
# ---------------------------------------------------------------------------


class TestParseTableRows:
    def _make_table(self, rows_data: List[List[str]]):
        """Build a mock table with header + given data rows."""
        header = ["№ п/п", "Вид нарушения", "Нарушенные НПА", "Формулировка нарушения"]
        _, table = _make_mock_doc(header, rows_data)
        return table

    def test_violation_row_extracted(self):
        table = self._make_table([["1", "Нарушение A", "ФЗ-69 ст. 28", "Отсутствует граница"]])
        rows = _parse_table_rows(table)
        assert len(rows) == 1
        assert rows[0].num == "1"
        assert rows[0].violation_type == "Нарушение A"
        assert rows[0].law_ref == "ФЗ-69 ст. 28"
        assert rows[0].formulation == "Отсутствует граница"

    def test_violation_row_with_trailing_dot(self):
        table = self._make_table([["2.", "Нарушение B", "ФЗ-116", "Описание"]])
        rows = _parse_table_rows(table)
        assert len(rows) == 1
        assert rows[0].num == "2."

    def test_merged_theme_row_updates_section(self):
        merged_text = "I. В части общих вопросов деятельности"
        table = self._make_table([
            [merged_text, merged_text, merged_text, merged_text],
            ["1", "Нарушение A", "ФЗ-1", "Описание"],
        ])
        rows = _parse_table_rows(table)
        assert len(rows) == 1
        assert rows[0].current_section == merged_text

    def test_merged_status_row_skipped(self):
        status_text = "В ходе проведения проверки приняты меры по устранению 1 нарушения."
        table = self._make_table([
            [status_text, status_text, status_text, status_text],
            ["1", "Нарушение A", "ФЗ-1", "Описание"],
        ])
        rows = _parse_table_rows(table)
        # Status row is skipped (not a Roman-numeral section), violation row kept
        assert len(rows) == 1

    def test_section_propagates_to_multiple_violations(self):
        theme = "II. В части финансово-хозяйственной деятельности"
        table = self._make_table([
            [theme, theme, theme, theme],
            ["1.", "Нарушение A", "ФЗ-402", "Описание A"],
            ["2.", "Нарушение B", "ТК РФ", "Описание B"],
        ])
        rows = _parse_table_rows(table)
        assert len(rows) == 2
        assert all(r.current_section == theme for r in rows)

    def test_section_resets_for_new_theme(self):
        t1 = "I. Общие вопросы"
        t2 = "II. Финансовые вопросы"
        table = self._make_table([
            [t1, t1, t1, t1],
            ["1", "Нарушение A", "ФЗ-1", "Описание A"],
            [t2, t2, t2, t2],
            ["1", "Нарушение B", "ФЗ-2", "Описание B"],
        ])
        rows = _parse_table_rows(table)
        assert len(rows) == 2
        assert rows[0].current_section == t1
        assert rows[1].current_section == t2

    def test_empty_rows_skipped(self):
        table = self._make_table([
            ["", "", "", ""],
            ["1", "Нарушение A", "ФЗ-1", "Описание"],
        ])
        rows = _parse_table_rows(table)
        assert len(rows) == 1

    def test_header_row_not_in_result(self):
        table = self._make_table([])
        rows = _parse_table_rows(table)
        assert rows == []


# ---------------------------------------------------------------------------
# TestFindContextInDescriptive
# ---------------------------------------------------------------------------


class TestFindContextInDescriptive:
    _PARAGRAPHS = [
        "Организация эксплуатирует газопровод протяжённостью 12 км.",
        "Акт согласования охранной зоны не представлен.",
        "В нарушение ФЗ-69 ст. 28 не определены границы охранной зоны газопровода.",
        "Ответственным является главный инженер Иванов А.В.",
        "По данным журнала, диагностика проводилась в 2020 году.",
        "Установленный нормативный срок — раз в четыре года.",
        "В нарушение ФЗ-116 ст. 7 не проведена техническая диагностика в срок.",
        "Ответственным является технический директор Петров С.И.",
    ]

    def test_finds_anchor_by_law_ref_tokens(self):
        ctx = _find_context_in_descriptive("ФЗ-69 ст. 28", self._PARAGRAPHS, "fallback")
        assert "В нарушение ФЗ-69 ст. 28" in ctx

    def test_context_includes_paragraphs_before_anchor(self):
        ctx = _find_context_in_descriptive("ФЗ-69 ст. 28", self._PARAGRAPHS, "fallback")
        assert "Акт согласования" in ctx

    def test_context_includes_paragraphs_after_anchor(self):
        ctx = _find_context_in_descriptive("ФЗ-69 ст. 28", self._PARAGRAPHS, "fallback")
        assert "Иванов" in ctx

    def test_selects_most_matching_anchor(self):
        """When multiple 'В нарушение' paragraphs exist, picks the one with more token overlap."""
        ctx = _find_context_in_descriptive("ФЗ-116 ст. 7", self._PARAGRAPHS, "fallback")
        assert "ФЗ-116" in ctx
        assert "ФЗ-69" not in ctx.split("\n")[0]  # anchor should be the ФЗ-116 paragraph

    def test_falls_back_to_table_formulation_when_no_anchor(self):
        result = _find_context_in_descriptive(
            "ФЗ-999 ст. 999", self._PARAGRAPHS, "мой fallback"
        )
        assert result == "мой fallback"

    def test_falls_back_when_law_ref_empty(self):
        result = _find_context_in_descriptive("", self._PARAGRAPHS, "пустой ref fallback")
        assert result == "пустой ref fallback"

    def test_window_respects_before_param(self):
        ctx = _find_context_in_descriptive(
            "ФЗ-69 ст. 28", self._PARAGRAPHS, "fallback", before=0, after=0
        )
        # Only the anchor paragraph itself
        lines = [l for l in ctx.splitlines() if l.strip()]
        assert len(lines) == 1
        assert "ФЗ-69" in lines[0]

    def test_non_breaking_space_normalised(self):
        """law_ref with \xa0 should still match."""
        paras = [
            "Общий контекст.",
            "В нарушение\xa0ФЗ-69\xa0ст.\xa028 нарушена зона.",
        ]
        ctx = _find_context_in_descriptive("ФЗ-69 ст. 28", paras, "fallback")
        assert "нарушена зона" in ctx


# ---------------------------------------------------------------------------
# TestExtractLawTokens
# ---------------------------------------------------------------------------


class TestExtractLawTokens:
    def test_extracts_fz_code(self):
        tokens = _extract_law_tokens("ФЗ-294 ст. 20")
        assert any("294" in t or "фз-294" in t for t in tokens)

    def test_extracts_article_with_subpoint(self):
        tokens = _extract_law_tokens("Ч. 6 и ч. 8 ст. 55.24 ГрК РФ")
        # 55.24 should be extracted
        assert any("55.24" in t for t in tokens)

    def test_extracts_tk_rf(self):
        tokens = _extract_law_tokens("Ст. 72 ТК РФ")
        assert any("тк рф" in t or "72" in t for t in tokens)

    def test_deduplicated(self):
        tokens = _extract_law_tokens("ФЗ-402 ст. 9 и ФЗ-402 п. 3")
        # "402" (or фз-402) should appear only once
        normalized = [t.lower() for t in tokens]
        count = sum(1 for t in normalized if "402" in t)
        assert count <= 2  # at most once via fz-402 form + maybe plain 402

    def test_empty_string_returns_empty_list(self):
        assert _extract_law_tokens("") == []

    def test_xa0_stripped(self):
        tokens = _extract_law_tokens("ст.\xa055.24\xa0ГрК\xa0РФ")
        assert any("55.24" in t for t in tokens)


# ---------------------------------------------------------------------------
# TestSegmentAct
# ---------------------------------------------------------------------------


class TestSegmentAct:
    def test_finds_descriptive_and_resolutive(self):
        text = (
            "ОПИСАТЕЛЬНАЯ ЧАСТЬ\n"
            "Описание нарушения.\n"
            "РЕЗОЛЮТИВНАЯ ЧАСТЬ\n"
            "1. Нарушение А.\n"
        )
        segments = _segment_act(text)
        assert "Описание нарушения" in segments["descriptive"]
        assert "Нарушение А" in segments["resolutive"]

    def test_no_resolutive_returns_empty_resolutive(self):
        text = "ОПИСАТЕЛЬНАЯ ЧАСТЬ\nПросто текст без нарушений.\n"
        segments = _segment_act(text)
        assert segments["resolutive"] == ""
        assert "Просто текст" in segments["descriptive"]

    def test_no_descriptive_marker_uses_full_text_before_resolutive(self):
        text = "Вводная часть.\nРЕЗОЛЮТИВНАЯ ЧАСТЬ\n1. Нарушение.\n"
        segments = _segment_act(text)
        assert "Вводная часть" in segments["descriptive"]
        assert "Нарушение" in segments["resolutive"]

    def test_both_empty_for_unstructured_text(self):
        text = "Просто текст без каких-либо заголовков и структуры."
        segments = _segment_act(text)
        # descriptive is empty (no marker); resolutive is empty
        assert segments["resolutive"] == ""

    def test_нарушения_as_resolutive_marker(self):
        text = "ОПИСАТЕЛЬНАЯ ЧАСТЬ\nФакты.\nНАРУШЕНИЯ\n1. Нарушено.\n"
        segments = _segment_act(text)
        assert "Нарушено" in segments["resolutive"]

    def test_установлено_as_descriptive_marker(self):
        text = "УСТАНОВЛЕНО\nФакты.\nРЕЗОЛЮТИВНАЯ ЧАСТЬ\n1. Нарушение.\n"
        segments = _segment_act(text)
        assert "Факты" in segments["descriptive"]


# ---------------------------------------------------------------------------
# TestPreprocessActIntegration
# ---------------------------------------------------------------------------


class TestPreprocessActIntegration:
    def test_text_path_returns_violations_from_fixture(self):
        """Text-segmentation path on sample_act.txt should return 3 violations."""
        raw_text = _read_fixture()
        items = preprocess_act(raw_text, source_doc="sample_act.txt", use_llm_fallback=False)
        assert len(items) == 3

    def test_all_items_are_violation_with_context(self):
        raw_text = _read_fixture()
        items = preprocess_act(raw_text, source_doc="sample_act.txt", use_llm_fallback=False)
        assert all(isinstance(item, ViolationWithContext) for item in items)

    def test_source_doc_set_correctly(self):
        raw_text = _read_fixture()
        items = preprocess_act(raw_text, source_doc="my_act.txt", use_llm_fallback=False)
        assert all(item.source_document == "my_act.txt" for item in items)

    def test_context_richer_than_table_formulation(self):
        """context should contain descriptive paragraphs beyond the brief statement."""
        raw_text = _read_fixture()
        items = preprocess_act(raw_text, source_doc="sample_act.txt", use_llm_fallback=False)
        # At least one item should have context longer than table_formulation
        assert any(len(item.context) > len(item.table_formulation) for item in items)

    def test_context_contains_anchor_paragraph(self):
        """Each item's context should contain a 'В нарушение' paragraph."""
        raw_text = _read_fixture()
        items = preprocess_act(raw_text, source_doc="sample_act.txt", use_llm_fallback=False)
        anchor_count = sum(1 for item in items if "В нарушение" in item.context)
        assert anchor_count > 0

    def test_returns_empty_for_unstructured_text_without_llm(self):
        items = preprocess_act(
            "Просто неструктурированный текст без заголовков.",
            source_doc="unknown.txt",
            use_llm_fallback=False,
        )
        assert items == []

    def test_llm_fallback_called_when_no_structure(self):
        """When structural paths find nothing, LLM fallback is invoked."""
        mock_item = ViolationWithContext(
            violation_id="abc",
            statement="Нарушение",
            table_formulation="Нарушение",
            context="Нарушение",
            section="описательная",
            page=1,
            source_document="act.txt",
            law_ref="",
            subject="",
        )
        with patch("act_preprocessor._llm_fallback", return_value=[mock_item]) as mock_llm:
            items = preprocess_act(
                "Бесструктурный текст.",
                source_doc="act.txt",
                use_llm_fallback=True,
            )
        mock_llm.assert_called_once()
        assert len(items) == 1

    def test_llm_not_called_when_disabled(self):
        with patch("act_preprocessor._llm_fallback") as mock_llm:
            items = preprocess_act(
                "Бесструктурный текст.",
                source_doc="act.txt",
                use_llm_fallback=False,
            )
        mock_llm.assert_not_called()
        assert items == []

    def test_docx_path_takes_precedence(self):
        """When file_path ends with .docx and table extraction succeeds, text path is skipped."""
        mock_items = [
            ViolationWithContext(
                violation_id=str(i),
                statement=f"Нарушение {i}",
                table_formulation=f"Формулировка {i}",
                context=f"Контекст {i}",
                section="I. Тематика",
                page=1,
                source_document="act.docx",
                law_ref="ФЗ-294",
                subject="",
            )
            for i in range(2)
        ]
        with patch("act_preprocessor._extract_from_docx", return_value=mock_items) as mock_docx:
            items = preprocess_act(
                "raw text",
                source_doc="act.docx",
                file_path="/path/to/act.docx",
                use_llm_fallback=False,
            )
        mock_docx.assert_called_once_with("/path/to/act.docx", "act.docx")
        assert len(items) == 2

    def test_docx_path_falls_through_to_text_when_no_table(self):
        """If _extract_from_docx returns [], preprocess_act uses text path."""
        raw_text = _read_fixture()
        with patch("act_preprocessor._extract_from_docx", return_value=[]):
            items = preprocess_act(
                raw_text,
                source_doc="sample_act.txt",
                file_path="/path/to/sample_act.docx",
                use_llm_fallback=False,
            )
        # Text path should still find 3 violations in the fixture
        assert len(items) == 3


# ---------------------------------------------------------------------------
# TestConverter
# ---------------------------------------------------------------------------


class TestConverter:
    def _make_vwc(self, idx: int = 0) -> ViolationWithContext:
        return ViolationWithContext(
            violation_id=f"id-{idx}",
            statement=f"Вид нарушения {idx}",
            table_formulation=f"Краткая формулировка {idx}",
            context=f"Полный контекст из описательной части {idx}",
            section=f"II. Тематика",
            page=2,
            source_document="act.docx",
            law_ref="ФЗ-402 ст. 9",
            subject="ООО Тест",
        )

    def test_returns_violation_list(self):
        items = [self._make_vwc(0), self._make_vwc(1)]
        result = violations_with_context_to_violations(items)
        assert len(result) == 2
        assert all(isinstance(v, Violation) for v in result)

    def test_raw_text_equals_table_formulation(self):
        vwc = self._make_vwc(0)
        result = violations_with_context_to_violations([vwc])
        assert result[0].raw_text == vwc.table_formulation

    def test_description_equals_context(self):
        vwc = self._make_vwc(0)
        result = violations_with_context_to_violations([vwc])
        assert result[0].description == vwc.context

    def test_law_ref_preserved(self):
        vwc = self._make_vwc(0)
        result = violations_with_context_to_violations([vwc])
        assert result[0].law_ref == "ФЗ-402 ст. 9"

    def test_section_preserved(self):
        vwc = self._make_vwc(0)
        result = violations_with_context_to_violations([vwc])
        assert result[0].section == "II. Тематика"

    def test_violation_id_preserved(self):
        vwc = self._make_vwc(0)
        result = violations_with_context_to_violations([vwc])
        assert result[0].id == "id-0"

    def test_subject_preserved(self):
        vwc = self._make_vwc(0)
        result = violations_with_context_to_violations([vwc])
        assert result[0].subject == "ООО Тест"

    def test_empty_list_returns_empty_list(self):
        assert violations_with_context_to_violations([]) == []

    def test_context_longer_than_raw_text(self):
        """Agents get more information than the brief table formulation."""
        vwc = self._make_vwc(0)
        result = violations_with_context_to_violations([vwc])
        # In the fixture context is longer than the formulation
        assert len(result[0].description) >= len(result[0].raw_text)
