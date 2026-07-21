from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

ReportFormat = Literal["html", "excel"]

# Маркеры строки заголовков сводного отчета СВК
_HEADER_MARKERS = (
    "№ п/п",
    "Наименование организации",
    "Статус отчета",
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_html_excel(path: str | Path) -> bool:
    """Detects HTML files saved with .XLS / .xls extension."""
    path = Path(path)
    head = path.read_bytes()[:2048].lstrip().lower()
    return head.startswith(b"<") and (b"<table" in head or b"<style" in head or b"<html" in head)


def _excel_engine(path: Path) -> str | None:
    """Pick pandas engine by content. .XLS may actually be xlsx (ZIP/PK)."""
    magic = path.read_bytes()[:8]
    if magic.startswith(b"PK"):
        return "openpyxl"
    if magic.startswith(b"\xd0\xcf\x11\xe0"):
        try:
            import xlrd  # noqa: F401

            return "xlrd"
        except ImportError:
            return None
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return "openpyxl"
    if suffix == ".xls":
        return None
    return None


def detect_report_format(path: str | Path) -> ReportFormat:
    """Determine report format by content first, then by extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    magic = path.read_bytes()[:8]
    is_html = is_html_excel(path)

    if is_html or suffix in {".html", ".htm"}:
        return "html"

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return "excel"

    if magic.startswith(b"PK"):
        return "excel"
    if magic.startswith(b"\xd0\xcf\x11\xe0"):
        return "excel"

    raise ValueError(f"Неподдерживаемый формат файла: {path.name}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", " ").replace("\r", " ") for c in df.columns]
    return df


def _row_looks_like_header(values: list[Any]) -> bool:
    texts = [str(v).strip() for v in values if pd.notna(v) and str(v).strip()]
    if not texts:
        return False
    hits = sum(1 for marker in _HEADER_MARKERS if any(marker in t for t in texts))
    return hits >= 2


def _has_svk_header_columns(df: pd.DataFrame) -> bool:
    cols = [str(c) for c in df.columns]
    hits = sum(1 for marker in _HEADER_MARKERS if any(marker in c for c in cols))
    return hits >= 2


def detect_excel_header_row(path: str | Path, *, max_scan_rows: int = 30, engine: str | None = None) -> int:
    """Find the header row index in Excel reports that include a title block."""
    preview = pd.read_excel(path, header=None, nrows=max_scan_rows, engine=engine)
    for idx in range(len(preview)):
        if _row_looks_like_header(preview.iloc[idx].tolist()):
            return int(idx)
    return 0


def _load_html_report(path: Path) -> pd.DataFrame:
    tables = pd.read_html(path, encoding="utf-8")
    if not tables:
        raise ValueError(f"В HTML-файле не найдены таблицы: {path}")
    return max(tables, key=lambda t: t.shape[0] * t.shape[1])


def _load_excel_report(path: Path) -> pd.DataFrame:
    engine = _excel_engine(path)
    header_row = detect_excel_header_row(path, engine=engine)
    df = pd.read_excel(path, header=header_row, engine=engine)
    if not _has_svk_header_columns(df):
        raise ValueError(
            "Не удалось найти строку заголовков СВК в Excel-файле "
            f"(ожидались колонки вроде «Наименование организации», «Статус отчета»). "
            f"Пробовали header={header_row}, engine={engine}."
        )
    return df


def load_report(path: str | Path) -> pd.DataFrame:
    """Load annual SVK report.

    Supports:
    - HTML-table exports saved as .XLS / .html
    - Real Excel files (.xlsx / .xls), including those with a title block
      above the actual column headers
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    fmt = detect_report_format(path)
    if fmt == "html":
        df = _load_html_report(path)
    else:
        df = _load_excel_report(path)

    df = _normalize_columns(df)

    if fmt == "excel" and not _has_svk_header_columns(df):
        raise ValueError(
            "Excel-файл прочитан, но колонки отчёта СВК не распознаны. "
            "Проверьте, что это сводный отчёт мониторинга."
        )

    return df


def find_latest_raw_file(raw_dir: str | Path = "data/raw") -> Path | None:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None
    candidates = []
    for pattern in ("*.xls", "*.XLS", "*.xlsx", "*.XLSX", "*.html", "*.htm"):
        candidates.extend(raw_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
