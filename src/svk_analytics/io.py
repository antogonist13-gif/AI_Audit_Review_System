from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_html_excel(path: str | Path) -> bool:
    """Detects HTML files saved with .XLS extension."""
    path = Path(path)
    head = path.read_bytes()[:2048].lstrip().lower()
    return head.startswith(b"<") and (b"<table" in head or b"<style" in head or b"<html" in head)


def load_report(path: str | Path) -> pd.DataFrame:
    """Load annual SVK report.

    The current exported report may have .XLS extension while actually being an HTML table.
    This function supports both real Excel files and HTML-table exports.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    suffix = path.suffix.lower()

    if is_html_excel(path) or suffix in {".html", ".htm"}:
        tables = pd.read_html(path, encoding="utf-8")
        if not tables:
            raise ValueError(f"В HTML-файле не найдены таблицы: {path}")
        df = max(tables, key=lambda t: t.shape[0] * t.shape[1])
    elif suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {suffix}")

    df.columns = [str(c).strip().replace("\n", " ").replace("\r", " ") for c in df.columns]
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
