from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ColumnResolution:
    canonical: str
    label: str
    original: str | None
    found: bool


def _norm(text: str) -> str:
    text = str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def resolve_columns(df: pd.DataFrame, columns_config: dict[str, Any]) -> tuple[dict[str, str], pd.DataFrame]:
    """Resolve source report columns to stable canonical names using regex patterns."""
    raw_columns = list(df.columns)
    normalized_columns = {_norm(c): c for c in raw_columns}

    resolved: dict[str, str] = {}
    rows: list[ColumnResolution] = []

    for canonical, spec in columns_config["columns"].items():
        patterns = spec.get("patterns", [])
        label = spec.get("label", canonical)
        matched_col: str | None = None

        # First: exact normalized match, useful for stable columns.
        for pattern in patterns:
            for norm_col, raw_col in normalized_columns.items():
                if pattern == norm_col:
                    matched_col = raw_col
                    break
            if matched_col:
                break

        # Second: regex search, case-insensitive.
        if not matched_col:
            for pattern in patterns:
                regex = re.compile(pattern, flags=re.IGNORECASE)
                for raw_col in raw_columns:
                    if regex.search(_norm(raw_col)):
                        matched_col = raw_col
                        break
                if matched_col:
                    break

        if matched_col:
            resolved[canonical] = matched_col

        rows.append(ColumnResolution(canonical, label, matched_col, matched_col is not None))

    report = pd.DataFrame([r.__dict__ for r in rows])
    return resolved, report


def build_canonical_frame(df: pd.DataFrame, resolved: dict[str, str]) -> pd.DataFrame:
    """Create dataframe with stable canonical columns while preserving organization rows."""
    out = pd.DataFrame(index=df.index)
    for canonical, original in resolved.items():
        out[canonical] = df[original]
    return out
