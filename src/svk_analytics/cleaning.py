from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


MISSING_VALUES = {"", "-", "—", "nan", "none", "нет данных", "не заполнено", "не указано"}


def clean_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if text.lower() in MISSING_VALUES:
        return None
    return text


def yes_flag(series: pd.Series) -> pd.Series:
    """Convert Russian yes/no answers to boolean-like 1/0 with NA."""
    s = series.map(clean_text)
    if s.dtype == 'object' or hasattr(s.dtype, 'pyarrow_dtype'):
        s = s.astype(str)
    s = s.str.lower().str.strip()
    result = pd.Series(np.nan, index=series.index, dtype="float")
    result[s.str.startswith("да", na=False)] = 1.0
    result[s.str.startswith("нет", na=False)] = 0.0
    return result


def to_number(series: pd.Series) -> pd.Series:
    """Robust numeric conversion for Russian-formatted values."""
    def _one(value):
        if pd.isna(value):
            return np.nan
        text = str(value).replace("\xa0", " ").strip()
        text = re.sub(r"\s+", "", text)
        text = text.replace(",", ".")
        if text.lower() in MISSING_VALUES:
            return np.nan
        # Keep only numeric characters, minus and decimal point.
        text = re.sub(r"[^0-9.\-]", "", text)
        if text in {"", "-", "."}:
            return np.nan
        try:
            return float(text)
        except ValueError:
            return np.nan

    return series.map(_one).astype(float)


def clean_canonical_frame(df: pd.DataFrame, numeric_columns: Iterable[str], yes_no_columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in numeric_columns:
            out[col] = to_number(out[col])
        elif col in yes_no_columns:
            out[col + "__flag"] = yes_flag(out[col])
        else:
            out[col] = out[col].map(clean_text)
    return out
