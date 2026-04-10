"""Universal parser for LLM score responses (JSON, markdown fences, prose)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prose score patterns (aligned with prompts._SCORE_PATTERNS for /10 and [0,1])
# ---------------------------------------------------------------------------

_WORD_TO_NUM: Dict[str, float] = {
    "ноль": 0,
    "один": 1,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_SCORE_PATTERNS: List[Tuple[re.Pattern[str], Any]] = [
    (
        re.compile(
            r"(?:БАЛЛ|ОЦЕНКА|SCORE)[:\s]+(\d+(?:[.,]\d+)?)\s*/\s*10",
            re.IGNORECASE,
        ),
        lambda m: float(m.group(1).replace(",", ".")) / 10,
    ),
    (
        re.compile(r"(?:БАЛЛ|ОЦЕНКА|SCORE)[:\s]+(0[.,]\d+)", re.IGNORECASE),
        lambda m: float(m.group(1).replace(",", ".")),
    ),
    (
        re.compile(
            r"(?:БАЛЛ|ОЦЕНКА|SCORE)[:\s]+(\d+(?:[.,]\d+)?)\b",
            re.IGNORECASE,
        ),
        lambda m: (
            lambda v: v / 10 if v > 1.0 else v
        )(float(m.group(1).replace(",", "."))),
    ),
    (
        re.compile(
            r"(?:БАЛЛ|ОЦЕНКА|SCORE)[:\s]+("
            + "|".join(re.escape(k) for k in _WORD_TO_NUM.keys())
            + r")\b",
            re.IGNORECASE,
        ),
        lambda m: _WORD_TO_NUM.get(m.group(1).lower(), 0) / 10,
    ),
]

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# Loose float in prose (e.g. "Score is 0.7")
_FLOAT_IN_TEXT_RE = re.compile(
    r"(?<![\w.])(0?\.\d+|\d+[.,]\d+)(?![\w.])",
)


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _extract_score_from_prose(text: str) -> Optional[float]:
    for pattern, converter in _SCORE_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return _clamp01(float(converter(m)))
            except (ValueError, KeyError, ZeroDivisionError):
                continue
    for m in _FLOAT_IN_TEXT_RE.finditer(text):
        try:
            v = float(m.group(1).replace(",", "."))
            if 0.0 <= v <= 1.0:
                return v
            if 1.0 < v <= 10.0:
                return _clamp01(v / 10.0)
        except ValueError:
            continue
    return None


_SCORE_KEYS = (
    "score",
    "Score",
    "SCORE",
    "rating",
    "value",
    "балл",
    "Балл",
    "БАЛЛ",
    "ball",
)

_REASONING_KEYS = (
    "reasoning",
    "Reasoning",
    "comment",
    "Comment",
    "COMMENT",
    "explanation",
    "rationale",
    "обоснование",
    "комментарий",
    "КОММЕНТАРИЙ",
    "вывод",
    "ВЫВОД",
)


def _scalar_to_unit(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        if isinstance(v, str):
            s = v.strip().replace(",", ".")
            try:
                v = float(s)
            except ValueError:
                return None
        else:
            return None
    x = float(v)
    if 0.0 <= x <= 1.0:
        return x
    if 1.0 < x <= 10.0:
        return _clamp01(x / 10.0)
    if x > 10.0:
        return _clamp01(x / 10.0)
    if x < 0.0:
        return 0.0
    return None


def _score_from_obj(obj: Dict[str, Any]) -> Optional[float]:
    for k in _SCORE_KEYS:
        if k in obj and obj[k] is not None:
            s = _scalar_to_unit(obj[k])
            if s is not None:
                return s
    return None


def _reasoning_from_obj(obj: Dict[str, Any]) -> str:
    for k in _REASONING_KEYS:
        if k in obj and obj[k] is not None:
            r = obj[k]
            if isinstance(r, str):
                return r.strip()
            return str(r).strip()
    return ""


def _normalize_parsed_dict(obj: Any) -> Optional[Tuple[float, str]]:
    if not isinstance(obj, dict):
        return None
    score = _score_from_obj(obj)
    if score is None:
        return None
    return (_clamp01(score), _reasoning_from_obj(obj))


def _try_load_json(s: str) -> Optional[Any]:
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _extract_json_from_fences(raw: str) -> Optional[str]:
    m = _FENCE_RE.search(raw)
    if not m:
        return None
    return m.group(1).strip()


def _iter_balanced_json_objects(text: str):
    n = len(text)
    i = 0
    while i < n:
        if text[i] == "{":
            depth = 0
            start = i
            for j in range(i, n):
                c = text[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start : j + 1]
                        i = j + 1
                        break
            else:
                i += 1
        else:
            i += 1


def _parse_json_with_fallback(raw: str) -> Tuple[Optional[Tuple[float, str]], str]:
    """Returns (result, method) where method is direct|fence|embedded|none."""
    direct = _try_load_json(raw)
    if direct is not None:
        norm = _normalize_parsed_dict(direct)
        if norm is not None:
            return norm, "direct"

    fenced = _extract_json_from_fences(raw)
    if fenced:
        loaded = _try_load_json(fenced)
        if loaded is not None:
            norm = _normalize_parsed_dict(loaded)
            if norm is not None:
                return norm, "fence"

    for chunk in _iter_balanced_json_objects(raw):
        loaded = _try_load_json(chunk)
        if loaded is None:
            continue
        norm = _normalize_parsed_dict(loaded)
        if norm is not None:
            return norm, "embedded"

    return None, "none"


def parse_llm_score(raw: str, caller: str = "") -> dict:
    """Parse LLM output into ``{"score": float, "reasoning": str}`` (score in [0, 1])."""
    prefix = f"[{caller}] " if caller else ""
    logger.debug("%sraw LLM response: %r", prefix, raw)

    if raw is None:
        raw = ""
    text = raw if isinstance(raw, str) else str(raw)

    parsed, method = _parse_json_with_fallback(text)
    if parsed is not None:
        score, reasoning = parsed
        if method != "direct":
            logger.warning(
                "%sused non-direct JSON parse (method=%s)", prefix, method
            )
        return {"score": float(score), "reasoning": reasoning}

    prose_score = _extract_score_from_prose(text)
    if prose_score is not None:
        logger.warning("%sused prose/numeric fallback (extracted_from_prose)", prefix)
        return {
            "score": float(prose_score),
            "reasoning": "extracted_from_prose",
        }

    logger.warning("%sparse failed, using parse_error fallback", prefix)
    return {"score": 0.0, "reasoning": "parse_error"}
