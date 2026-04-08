"""Prompts module — parsers first, then LLM prompt templates.

Parser functions are implemented first because they define the output format
that the templates must produce.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Score word mapping
# ---------------------------------------------------------------------------

_WORD_TO_NUM: Dict[str, float] = {
    "ноль": 0, "один": 1, "два": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# ---------------------------------------------------------------------------
# Score patterns — ordered from most to least specific
# ---------------------------------------------------------------------------

_SCORE_PATTERNS = [
    # "БАЛЛ: 7/10" or "ОЦЕНКА: 7 / 10"
    (
        re.compile(r"(?:БАЛЛ|ОЦЕНКА|SCORE)[:\s]+(\d+(?:[.,]\d+)?)\s*/\s*10", re.IGNORECASE),
        lambda m: float(m.group(1).replace(",", ".")) / 10,
    ),
    # "БАЛЛ: 0.75" or "БАЛЛ: 0,75" — already in [0,1]
    (
        re.compile(r"(?:БАЛЛ|ОЦЕНКА|SCORE)[:\s]+(0[.,]\d+)", re.IGNORECASE),
        lambda m: float(m.group(1).replace(",", ".")),
    ),
    # "БАЛЛ: 7" — integer, treat as /10
    (
        re.compile(r"(?:БАЛЛ|ОЦЕНКА|SCORE)[:\s]+(\d+(?:[.,]\d+)?)\b", re.IGNORECASE),
        lambda m: (
            lambda v: v / 10 if v > 1.0 else v
        )(float(m.group(1).replace(",", "."))),
    ),
    # Word-form: "БАЛЛ: семь"
    (
        re.compile(
            r"(?:БАЛЛ|ОЦЕНКА|SCORE)[:\s]+("
            + "|".join(_WORD_TO_NUM.keys())
            + r")\b",
            re.IGNORECASE,
        ),
        lambda m: _WORD_TO_NUM.get(m.group(1).lower(), 0) / 10,
    ),
]


def _parse_score(text: str) -> float:
    """Extract a normalised [0,1] score from LLM output. Returns 0.0 on failure."""
    for pattern, converter in _SCORE_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                value = converter(m)
                return min(1.0, max(0.0, value))
            except (ValueError, KeyError):
                continue
    return 0.0


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def parse_scored_response(text: str) -> Dict[str, str]:
    """Parse a scored LLM evaluation response.

    Returns a dict with keys: 'score' (str float), 'comment'.
    Falls back gracefully — never raises.
    """
    score = _parse_score(text)

    comment_match = re.search(
        r"(?:КОММЕНТАРИЙ|COMMENT|ОБОСНОВАНИЕ|ВЫВОД)[:\s]+(.+?)(?:\n|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    comment = comment_match.group(1).strip() if comment_match else text.strip()

    return {"score": str(score), "comment": comment}


def parse_extraction_blocks(text: str) -> List[Dict[str, str]]:
    """Parse LLM extraction output into a list of violation dicts.

    Blocks are separated by '---'. Each block may contain:
    НАРУШЕНИЕ:, СУБЪЕКТ:, НОРМА:, РАЗДЕЛ:
    """
    blocks = re.split(r"\n---+\n", text)
    results: List[Dict[str, str]] = []

    field_patterns = {
        "violation": re.compile(r"НАРУШЕНИЕ[:\s]+(.+?)(?=\n[А-ЯA-Z]|$)", re.DOTALL | re.IGNORECASE),
        "subject": re.compile(r"СУБЪЕКТ[:\s]+(.+?)(?=\n[А-ЯA-Z]|$)", re.DOTALL | re.IGNORECASE),
        "law_ref": re.compile(r"НОРМА[:\s]+(.+?)(?=\n[А-ЯA-Z]|$)", re.DOTALL | re.IGNORECASE),
        "section": re.compile(r"РАЗДЕЛ[:\s]+(.+?)(?=\n[А-ЯA-Z]|$)", re.DOTALL | re.IGNORECASE),
    }

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        parsed: Dict[str, str] = {}
        for key, pattern in field_patterns.items():
            m = pattern.search(block)
            parsed[key] = m.group(1).strip() if m else ""
        if parsed.get("violation"):
            results.append(parsed)

    return results


def parse_improvement_response(text: str) -> Dict[str, str]:
    """Parse the formulation improvement LLM response.

    Returns dict with: improved_text, legal_qualification, justification.
    """
    patterns = {
        "improved_text": re.compile(
            r"УЛУЧШЕННАЯ[_\s]ФОРМУЛИРОВКА[:\s]+(.+?)(?=\n[А-ЯA-Z_]|$)",
            re.DOTALL | re.IGNORECASE,
        ),
        "legal_qualification": re.compile(
            r"ПРАВОВАЯ[_\s]КВАЛИФИКАЦИЯ[:\s]+(.+?)(?=\n[А-ЯA-Z_]|$)",
            re.DOTALL | re.IGNORECASE,
        ),
        "justification": re.compile(
            r"ОБОСНОВАНИЕ[:\s]+(.+?)(?=\n[А-ЯA-Z_]|$)",
            re.DOTALL | re.IGNORECASE,
        ),
        "recommendation": re.compile(
            r"РЕКОМЕНДАЦИЯ[:\s]+(.+?)(?=\n[А-ЯA-Z_]|$)",
            re.DOTALL | re.IGNORECASE,
        ),
    }
    result: Dict[str, str] = {}
    for key, pattern in patterns.items():
        m = pattern.search(text)
        result[key] = m.group(1).strip() if m else ""
    return result


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ACT_EXTRACTION_TEMPLATE = """\
Ты — эксперт по проверке контрольно-надзорной деятельности.
Проанализируй текст акта проверки и выдели все нарушения.

Для каждого нарушения укажи:
НАРУШЕНИЕ: <описание нарушения>
СУБЪЕКТ: <наименование организации или ФИО нарушителя>
НОРМА: <нарушенная правовая норма (номер ФЗ, статья, пункт)>
РАЗДЕЛ: <раздел акта, в котором выявлено нарушение>

Разделяй нарушения строкой ---

ТЕКСТ АКТА:
{act_text}
"""

EVIDENCE_EVAL_TEMPLATE = """\
Оцени доказательность описания нарушения.
Ищи: конкретные даты, суммы, номера документов, ФИО, ссылки на нормы.

НАРУШЕНИЕ: {violation_description}
СУБЪЕКТ: {subject}

НОРМАТИВНЫЙ КОНТЕКСТ:
{norm_context}

КОРПУС ТИПОВЫХ НАРУШЕНИЙ:
{corpus_context}

Ответь строго в формате:
БАЛЛ: <число от 0 до 10>
КОММЕНТАРИЙ: <краткое обоснование оценки>
"""

LEGAL_EVAL_TEMPLATE = """\
Оцени правовую корректность квалификации нарушения.
Проверь: правильно ли указана норма, применима ли она к данной ситуации.

НАРУШЕНИЕ: {violation_description}
УКАЗАННАЯ НОРМА: {law_ref}

НОРМАТИВНЫЙ КОНТЕКСТ:
{norm_context}

Ответь строго в формате:
БАЛЛ: <число от 0 до 10>
КОММЕНТАРИЙ: <краткое обоснование оценки>
"""

ACTIONABILITY_EVAL_TEMPLATE = """\
Оцени исполнимость предписания по устранению нарушения.
Проверь: понятно ли что делать, кому, в какой срок.

НАРУШЕНИЕ: {violation_description}
СУБЪЕКТ: {subject}
НОРМА: {law_ref}

Ответь строго в формате:
БАЛЛ: <число от 0 до 10>
КОММЕНТАРИЙ: <краткое обоснование>
"""

FORMULATION_TEMPLATE = """\
Улучши формулировку нарушения для включения в официальный акт проверки.
Используй официально-деловой стиль, точные правовые ссылки.

ИСХОДНАЯ ФОРМУЛИРОВКА: {raw_text}
СУБЪЕКТ: {subject}
НОРМА: {law_ref}

НОРМАТИВНЫЙ КОНТЕКСТ:
{norm_context}

Ответь строго в формате:
УЛУЧШЕННАЯ_ФОРМУЛИРОВКА: <улучшенный текст нарушения>
ПРАВОВАЯ_КВАЛИФИКАЦИЯ: <точная квалификация с указанием статьи и закона>
ОБОСНОВАНИЕ: <краткое обоснование квалификации>
"""

RECOMMENDATION_TEMPLATE = """\
Сформулируй конкретную рекомендацию по устранению нарушения.

НАРУШЕНИЕ: {violation_description}
СУБЪЕКТ: {subject}
ПРАВОВАЯ КВАЛИФИКАЦИЯ: {legal_qualification}

НОРМАТИВНЫЙ КОНТЕКСТ:
{norm_context}

Ответь строго в формате:
РЕКОМЕНДАЦИЯ: <конкретные меры по устранению нарушения>
"""
