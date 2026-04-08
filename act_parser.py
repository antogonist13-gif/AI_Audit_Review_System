"""Act parser — two-pass violation extraction (deterministic → LLM fallback)."""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import replace
from typing import List, Optional, Pattern

from models import Violation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns — compiled once at import
# ---------------------------------------------------------------------------

# Section header detecting the body of the act
_SECTION_HEADER_RE = re.compile(
    r"(?im)^(описательная часть|мотивировочная часть|установлено|нарушения)[:\s]*$"
)

# Numbered item: "1. Нарушение ...", "1) Нарушение ..."
_NUMBERED_ITEM_RE = re.compile(
    r"(?m)^(\d+)[.)]\s+(.{20,}?)(?=\n\d+[.)]\s|\Z)",
    re.DOTALL,
)

# Keyword-based violation sentence
_VIOLATION_KEYWORD_RE = re.compile(
    r"(?i)(?:нарушение|нарушен[оаы]?|не\s+(?:соблюден|выполнен|обеспечен|определен|представлен)|"
    r"отсутствует|не\s+имеет|несоответствие|превышение)[^.!?]{10,200}[.!?]",
    re.UNICODE,
)

# Law reference in text
_LAW_REF_RE = re.compile(
    r"(?:Федеральный закон|ФЗ|Постановление|приказ|ГОСТ|СП|СНиП|КоАП)"
    r"[\s\-—]*(?:№\s*)?[\d\-]+[^\s,;.]{0,40}",
    re.IGNORECASE | re.UNICODE,
)

# Subject (organisation) patterns
_SUBJECT_RE = re.compile(
    r"(?:ООО|АО|ПАО|ГУП|МУП|ИП|ФГУП)[^,;\n]{3,60}",
    re.UNICODE | re.IGNORECASE,
)


def _regex_patterns() -> List[Pattern]:
    return [_NUMBERED_ITEM_RE, _VIOLATION_KEYWORD_RE]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_law_ref(text: str) -> str:
    m = _LAW_REF_RE.search(text)
    return m.group(0).strip() if m else ""


def _extract_subject(text: str) -> str:
    m = _SUBJECT_RE.search(text)
    return m.group(0).strip() if m else ""


def _detect_section(pos: int, text: str) -> str:
    """Return the section header that precedes pos in text."""
    preceding = text[:pos]
    headers = list(_SECTION_HEADER_RE.finditer(preceding))
    if headers:
        return headers[-1].group(1).strip()
    return "описательная"


def _detect_page(pos: int, text: str) -> int:
    """Estimate page number by counting page-break markers before pos."""
    page_breaks = text[:pos].count("\f") + text[:pos].count("[PAGE]")
    return page_breaks + 1


def _parse_violation_block(
    block_text: str,
    source_doc: str,
    page: int,
    section: str,
) -> Optional[Violation]:
    """Parse a single text block into a Violation or None."""
    block_text = block_text.strip()
    if len(block_text) < 20:
        return None

    law_ref = _extract_law_ref(block_text)
    subject = _extract_subject(block_text)

    return Violation(
        raw_text=block_text,
        source_document=source_doc,
        page=page,
        section=section,
        description=block_text[:500],
        subject=subject,
        law_ref=law_ref,
    )


def _assign_ids(violations: List[Violation]) -> List[Violation]:
    """Ensure every violation has a non-empty UUID."""
    result = []
    for v in violations:
        if not v.id:
            result.append(replace(v, id=str(uuid.uuid4())))
        else:
            result.append(v)
    return result


# ---------------------------------------------------------------------------
# Two-pass extraction
# ---------------------------------------------------------------------------


def _extract_deterministic(text: str, source_doc: str) -> List[Violation]:
    """First pass: regex-based extraction."""
    violations: List[Violation] = []

    # Try numbered items first (most structured)
    for m in _NUMBERED_ITEM_RE.finditer(text):
        block = m.group(2).strip()
        page = _detect_page(m.start(), text)
        section = _detect_section(m.start(), text)
        v = _parse_violation_block(block, source_doc, page, section)
        if v is not None:
            violations.append(v)

    # If numbered items found nothing, try keyword sentences
    if not violations:
        for m in _VIOLATION_KEYWORD_RE.finditer(text):
            block = m.group(0).strip()
            page = _detect_page(m.start(), text)
            section = _detect_section(m.start(), text)
            v = _parse_violation_block(block, source_doc, page, section)
            if v is not None:
                violations.append(v)

    return violations


def _extract_llm(text: str, source_doc: str) -> List[Violation]:
    """Second pass (fallback): LLM-based extraction via rag_pipeline."""
    from prompts import ACT_EXTRACTION_TEMPLATE, parse_extraction_blocks
    from rag_pipeline import get_pipeline

    pipeline = get_pipeline()

    # Split text into chunks to fit context window
    chunk_size = 3000
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    violations: List[Violation] = []
    for chunk in text_chunks:
        prompt = ACT_EXTRACTION_TEMPLATE.format(act_text=chunk)
        try:
            response = pipeline.analyze(prompt)
        except Exception as exc:
            logger.error("LLM extraction failed for chunk: %s", exc)
            continue

        blocks = parse_extraction_blocks(response)
        for block in blocks:
            v = Violation(
                raw_text=block.get("violation", ""),
                source_document=source_doc,
                page=1,
                section=block.get("section", "описательная"),
                description=block.get("violation", ""),
                subject=block.get("subject", ""),
                law_ref=block.get("law_ref", ""),
            )
            if v.description:
                violations.append(v)

    return violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_violations(act_text: str, source_doc: str) -> List[Violation]:
    """Extract violations from act text using two-pass strategy.

    First tries regex (deterministic). Falls back to LLM only if regex
    finds zero violations.
    """
    results = _extract_deterministic(act_text, source_doc)

    if len(results) < 1:
        logger.warning(
            '{"event": "PARSER_FALLBACK_TO_LLM", "source": "%s"}',
            source_doc,
        )
        results = _extract_llm(act_text, source_doc)
    else:
        logger.info(
            '{"event": "VIOLATION_EXTRACTED", "source": "%s", "count": %d, "method": "deterministic"}',
            source_doc, len(results),
        )

    return _assign_ids(results)
