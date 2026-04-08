"""Section parser — split raw act text into labelled sections."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Patterns that indicate the start of a named section in a Russian audit act
_SECTION_PATTERNS = [
    re.compile(r"(?m)^(ВВОДНАЯ ЧАСТЬ|вводная часть)[^\n]*$"),
    re.compile(r"(?m)^(ОПИСАТЕЛЬНАЯ ЧАСТЬ|описательная часть)[^\n]*$"),
    re.compile(r"(?m)^(МОТИВИРОВОЧНАЯ ЧАСТЬ|мотивировочная часть)[^\n]*$"),
    re.compile(r"(?m)^(РЕЗОЛЮТИВНАЯ ЧАСТЬ|резолютивная часть)[^\n]*$"),
    re.compile(r"(?m)^(\d+\.\s+[А-ЯЁ][^.]{5,80})[:\.]?\s*$"),  # numbered section headers
]


@dataclass
class Section:
    title: str
    text: str
    index: int = 0


def parse_sections(raw_text: str) -> List[Section]:
    """Split raw text into labelled sections.

    Falls back to a single section named 'full_text' if no headings found.
    """
    if not raw_text.strip():
        return []

    splits: List[tuple[int, str]] = []
    for pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(raw_text):
            splits.append((m.start(), m.group(0).strip()))

    if not splits:
        return [Section(title="full_text", text=raw_text.strip(), index=0)]

    splits.sort(key=lambda x: x[0])

    sections: List[Section] = []
    for idx, (start, title) in enumerate(splits):
        end = splits[idx + 1][0] if idx + 1 < len(splits) else len(raw_text)
        text = raw_text[start:end].strip()
        sections.append(Section(title=title, text=text, index=idx))

    return sections
