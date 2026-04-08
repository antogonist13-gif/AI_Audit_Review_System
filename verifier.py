"""Verifier — deterministic, pattern-based checks (no LLM)."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Pattern

import config
from models import RetrievalResult, VerificationResult, Violation

# ---------------------------------------------------------------------------
# Evidence patterns
# ---------------------------------------------------------------------------

EVIDENCE_PATTERNS: List[Pattern] = [
    re.compile(r"\d{2}[./]\d{2}[./]\d{4}"),                      # date dd.mm.yyyy
    re.compile(r"\d+[\s\xa0]?(?:руб|рублей|тыс\.?\s*руб)", re.IGNORECASE),  # monetary amount
    re.compile(r"№\s*\d+"),                                        # document number
    re.compile(r"[А-ЯЁ][а-яё]{1,}(?:\s+[А-ЯЁ]\.){1,2}", re.UNICODE),  # initials (Иванов И.И.)
    re.compile(r"\bп\.\s*\d+", re.IGNORECASE),                    # paragraph reference
    re.compile(r"\bст(?:атья|\.)\s*\d+", re.IGNORECASE),          # article reference
]

# Actionability patterns — text should name who must do what
ACTIONABILITY_PATTERNS: List[Pattern] = [
    re.compile(r"\bобязан\b", re.IGNORECASE),
    re.compile(r"\bнеобходимо\b", re.IGNORECASE),
    re.compile(r"\bдолжен\b", re.IGNORECASE),
    re.compile(r"\bустранить\b", re.IGNORECASE),
    re.compile(r"\bпринять меры\b", re.IGNORECASE),
    re.compile(r"\bобеспечить\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pattern_count(text: str, patterns: List[Pattern]) -> int:
    """Count how many distinct patterns match in text."""
    return sum(1 for p in patterns if p.search(text))


def _fuzzy_ratio(a: str, b: str) -> float:
    """Return SequenceMatcher ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------


def verify_evidence_sufficiency(violation: Violation) -> VerificationResult:
    """Check whether the violation description contains enough concrete evidence."""
    text = violation.raw_text + " " + violation.description
    count = _pattern_count(text, EVIDENCE_PATTERNS)

    if count >= config.EVIDENCE_PATTERN_SUFFICIENT:
        status, detail = "confirmed", "sufficient"
    elif count == config.EVIDENCE_PATTERN_UNCLEAR:
        status, detail = "unclear", "partial"
    else:
        status, detail = "unconfirmed", "insufficient"

    return VerificationResult(
        axis="evidence",
        status=status,
        detail=detail,
        pattern_count=count,
        matched_patterns=[p.pattern for p in EVIDENCE_PATTERNS if p.search(text)],
    )


def verify_law_applicability(
    violation: Violation,
    norm_results: List[RetrievalResult],
) -> VerificationResult:
    """Check whether the cited law reference appears in retrieved norms.

    Uses both direct substring check and fuzzy ratio on short windows to
    handle short law refs (e.g. 'ФЗ-294') inside long chunk texts.
    """
    law = violation.law_ref.strip()
    if not law or not norm_results:
        return VerificationResult(
            axis="legal",
            status="unconfirmed",
            detail="no_law_ref" if not law else "no_norm_results",
        )

    law_lower = law.lower()

    for r in norm_results:
        text_lower = r.text.lower()
        # Fast substring check first
        if law_lower in text_lower:
            return VerificationResult(
                axis="legal",
                status="confirmed",
                detail="substring_match",
            )
        # Fuzzy sliding window: compare law against same-length windows in text
        win = len(law_lower)
        best_ratio = max(
            (_fuzzy_ratio(law_lower, text_lower[i:i + win])
             for i in range(0, max(1, len(text_lower) - win + 1), max(1, win // 2))),
            default=0.0,
        )
        if best_ratio >= config.LAW_GROUNDING_FUZZY_MIN:
            return VerificationResult(
                axis="legal",
                status="confirmed",
                detail=f"fuzzy_match={best_ratio:.2f}",
            )

    return VerificationResult(
        axis="legal",
        status="unconfirmed",
        detail="no_match_found",
    )


def verify_actionability(violation: Violation) -> VerificationResult:
    """Check whether the violation description implies a clear actionable remedy."""
    text = violation.raw_text + " " + violation.description
    count = _pattern_count(text, ACTIONABILITY_PATTERNS)
    status = "confirmed" if count >= 1 else "unconfirmed"
    return VerificationResult(
        axis="actionability",
        status=status,
        detail=f"action_patterns={count}",
        pattern_count=count,
    )


def verify_checklist_item(
    violation: Violation,
    ref_results: dict,
) -> List[VerificationResult]:
    """Run all verifier checks and return combined list."""
    norm_results = ref_results.get("norms", [])
    return [
        verify_evidence_sufficiency(violation),
        verify_law_applicability(violation, norm_results),
        verify_actionability(violation),
    ]
