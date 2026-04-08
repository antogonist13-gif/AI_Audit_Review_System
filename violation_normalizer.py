"""Violation normalizer — lemmatize description, extract keywords."""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import replace
from typing import List, Optional

import config
from models import Violation

logger = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Lazy-loaded pymorphy2 analyser
_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        try:
            # inspect.getargspec удалён в Python 3.11+; pymorphy2 0.9.1 его использует.
            # getfullargspec возвращает 7 полей, getargspec возвращал 4 —
            # возвращаем только первые 4, чтобы распаковка в pymorphy2 не сломалась.
            import inspect
            if not hasattr(inspect, "getargspec"):
                inspect.getargspec = lambda f: inspect.getfullargspec(f)[:4]

            import pymorphy2
            _morph = pymorphy2.MorphAnalyzer()
        except ImportError:
            logger.warning(
                "pymorphy2 not installed — falling back to whitespace tokenizer"
            )
            _morph = _FallbackAnalyzer()
    return _morph


class _FallbackAnalyzer:
    """No-op analyser when pymorphy2 is unavailable."""

    def parse(self, word: str):
        return [_FallbackParsed(word)]


class _FallbackParsed:
    def __init__(self, word: str) -> None:
        self.normal_form = word.lower()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Lower-case → strip punctuation → lemmatize → join."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()
    morph = _get_morph()
    lemmas: List[str] = []
    for token in tokens:
        if not token.strip():
            continue
        parsed = morph.parse(token)
        if parsed:
            lemmas.append(parsed[0].normal_form)
    return " ".join(lemmas)


def _extract_keywords(lemma_text: str, top_n: int = 10) -> List[str]:
    """Extract top_n keywords by frequency, excluding Russian stopwords."""
    tokens = lemma_text.split()
    filtered = [t for t in tokens if t not in config.RUSSIAN_STOPWORDS and len(t) > 2]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_violation(violation: Violation) -> Violation:
    """Return a new Violation with normalized_text and keywords populated.

    Does NOT mutate the input object.
    """
    assert violation.description, (
        f"Empty description for violation {violation.id}"
    )

    normalized = _normalize_text(violation.description)

    assert normalized, (
        f"Normalization produced empty string for violation {violation.id}"
    )

    keywords = _extract_keywords(normalized, top_n=10)
    token_count = len(normalized.split())

    possibly_weak = token_count < config.VIOLATION_MIN_TOKEN_COUNT

    if possibly_weak:
        logger.warning(
            "Violation %s has only %d tokens — flagged as possibly_not_a_violation",
            violation.id, token_count,
        )

    logger.info(
        '{"event": "NORMALIZATION_DONE", "violation_id": "%s", "token_count": %d, "possibly_weak": %s}',
        violation.id, token_count, str(possibly_weak).lower(),
    )

    return replace(
        violation,
        normalized_text=normalized,
        keywords=keywords,
        possibly_not_a_violation=violation.possibly_not_a_violation or possibly_weak,
    )
