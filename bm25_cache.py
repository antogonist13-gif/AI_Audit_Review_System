"""BM25 index — per-collection, built lazily from stored chunks."""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_bm25_indexes: Dict[str, "_BM25Index"] = {}


class _BM25Index:
    """Minimal BM25 implementation (no external dependency)."""

    def __init__(self, corpus: List[str]) -> None:
        import math

        self._corpus = corpus
        self._tokenized = [_tokenize(doc) for doc in corpus]
        self._avg_dl = sum(len(d) for d in self._tokenized) / max(len(self._tokenized), 1)
        self._idf: Dict[str, float] = {}
        N = len(self._tokenized)
        df: Dict[str, int] = {}
        for tokens in self._tokenized:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        for term, freq in df.items():
            self._idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        self.k1 = 1.5
        self.b = 0.75

    def score(self, query: str, doc_idx: int) -> float:
        tokens = _tokenize(query)
        doc_tokens = self._tokenized[doc_idx]
        dl = len(doc_tokens)
        score = 0.0
        tf_map: Dict[str, int] = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        for token in tokens:
            if token not in self._idf:
                continue
            tf = tf_map.get(token, 0)
            idf = self._idf[token]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += idf * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self._corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def build_bm25_index(collection_name: str, texts: List[str]) -> None:
    """Build (or rebuild) the BM25 index for a named collection."""
    logger.info("Building BM25 index for %s — %d documents", collection_name, len(texts))
    _bm25_indexes[collection_name] = _BM25Index(texts)


def bm25_search(
    collection_name: str,
    query: str,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """Search the BM25 index; returns (doc_index, score) pairs."""
    idx = _bm25_indexes.get(collection_name)
    if idx is None:
        logger.warning("No BM25 index for %s — returning empty results", collection_name)
        return []
    return idx.search(query, top_k=top_k)


def get_index(collection_name: str) -> Optional[_BM25Index]:
    return _bm25_indexes.get(collection_name)
