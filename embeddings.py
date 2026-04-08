"""Embedding model wrapper — sentence-transformers with lazy loading."""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model %s", _MODEL_NAME)
            _model = SentenceTransformer(_MODEL_NAME)
        except ImportError:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            )
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return embedding vectors for a list of texts."""
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(text: str) -> List[float]:
    """Return embedding vector for a single query string."""
    return embed_texts([text])[0]
