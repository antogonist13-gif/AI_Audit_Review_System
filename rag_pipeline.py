"""RAG pipeline — LLM call wrapper with optional ChromaDB retrieval."""
from __future__ import annotations

import logging
from typing import Optional

import config
from llm_cache import cached_llm_call

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Wrapper around the Ollama LLM with integrated caching."""

    def __init__(
        self,
        model_name: str = config.OLLAMA_MODEL,
        base_url: str = config.OLLAMA_BASE_URL,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, prompt: str) -> str:
        """Run prompt through LLM (cached)."""
        return cached_llm_call(
            prompt=prompt,
            model=self.model_name,
            call_fn=self._call_ollama_raw,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        """Cached entry point (kept for backward compatibility)."""
        return cached_llm_call(
            prompt=prompt,
            model=self.model_name,
            call_fn=self._call_ollama_raw,
        )

    def _call_ollama_raw(self, prompt: str) -> str:
        """Direct Ollama call — no cache."""
        try:
            import ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
            )
            return response.get("response", "")
        except ImportError:
            raise ImportError("Install ollama client: pip install ollama")
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            raise


# Module-level singleton
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
