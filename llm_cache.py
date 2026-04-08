"""Thread-safe LLM response cache with TTL and disk persistence."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    response: str
    created_at: float = field(default_factory=time.time)


class LLMCache:
    def __init__(
        self,
        cache_file: Path = config.LLM_CACHE_PATH,
        ttl_seconds: int = config.LLM_CACHE_TTL_SECONDS,
    ) -> None:
        self._cache_file = cache_file
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(prompt: str, model: str) -> str:
        raw = model + "|||" + prompt
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _is_valid(self, entry: CacheEntry) -> bool:
        return time.time() - entry.created_at < self.ttl_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, prompt: str, model: str) -> Optional[str]:
        key = self._hash(prompt, model)
        with self._lock:
            entry = self._cache.get(key)
            if entry and self._is_valid(entry):
                logger.info(
                    '{"event": "LLM_CACHE_HIT", "prompt_hash": "%s", "model": "%s"}',
                    key, model,
                )
                return entry.response
        logger.info(
            '{"event": "LLM_CACHE_MISS", "prompt_hash": "%s", "model": "%s"}',
            key, model,
        )
        return None

    def set(self, prompt: str, model: str, response: str) -> None:
        key = self._hash(prompt, model)
        with self._lock:
            self._cache[key] = CacheEntry(response=response)

    def save(self) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {k: asdict(v) for k, v in self._cache.items()}
        with open(self._cache_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not self._cache_file.exists():
            return
        try:
            with open(self._cache_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            with self._lock:
                self._cache = {
                    k: CacheEntry(**v)
                    for k, v in data.items()
                }
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("LLM cache load failed, starting fresh: %s", exc)
            with self._lock:
                self._cache = {}

    def clear_expired(self) -> int:
        """Remove expired entries; returns count removed."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._cache.items()
                       if now - v.created_at >= self.ttl_seconds]
            for k in expired:
                del self._cache[k]
        return len(expired)


# Module-level singleton; lazy-initialised on first use.
_cache_instance: Optional[LLMCache] = None
_instance_lock = threading.Lock()


def _get_cache() -> LLMCache:
    global _cache_instance
    if _cache_instance is None:
        with _instance_lock:
            if _cache_instance is None:
                _cache_instance = LLMCache()
    return _cache_instance


def cached_llm_call(
    prompt: str,
    model: str,
    call_fn: Callable[[str], str],
) -> str:
    """Public API: return cached response or call call_fn and cache the result."""
    cache = _get_cache()
    cached = cache.get(prompt, model)
    if cached is not None:
        return cached
    response = call_fn(prompt)
    cache.set(prompt, model, response)
    cache.save()
    return response
