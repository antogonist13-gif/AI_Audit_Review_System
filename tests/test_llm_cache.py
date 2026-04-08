"""Tests for llm_cache.py — thread safety, TTL, disk persistence."""
from __future__ import annotations

import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pathlib import Path
from llm_cache import CacheEntry, LLMCache, cached_llm_call


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache(tmp_path) -> LLMCache:
    cache_file = tmp_path / "test_cache.json"
    return LLMCache(cache_file=cache_file, ttl_seconds=60)


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_instantiation(self):
        entry = CacheEntry(response="hello")
        assert entry.response == "hello"
        assert entry.created_at > 0

    def test_created_at_is_recent(self):
        before = time.time()
        entry = CacheEntry(response="x")
        after = time.time()
        assert before <= entry.created_at <= after


# ---------------------------------------------------------------------------
# LLMCache basics
# ---------------------------------------------------------------------------


class TestLLMCacheBasics:
    def test_miss_returns_none(self, tmp_cache):
        assert tmp_cache.get("unknown prompt", "model-a") is None

    def test_set_and_get(self, tmp_cache):
        tmp_cache.set("prompt-1", "model-a", "response-1")
        assert tmp_cache.get("prompt-1", "model-a") == "response-1"

    def test_different_models_independent(self, tmp_cache):
        tmp_cache.set("same prompt", "model-a", "resp-a")
        tmp_cache.set("same prompt", "model-b", "resp-b")
        assert tmp_cache.get("same prompt", "model-a") == "resp-a"
        assert tmp_cache.get("same prompt", "model-b") == "resp-b"

    def test_hash_deterministic(self):
        h1 = LLMCache._hash("prompt", "model")
        h2 = LLMCache._hash("prompt", "model")
        assert h1 == h2

    def test_hash_differs_for_different_inputs(self):
        assert LLMCache._hash("p1", "m") != LLMCache._hash("p2", "m")
        assert LLMCache._hash("p", "m1") != LLMCache._hash("p", "m2")


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


class TestTTL:
    def test_entry_valid_within_ttl(self, tmp_path):
        cache = LLMCache(cache_file=tmp_path / "c.json", ttl_seconds=10)
        cache.set("p", "m", "r")
        assert cache.get("p", "m") == "r"

    def test_entry_expired_after_ttl(self, tmp_path):
        cache = LLMCache(cache_file=tmp_path / "c.json", ttl_seconds=1)
        cache.set("p", "m", "r")
        # Manually backdate the entry
        key = LLMCache._hash("p", "m")
        cache._cache[key] = CacheEntry(response="r", created_at=time.time() - 2)
        assert cache.get("p", "m") is None

    def test_clear_expired_removes_old_entries(self, tmp_path):
        cache = LLMCache(cache_file=tmp_path / "c.json", ttl_seconds=1)
        key = LLMCache._hash("p", "m")
        cache._cache[key] = CacheEntry(response="r", created_at=time.time() - 10)
        removed = cache.clear_expired()
        assert removed == 1
        assert len(cache._cache) == 0


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


class TestDiskPersistence:
    def test_save_and_reload(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache1 = LLMCache(cache_file=cache_file, ttl_seconds=60)
        cache1.set("prompt-x", "model-y", "saved-response")
        cache1.save()

        cache2 = LLMCache(cache_file=cache_file, ttl_seconds=60)
        assert cache2.get("prompt-x", "model-y") == "saved-response"

    def test_load_with_missing_file(self, tmp_path):
        cache = LLMCache(cache_file=tmp_path / "nonexistent.json", ttl_seconds=60)
        assert cache.get("any", "model") is None

    def test_load_with_corrupt_file(self, tmp_path):
        cache_file = tmp_path / "corrupt.json"
        cache_file.write_text("NOT VALID JSON {{{", encoding="utf-8")
        cache = LLMCache(cache_file=cache_file, ttl_seconds=60)
        assert cache.get("any", "model") is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_set_same_key(self, tmp_cache):
        errors = []

        def worker():
            try:
                tmp_cache.set("concurrent-prompt", "model", "response")
                _ = tmp_cache.get("concurrent-prompt", "model")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_cached_llm_call_invokes_fn_once(self, tmp_path):
        """10 threads calling cached_llm_call with same prompt → call_fn called exactly once."""
        cache = LLMCache(cache_file=tmp_path / "t.json", ttl_seconds=60)
        call_count = [0]
        lock = threading.Lock()

        def slow_fn(prompt: str) -> str:
            time.sleep(0.02)
            with lock:
                call_count[0] += 1
            return "result"

        results = []
        errors = []

        def worker():
            try:
                r = cache.get("shared-prompt", "model")
                if r is not None:
                    results.append(r)
                    return
                response = slow_fn("shared-prompt")
                cache.set("shared-prompt", "model", response)
                results.append(response)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert all(r == "result" for r in results)

    def test_cached_llm_call_helper_fn_called_once(self, tmp_path):
        """cached_llm_call public API: call_fn called exactly once for same prompt."""
        import llm_cache as lc

        cache = LLMCache(cache_file=tmp_path / "pub.json", ttl_seconds=60)
        lc._cache_instance = cache

        call_count = [0]

        def call_fn(prompt: str) -> str:
            call_count[0] += 1
            return "api-result"

        results = []
        for _ in range(5):
            r = lc.cached_llm_call("same-prompt", "model", call_fn)
            results.append(r)

        assert call_count[0] == 1
        assert all(r == "api-result" for r in results)
        lc._cache_instance = None  # cleanup
