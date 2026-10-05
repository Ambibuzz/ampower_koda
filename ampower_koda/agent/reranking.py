"""OpenRouter's dedicated rerank API. No chat prompt, tools, or conversation."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from http.client import HTTPException
from math import isfinite
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .core.contracts.rerank import RerankScores

ENDPOINT = "https://openrouter.ai/api/v1/rerank"
_MAX_RESPONSE_BYTES = 1_000_000
_CACHE_SIZE = 128
_FAILURE_COOLDOWN = 30.0
_LOG = logging.getLogger(__name__)


@dataclass
class OpenRouterReranker:
    api_key: str = field(repr=False)
    model: str = "cohere/rerank-v3.5"
    timeout_seconds: float = 5.0
    _cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _lock: object = field(default_factory=threading.Lock, init=False, repr=False)
    _retry_after: float = field(default=0.0, init=False, repr=False)
    calls: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    cost: float = field(default=0.0, init=False)

    def score(self, query: str, documents: tuple[str, ...]) -> RerankScores:
        if not self.api_key.strip():
            return RerankScores(error="OpenRouter API key is not configured")
        if not documents:
            return RerankScores()
        payload = json.dumps({"model": self.model, "query": query,
                              "documents": documents, "top_n": len(documents)}).encode("utf-8")
        key = hashlib.sha256(payload).digest()
        # Coalesce identical simultaneous requests and bound cache memory.
        with self._lock:
            if key in self._cache:
                self.cache_hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            if time.monotonic() < self._retry_after:
                return RerankScores(error="OpenRouter reranker is temporarily unavailable")
            started = time.monotonic()
            self.calls += 1
            data = None
            try:
                data = self._request(payload)
                result = _scores(data, len(documents))
            except HTTPError as error:
                result = RerankScores(error=f"OpenRouter HTTP {error.code}")
                error.close()
            except (OSError, HTTPException, ValueError, TypeError, KeyError):
                result = RerankScores(error="OpenRouter timeout, connection, or response error")
            usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            cost = usage.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and isfinite(cost) and cost >= 0:
                self.cost += cost
            else:
                cost = None
            _LOG.info("Rerank model=%s candidates=%s seconds=%.3f search_units=%s cost=%s",
                      self.model, len(documents), time.monotonic() - started,
                      usage.get("search_units"), cost)
            if result.error:
                self._retry_after = time.monotonic() + _FAILURE_COOLDOWN
                _LOG.warning("Rerank fallback: %s", result.error)
                return result
            self._cache[key] = result
            while len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
            return result

    def _request(self, payload: bytes) -> dict:
        request = Request(ENDPOINT, data=payload, method="POST", headers={
            "Authorization": "Bearer " + self.api_key.strip(),
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "Koda",
        })
        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ValueError("rerank response exceeded its size limit")
        return json.loads(body)


def _scores(data: dict, count: int) -> RerankScores:
    """Require a complete permutation; missing/duplicate indices are failures."""
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("missing results")
    values: dict[int, float] = {}
    for result in data["results"]:
        if not isinstance(result, dict):
            raise ValueError("invalid result")
        index, score = result.get("index"), result.get("relevance_score")
        if type(index) is not int or not 0 <= index < count or index in values:
            raise ValueError("invalid document index")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not isfinite(score) or not 0 <= score <= 1):
            raise ValueError("invalid relevance score")
        values[index] = float(score)
    if len(values) != count:
        raise ValueError("incomplete results")
    return RerankScores(tuple(values[i] for i in range(count)))
