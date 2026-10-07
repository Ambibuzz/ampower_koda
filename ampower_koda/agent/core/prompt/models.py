"""What each model family will actually cache."""

from __future__ import annotations

from ..constants import (
    MAX_TOTAL_BREAKPOINTS,
    MIN_CACHEABLE_BY_FAMILY,
    MIN_CACHEABLE_DEFAULT,
    SESSION_ID_MAX_CHARS,
)
from ..contracts.prompt import ModelCacheLimits

_FAMILIES: tuple[str, ...] = tuple(
    sorted(MIN_CACHEABLE_BY_FAMILY, key=lambda name: (-len(name), name))
)


def cache_limits(model: str) -> ModelCacheLimits:
    """What ``model`` will cache, defaulting conservatively."""
    identifier = model.lower()
    for family in _FAMILIES:
        if family in identifier:
            return ModelCacheLimits(
                min_cacheable=MIN_CACHEABLE_BY_FAMILY[family],
                max_breakpoints=MAX_TOTAL_BREAKPOINTS,
                families=(family,),
            )
    return ModelCacheLimits(
        min_cacheable=MIN_CACHEABLE_DEFAULT,
        max_breakpoints=MAX_TOTAL_BREAKPOINTS,
    )


def routing_key(session_id: str) -> str:
    """The session id as a provider will accept it."""
    return session_id[:SESSION_ID_MAX_CHARS]
