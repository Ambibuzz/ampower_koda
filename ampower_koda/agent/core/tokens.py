"""Token estimation."""

from __future__ import annotations

import math

from .constants import CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    """Return the estimated token count of ``text``."""
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def token_budget_chars(max_tokens: int) -> int:
    """Return the character budget corresponding to ``max_tokens``."""
    if max_tokens <= 0:
        return 0
    return int(max_tokens * CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, max_tokens: int, *, marker: str = "\n…") -> str:
    """Return ``text`` trimmed to fit ``max_tokens``, at a line boundary if one is near."""
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    limit = max(0, token_budget_chars(max_tokens) - len(marker))
    if limit == 0:
        return ""

    head = text[:limit]
    newline = head.rfind("\n")
    if newline >= limit * 0.75:
        head = head[:newline]

    return head.rstrip() + marker
