"""Bounding history: stub old tool results, then summarize under input pressure."""

from __future__ import annotations

from .collapse import ELIDED, READ_TOOLS, collapse, stub
from .compact import SUMMARY_HEADER, Compaction, compact_for_request
from .hotcold import Elision, hot_cold

__all__ = [
    "ELIDED",
    "READ_TOOLS",
    "SUMMARY_HEADER",
    "Compaction",
    "Elision",
    "collapse",
    "compact_for_request",
    "hot_cold",
    "stub",
]
