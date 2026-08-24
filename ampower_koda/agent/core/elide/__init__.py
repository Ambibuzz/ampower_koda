"""§12 — two mechanisms at two timescales, easy to conflate."""

from __future__ import annotations

from .collapse import ELIDED, READ_TOOLS, ROADMAP_TOOLS, collapse, roadmap, stub
from .compact import (
    CLIFF_PROMPT,
    PRESSURE_OPTIONS,
    PRESSURE_QUESTION,
    SUMMARY_HEADER,
    Compaction,
    ThrashGuard,
    compact,
    trim_folded,
)
from .hotcold import Elision, hot_cold

__all__ = [
    "CLIFF_PROMPT",
    "ELIDED",
    "PRESSURE_OPTIONS",
    "PRESSURE_QUESTION",
    "READ_TOOLS",
    "ROADMAP_TOOLS",
    "SUMMARY_HEADER",
    "Compaction",
    "Elision",
    "ThrashGuard",
    "collapse",
    "compact",
    "hot_cold",
    "roadmap",
    "stub",
    "trim_folded",
]
