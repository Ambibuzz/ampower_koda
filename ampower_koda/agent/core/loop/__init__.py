"""§14 — the agent loop's decisions, as pure functions of what the turn spent."""

from __future__ import annotations

from .dedupe import (
    NOT_REPLAYABLE,
    PREFETCH_CONCURRENCY,
    REPLAYABLE,
    Memo,
    Suppressed,
    canonical,
    leading_replayable,
)
from .gates import (
    DRY_ROUNDS_LIMIT,
    Decision,
    TurnMeters,
    after_max_tokens,
    check,
    coverage_gate,
    evidence_yield,
    is_dry,
    late_tool_call,
)
from .leaks import CORRECTION, LEAK_MARKER, Leak, detect, recover
from .nudges import Nudge, coverage

__all__ = [
    "CORRECTION",
    "DRY_ROUNDS_LIMIT",
    "LEAK_MARKER",
    "NOT_REPLAYABLE",
    "PREFETCH_CONCURRENCY",
    "REPLAYABLE",
    "Decision",
    "Leak",
    "Memo",
    "Nudge",
    "Suppressed",
    "TurnMeters",
    "after_max_tokens",
    "canonical",
    "check",
    "coverage",
    "coverage_gate",
    "detect",
    "evidence_yield",
    "is_dry",
    "late_tool_call",
    "leading_replayable",
    "recover",
]
