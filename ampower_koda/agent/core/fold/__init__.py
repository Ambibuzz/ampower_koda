"""§13 — the session fold: proactive, so the compaction cliff stays unreachable."""

from __future__ import annotations

from .document import (
    DROPPABLE,
    HEADER,
    SECTIONS,
    SessionState,
    blank,
    parse,
    template,
)
from .run import FOLD_RULES, Fold, fold_turn

__all__ = [
    "DROPPABLE",
    "FOLD_RULES",
    "HEADER",
    "SECTIONS",
    "Fold",
    "SessionState",
    "blank",
    "fold_turn",
    "parse",
    "template",
]
