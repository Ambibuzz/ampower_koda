"""§9 — the escalation ladder: spend a model call on *vocabulary*, never on
exploration.
"""

from __future__ import annotations

from .ladder import Attempted, band, decide
from .memo import Formulated, FormulationCache, formulate
from .merge import added, annotate, round_robin
from .prompts import (
    FANOUT_ANGLES,
    FANOUT_SYSTEM,
    TRANSLATE_MAX_TOKENS,
    TRANSLATE_SYSTEM,
    parse_fanout,
    parse_translation,
)
from .run import escalated_search

__all__ = [
    "FANOUT_ANGLES",
    "FANOUT_SYSTEM",
    "TRANSLATE_MAX_TOKENS",
    "TRANSLATE_SYSTEM",
    "Attempted",
    "Formulated",
    "FormulationCache",
    "added",
    "annotate",
    "band",
    "decide",
    "escalated_search",
    "formulate",
    "parse_fanout",
    "parse_translation",
    "round_robin",
]
