"""§15 — the tool catalogue and results as values."""

from __future__ import annotations

from ..contracts.agent import ToolOutcome
from .catalogue import CATALOGUE, TOOL_NAMES, ToolSpec, by_name
from .results import cap_chars, cap_rows, error, ok, parse_error

__all__ = [
    "CATALOGUE",
    "TOOL_NAMES",
    "ToolOutcome",
    "ToolSpec",
    "by_name",
    "cap_chars",
    "cap_rows",
    "error",
    "ok",
    "parse_error",
]
