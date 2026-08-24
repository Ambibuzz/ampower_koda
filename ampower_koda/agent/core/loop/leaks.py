"""A model writing a tool call as *prose*, and the three rules for handling it."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

LEAK_MARKER = re.compile(r"\[tool_call[=:\s]|<function=|<parameter=")

_FUNCTION = re.compile(r"<function=([A-Za-z_][\w]*)", re.IGNORECASE)
_BRACKET = re.compile(r"\[tool_call[=:\s]+([A-Za-z_][\w]*)", re.IGNORECASE)
_JSON_ARGS = re.compile(r"(\{.*\})", re.DOTALL)

CORRECTION = (
    "Your last message contained what looked like a tool call written as text. Tool "
    "calls must be emitted through the tool-call interface, not written into the "
    "message. Call the tool properly, or answer without it."
)


@dataclass(frozen=True, slots=True)
class Leak:
    """A recovered call, or the fact that one could not be recovered."""

    detected: bool = False
    tool: str = ""
    arguments: dict[str, object] | None = None
    detail: str = ""

    @property
    def recovered(self) -> bool:
        return bool(self.tool)


def detect(text: str) -> bool:
    """Whether the accumulated stream contains a leak marker."""
    return bool(LEAK_MARKER.search(text))


def recover(text: str, tool_names: frozenset[str]) -> Leak:
    """Try to turn leaked prose back into a call, against the current tool set."""
    if not detect(text):
        return Leak()

    match = _FUNCTION.search(text) or _BRACKET.search(text)
    if match is None:
        return Leak(detected=True, detail="no tool name in the leaked text")

    name = match.group(1)
    if name not in tool_names:
        return Leak(
            detected=True,
            detail=f"{name!r} is not in this turn's tool set — nothing was executed",
        )

    return Leak(detected=True, tool=name, arguments=_arguments(text))


def _arguments(text: str) -> dict[str, object]:
    """A JSON object from the leaked text, or nothing."""
    match = _JSON_ARGS.search(text)
    if match is None:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
