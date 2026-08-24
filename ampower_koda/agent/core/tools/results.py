"""Every tool result is a value. Nothing here ever throws."""

from __future__ import annotations

from ..contracts.agent import ToolOutcome

PARSE_ERROR_KEY = "__parse_error"


def ok(text: str) -> ToolOutcome:
    return ToolOutcome(text=text)


def error(detail: str) -> ToolOutcome:
    """``[error: …]`` — a value the model can read and act on."""
    return ToolOutcome(text=f"[error: {detail}]", ok=False)


def parse_error(raw: str) -> ToolOutcome:
    """The provider sent tool-call JSON that does not parse."""
    return ToolOutcome(text=f"[error: malformed tool call — {PARSE_ERROR_KEY}: {raw}]", ok=False)


def cap_rows(rows: list[str], limit: int, *, unit: str = "rows") -> ToolOutcome:
    """Keep ``limit`` rows and say how many were dropped."""
    if len(rows) <= limit:
        return ToolOutcome(text="\n".join(rows))
    dropped = len(rows) - limit
    kept = [*rows[:limit], f"… +{dropped} more {unit} (truncated)"]
    return ToolOutcome(text="\n".join(kept), truncated=True, dropped=dropped)


def cap_chars(text: str, limit: int) -> ToolOutcome:
    """Trim to ``limit`` characters at a line boundary where one is close."""
    if len(text) <= limit:
        return ToolOutcome(text=text)

    head = text[:limit]
    boundary = head.rfind("\n")
    if boundary > limit * 0.75:
        head = head[:boundary]
    dropped = len(text) - len(head)
    return ToolOutcome(
        text=f"{head}\n… +{dropped} characters (truncated)",
        truncated=True,
        dropped=dropped,
    )
