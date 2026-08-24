"""Is this string a command? — and what does a failure say?"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..constants import CHECK_MAX_CHARS, CHECK_RUNNERS, FAILURE_LINE_MAX_CHARS, PROSE_HEADS
from ..tools.bash import classify

DECISIVE: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:FAIL|ERROR|error|Error)\b"),
    re.compile(r"\b(?:AssertionError|expected .* (?:to|but)\b)"),
    re.compile(r"^[^\s:]+\.[a-zA-Z]+[:(]\d+[:,)]"),
    re.compile(r"\berror TS\d+\b"),
    re.compile(r"\bnot ok \d+\b"),
)


@dataclass(frozen=True, slots=True)
class CommandCheck:
    """Whether a string may be run as a check, and why not if it may not."""

    runnable: bool
    reason: str = ""
    head: str = ""
    read_only: bool = False


def is_command(text: str) -> CommandCheck:
    """Gate a model-written check before it reaches a permission dialog."""
    candidate = text.strip()
    if not candidate:
        return CommandCheck(runnable=False, reason="empty check")
    if "\n" in candidate:
        return CommandCheck(
            runnable=False, reason="a newline is a second command nobody reviewed"
        )
    if len(candidate) > CHECK_MAX_CHARS:
        return CommandCheck(runnable=False, reason="the check is prose, not a command")

    classification = classify(candidate)
    head = classification.head or candidate.split()[0]

    if classification.read_only:
        return CommandCheck(runnable=True, head=head, read_only=True)

    if head.lower() in PROSE_HEADS:
        return CommandCheck(runnable=False, reason=f"{head!r} begins a sentence, not a command")
    if head not in CHECK_RUNNERS:
        return CommandCheck(runnable=False, reason=f"{head!r} is not a known check runner")

    return CommandCheck(runnable=True, head=head)


def failure_line(output: str) -> str:
    """The one line worth carrying forward from a failed run."""
    lines = [line for line in output.split("\n") if line.strip()]
    if not lines:
        return ""

    for pattern in DECISIVE:
        for line in lines:
            if pattern.search(line):
                return _clip(line.strip())

    return _clip(lines[-1].strip())


def _clip(line: str) -> str:
    if len(line) <= FAILURE_LINE_MAX_CHARS:
        return line
    return line[: FAILURE_LINE_MAX_CHARS - 1] + "…"


def is_wish(check: str) -> str:
    """Why this check is not a falsifier, or ``""`` if it is one."""
    candidate = check.strip()
    if len(candidate.split()) < 2:
        return f"check {candidate!r} is a wish, not a falsifier — it is not a command"
    if re.match(r"^(verify|check|make sure|ensure)\b", candidate, re.IGNORECASE):
        return (
            f"check {candidate!r} is a wish, not a falsifier — "
            "name a command, a test, or a grep that fails out loud"
        )
    return ""
