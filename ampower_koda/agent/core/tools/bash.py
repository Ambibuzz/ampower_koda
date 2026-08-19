"""Is this command read-only? — and the reason each answer is the way it is."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from ..constants import (
    FD_WRITE_FLAGS,
    FIND_WRITE_FLAGS,
    GIT_READ_ONLY,
    READ_ONLY_COMMANDS,
    SHELL_WRITE_MARKERS,
)


@dataclass(frozen=True, slots=True)
class Classification:
    """What the command is, and what should happen to it."""

    read_only: bool
    reason: str = ""
    head: str = ""

    @property
    def needs_permission(self) -> bool:
        return not self.read_only


def classify(command: str) -> Classification:
    """Whether ``command`` only reads. Unknown means *not* read-only."""
    text = command.strip()
    if not text:
        return Classification(read_only=False, reason="empty command")

    marker = next((mark for mark in SHELL_WRITE_MARKERS if mark in text), "")
    if marker:
        return Classification(
            read_only=False,
            reason=f"contains {marker!r} — the head no longer predicts what runs",
            head=_head(text),
        )

    words = _words(text)
    if not words:
        return Classification(read_only=False, reason="unparseable command")

    head = words[0]
    if head == "git":
        subcommand = words[1] if len(words) > 1 else ""
        if subcommand in GIT_READ_ONLY:
            return Classification(read_only=True, head=head)
        return Classification(
            read_only=False,
            reason=f"git {subcommand or '<none>'} is not one of the read-only subcommands",
            head=head,
        )

    if head in ("find", "fd"):
        candidates = FIND_WRITE_FLAGS if head == "find" else FD_WRITE_FLAGS
        flag = next((flag for flag in candidates if flag in words), "")
        if flag:
            return Classification(
                read_only=False, reason=f"{head} {flag} executes rather than searches", head=head
            )
        return Classification(read_only=True, head=head)

    if head in READ_ONLY_COMMANDS:
        return Classification(read_only=True, head=head)

    return Classification(read_only=False, reason=f"{head} is not known to be read-only", head=head)


def scope_key(tool: str, scope: str) -> str:
    """``<tool>:<scope>`` — how an allow-always grant is remembered."""
    return f"{tool}:{scope}"


def _words(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _head(command: str) -> str:
    words = _words(command)
    return words[0] if words else ""
