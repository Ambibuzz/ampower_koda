"""Two deduplication nets, and the one event that clears both."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

REPLAYABLE: frozenset[str] = frozenset(
    {
        "search",
        "grep",
        "glob",
        "outline",
        "symbols",
        "refs",
        "read",
        "definition",
        "explore",
        "ast_search",
    }
)

NOT_REPLAYABLE: frozenset[str] = frozenset(
    {"recall", "trace_discover", "read_doctype_schema"}
)

RESULT_FINGERPRINT_MIN_CHARS = 400

PREFETCH_CONCURRENCY = 4


def canonical(tool: str, arguments: Mapping[str, object]) -> str:
    """The call's identity: name plus canonical JSON of its arguments."""
    return (
        f"{tool}\0"
        f"{json.dumps(_prune(arguments), sort_keys=True, separators=(',', ':'), default=repr)}"
    )


def _prune(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _prune(item)
            for key, item in sorted(value.items())
            if item not in (None, "", [], {})
        }
    if isinstance(value, (list, tuple)):
        return [_prune(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Suppressed:
    """Why a call was not run, in words the model can act on."""

    reason: str
    text: str

    @property
    def is_empty(self) -> bool:
        return not self.reason


def message(tool: str, kind: str) -> str:
    """The suppression text. Names the call, the fact, and the way forward."""
    if kind == "call":
        return (
            f"[{tool} was already run this turn with these exact arguments and nothing has "
            "changed since. Read the earlier result, or call it with different arguments.]"
        )
    return (
        f"[{tool} returned bytes identical to an earlier result this turn. Nothing has "
        "changed since. Move on, or ask a different question.]"
    )


@dataclass(slots=True)
class Memo:
    """The two nets for one turn. Mutable, and scoped to a turn on purpose."""

    calls: dict[str, int] = field(default_factory=dict)
    results: dict[str, int] = field(default_factory=dict)
    suppressed: int = 0

    def check_call(self, tool: str, arguments: Mapping[str, object]) -> Suppressed | None:
        """Whether this exact call has already run this turn."""
        key = canonical(tool, arguments)
        if key in self.calls:
            self.suppressed += 1
            return Suppressed(reason="duplicate call", text=message(tool, "call"))
        return None

    def record_call(self, tool: str, arguments: Mapping[str, object], index: int) -> None:
        self.calls[canonical(tool, arguments)] = index

    def check_result(self, tool: str, text: str) -> Suppressed | None:
        """Whether these exact bytes have already been returned this turn."""
        if len(text) < RESULT_FINGERPRINT_MIN_CHARS:
            return None
        if text in self.results:
            self.suppressed += 1
            return Suppressed(reason="duplicate result", text=message(tool, "result"))
        return None

    def record_result(self, text: str, index: int) -> None:
        if len(text) >= RESULT_FINGERPRINT_MIN_CHARS:
            self.results[text] = index

    def clear(self) -> None:
        """Both nets, entirely. Called after any non-replayable call."""
        self.calls.clear()
        self.results.clear()


def leading_replayable(tools: list[str]) -> int:
    """How many calls at the head of a batch may be dispatched together."""
    count = 0
    for tool in tools:
        if tool not in REPLAYABLE:
            break
        count += 1
    return min(count, PREFETCH_CONCURRENCY)
