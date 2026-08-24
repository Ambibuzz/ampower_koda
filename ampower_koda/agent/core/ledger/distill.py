"""Tool result → one ledger line, at result time."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..constants import DISTILL_MAX_ANCHORS, DISTILL_SCAN_LINES
from ..contracts.ledger import BlobRef, Confidence, Ledger, LedgerKind
from ..workspace.ports import Workspace
from .recall import ref_for
from .write import record

TOOL_KINDS: Mapping[str, LedgerKind] = {
    "search": "hits",
    "explore": "hits",
    "grep": "probe",
    "glob": "probe",
    "outline": "shape",
    "symbols": "shape",
    "refs": "shape",
    "definition": "shape",
    "ast_search": "shape",
    "diff": "state",
}

_ANCHOR = re.compile(r"\b([\w./-]+\.[A-Za-z0-9]{1,6}):(\d+)(?:-(\d+))?\b")


@dataclass(frozen=True, slots=True)
class Distillate:
    """One line of ledger, and the anchors that back it."""

    kind: LedgerKind
    text: str
    refs: tuple[BlobRef, ...] = ()
    confidence: Confidence = "read"

    @property
    def is_empty(self) -> bool:
        return not self.text


def distil(
    tool: str,
    arguments: Mapping[str, object],
    result: str,
    *,
    workspace: Workspace | None = None,
) -> Distillate:
    """Summarise one tool result into the line the ledger will keep."""
    kind = TOOL_KINDS.get(tool)
    if kind is None:
        return Distillate(kind="probe", text="")

    head = "\n".join(result.split("\n")[:DISTILL_SCAN_LINES])
    refs = _anchors(head, workspace)
    summary = _summary(tool, arguments, result, refs)
    return Distillate(kind=kind, text=summary, refs=refs, confidence="read")


def distil_into(
    ledger: Ledger,
    tool: str,
    arguments: Mapping[str, object],
    result: str,
    *,
    workspace: Workspace | None = None,
) -> tuple[Ledger, str]:
    """Distil and record in one step. Returns the ledger and the entry id."""
    distillate = distil(tool, arguments, result, workspace=workspace)
    if distillate.is_empty:
        return ledger, ""
    return record(
        ledger,
        distillate.kind,
        distillate.text,
        refs=distillate.refs,
        confidence=distillate.confidence,
    )


def _summary(
    tool: str,
    arguments: Mapping[str, object],
    result: str,
    refs: Sequence[BlobRef],
) -> str:
    """The one line, per tool family."""
    query = _argument(arguments, "query", "pattern", "regex", "path", "glob", "symbol")
    quoted = f'"{query}"' if query else ""

    if tool in ("search", "explore"):
        top = f", top {refs[0].location}" if refs else ""
        return f"{tool} {quoted}: {_count(result)} hits{top}".strip()

    if tool == "glob":
        return f"glob {quoted}: {_count(result)} path(s)".strip()

    if tool == "grep":
        return f"grep {quoted}: {_grep_shape(arguments, result)}".strip()

    if tool == "diff":
        return f"diff: {_lines(result)} line(s) of working-tree state"

    return f"{tool} {query or ''}: {_lines(result)} line(s)".strip()


def _grep_shape(arguments: Mapping[str, object], result: str) -> str:
    """``0 matches`` · ``4 file(s)`` · ``12 matches in 3 file(s)``."""
    count = _count(result)
    if not count:
        return "0 matches"
    if str(arguments.get("mode", "")) == "files":
        return f"{count} file(s)"
    files = len({line.split(":", 1)[0] for line in _content_lines(result) if ":" in line})
    return f"{count} matches in {files or 1} file(s)"


def _anchors(head: str, workspace: Workspace | None) -> tuple[BlobRef, ...]:
    """Up to :data:`DISTILL_MAX_ANCHORS` distinct locations, in output order."""
    found: dict[str, BlobRef] = {}
    for match in _ANCHOR.finditer(head):
        path, start, end = match.group(1), int(match.group(2)), match.group(3)
        finish = int(end) if end else start
        if finish < start:
            continue
        ref = (
            ref_for(workspace, path, start, finish)
            if workspace is not None
            else BlobRef(path=path, start=start, end=finish)
        )
        found.setdefault(ref.location, ref)
        if len(found) >= DISTILL_MAX_ANCHORS:
            break
    return tuple(found.values())


def _argument(arguments: Mapping[str, object], *names: str) -> str:
    """The first present, non-empty argument among ``names``."""
    for name in names:
        value = arguments.get(name)
        if value not in (None, "", [], {}):
            return str(value)
    return ""


def _content_lines(result: str) -> list[str]:
    return [line for line in result.split("\n") if line.strip()]


def _lines(result: str) -> int:
    return len(_content_lines(result))


def _count(result: str) -> int:
    """How many results this output represents."""
    return _lines(result)
