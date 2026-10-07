"""Dereferencing a handle: re-read the refs, recompute the shas, tell the truth."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.ledger import BlobRef, Ledger
from ..contracts.source import split_lines
from ..errors import CoreError
from ..workspace.ports import Workspace
from .blobs import blob_sha
from .write import mark_stale

CHANGED = "THE FILE HAS CHANGED since you read it; this is the current text"
GONE = "THE FILE NO LONGER EXISTS"


@dataclass(frozen=True, slots=True)
class Recalled:
    """One dereferenced entry, and whether it is still what it said it was."""

    entry_id: str
    text: str
    changed: bool = False
    missing: tuple[str, ...] = ()
    """Paths that could not be read at all. A subset of ``changed`` — a file
    that is gone has certainly changed — kept separate because "deleted" and
    "edited" lead a reader somewhere different."""

    unverified: tuple[str, ...] = ()
    """Paths whose ref carried no blob sha, so "unchanged" could not be checked.

    Distinct from ``changed=False``, and the distinction is the whole reason
    this field exists. A ref with no sha compares equal to everything, so
    treating it as unchanged would hand a model rewritten bytes under the
    framing that they are the bytes it read — the precise failure the shouting
    header exists to prevent, arrived at by the one path that skips the check.
    """

    @property
    def is_empty(self) -> bool:
        return not self.text


def rehydrate(ledger: Ledger, entry_id: str, workspace: Workspace) -> tuple[Ledger, Recalled]:
    """Resolve ``entry_id`` against the current tree."""
    entry = ledger.get(entry_id)
    if entry is None:
        return ledger, Recalled(entry_id=entry_id, text="")

    if not entry.refs:
        return ledger, Recalled(entry_id=entry.id, text=entry.text)

    sections: list[str] = []
    changed = False
    missing: list[str] = []
    unverified: list[str] = []

    for ref in entry.refs:
        content = _read(workspace, ref)
        if content is None:
            changed = True
            missing.append(ref.path)
            sections.append(f"[{ref.location} — {GONE}]")
            continue
        text, sha = content
        if not ref.sha:
            unverified.append(ref.path)
        moved = bool(ref.sha) and sha != ref.sha
        changed = changed or moved
        sections.append(_section(ref, text, moved=moved, unverified=not ref.sha))

    if changed:
        ledger = mark_stale(ledger, entry.id)

    header = f"[{entry.id} — {CHANGED}]\n" if changed else ""
    return ledger, Recalled(
        entry_id=entry.id,
        text=header + "\n".join(sections),
        changed=changed,
        missing=tuple(missing),
        unverified=tuple(unverified),
    )


def ref_for(workspace: Workspace, path: str, start: int, end: int) -> BlobRef:
    """Build a ref by reading the file — the only way one is ever built."""
    try:
        return BlobRef(path=path, start=start, end=end, sha=blob_sha(workspace.read_bytes(path)))
    except (CoreError, OSError):
        return BlobRef(path=path, start=start, end=end)


def _read(workspace: Workspace, ref: BlobRef) -> tuple[str, str] | None:
    """``(text, sha)`` for one ref, or ``None`` when the file cannot be read."""
    try:
        raw = workspace.read_bytes(ref.path)
    except (CoreError, OSError):
        return None
    return raw.decode("utf-8", errors="replace"), blob_sha(raw)


def _section(ref: BlobRef, content: str, *, moved: bool, unverified: bool = False) -> str:
    """One ref's current text, sliced to its span, headed by its location."""
    lines = split_lines(content)
    start = max(1, ref.start or 1)
    end = min(len(lines), ref.end or start)
    body = "\n".join(lines[start - 1 : end]) if start <= len(lines) else ""
    marker = " (moved)" if moved else (" (unverified — no sha was recorded)" if unverified else "")
    return f"{ref.location}{marker}\n{body}" if body else f"{ref.location}{marker}"
