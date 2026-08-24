"""Dry-apply the whole edit list in memory, then describe what would change."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.source import Span, split_lines
from ..errors import CoreError
from .locate import Located, locate


@dataclass(frozen=True, slots=True)
class Edit:
    """One replacement, as the model described it."""

    path: str
    old_string: str
    new_string: str
    reason: str = ""
    occurrence: int = 0
    anchor: str = ""
    """A content digest the model recorded when it read the span. Verified
    before the edit resolves, which narrows the search to that span and turns a
    stale coordinate into a refusal instead of a wrong replacement."""


@dataclass(frozen=True, slots=True)
class ChangedRegion:
    """A span of the *new* document that this edit produced."""

    path: str
    span: Span
    lines_added: int = 0
    lines_removed: int = 0


@dataclass(frozen=True, slots=True)
class Patch:
    """A described edit to one file. Not applied — described."""

    path: str
    before: str
    after: str
    regions: tuple[ChangedRegion, ...] = ()
    normalized_eol: bool = False

    @property
    def changed(self) -> bool:
        return self.before != self.after

    @property
    def preview_line(self) -> int:
        """The line a permission dialog should show."""
        return self.regions[0].span.start if self.regions else 0


@dataclass(frozen=True, slots=True)
class DryRun:
    """The whole list, resolved in memory, or the first reason it could not be."""

    patches: tuple[Patch, ...] = ()
    failures: tuple[str, ...] = ()
    normalized_eol: bool = False

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(patch.path for patch in self.patches))


@dataclass(frozen=True, slots=True)
class AnchorCheck:
    """Whether an anchor still names what it named."""

    path: str
    digest: str
    span: Span | None = None
    verified: bool = True
    detail: str = ""


def dry_apply(
    edits: list[Edit],
    contents: dict[str, str],
    *,
    anchors: dict[str, AnchorCheck] | None = None,
) -> DryRun:
    """Resolve every edit against an in-memory copy of the tree."""
    working = dict(contents)
    patches: dict[str, Patch] = {}
    failures: list[str] = []
    normalized = False

    for index, edit in enumerate(edits):
        if edit.path not in working:
            failures.append(
                f"edit {index + 1}: {edit.path} does not exist — use write to create it"
            )
            continue

        checked = (anchors or {}).get(edit.anchor)
        if edit.anchor and (checked is None or not checked.verified):
            detail = checked.detail if checked else "anchor not found"
            failures.append(f"edit {index + 1}: {edit.path} — {detail}")
            continue

        before = working[edit.path]
        window = _window(before, checked)
        if window is None:
            failures.append(
                f"edit {index + 1}: {edit.path} — the anchor names lines this file no longer "
                "has; re-read the span before editing it"
            )
            continue

        found = locate(window.text, edit.old_string, occurrence=edit.occurrence)
        if not found.ok:
            failures.append(f"edit {index + 1}: {edit.path} — {found.error}")
            continue

        after = _replace(before, window.offset, found, edit.new_string)
        normalized = normalized or found.normalized_eol
        working[edit.path] = after
        earlier = patches.get(edit.path)
        patches[edit.path] = _patch(
            edit.path,
            earlier.before if earlier else before,
            after,
            normalized_eol=found.normalized_eol or bool(earlier and earlier.normalized_eol),
        )

    return DryRun(
        patches=tuple(patches.values()),
        failures=tuple(failures),
        normalized_eol=normalized,
    )


@dataclass(frozen=True, slots=True)
class _Window:
    """The region an edit is allowed to match inside, and where it starts."""

    text: str
    offset: int = 0


def _window(content: str, anchor: AnchorCheck | None) -> _Window | None:
    """The anchored span, the whole file, or ``None`` when the anchor is stale."""
    if anchor is None or anchor.span is None:
        return _Window(text=content)

    lines = split_lines(content)
    start = max(1, anchor.span.start)
    if start > len(lines):
        return None

    end = min(len(lines), anchor.span.end or start)
    offset = sum(len(line) + 1 for line in lines[: start - 1])
    return _Window(text="\n".join(lines[start - 1 : end]), offset=offset)


def _replace(content: str, offset: int, found: Located, replacement: str) -> str:
    """Splice the replacement in, re-spelled to the file's line endings."""
    if found.match is None:
        raise CoreError("cannot replace without a match")
    text = replacement.replace("\r\n", "\n").replace("\n", found.eol) if (
        found.normalized_eol
    ) else replacement
    start = offset + found.match.start
    end = offset + found.match.end
    return content[:start] + text + content[end:]


def _patch(path: str, before: str, after: str, *, normalized_eol: bool) -> Patch:
    return Patch(
        path=path,
        before=before,
        after=after,
        regions=changed_regions(path, before, after),
        normalized_eol=normalized_eol,
    )


def changed_regions(path: str, before: str, after: str) -> tuple[ChangedRegion, ...]:
    """The line spans of the *new* document that differ from the old."""
    if before == after:
        return ()

    old = split_lines(before)
    new = split_lines(after)

    head = 0
    while head < len(old) and head < len(new) and old[head] == new[head]:
        head += 1

    tail = 0
    while (
        tail < len(old) - head
        and tail < len(new) - head
        and old[len(old) - 1 - tail] == new[len(new) - 1 - tail]
    ):
        tail += 1

    last = _numbered_lines(new)
    start = min(head + 1, max(1, last))
    end = min(max(start, len(new) - tail), max(1, last))
    return (
        ChangedRegion(
            path=path,
            span=Span(start=start, end=end),
            lines_added=max(0, len(new) - tail - head),
            lines_removed=max(0, len(old) - tail - head),
        ),
    )


def _numbered_lines(lines: list[str]) -> int:
    """How many lines a reader would number."""
    return len(lines) - 1 if lines and lines[-1] == "" else len(lines)
