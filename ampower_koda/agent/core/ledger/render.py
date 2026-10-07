"""The ledger as one prompt block, fitted to a budget by merging, never dropping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..constants import (
    LEDGER_MERGE_MAX_CHARS,
    LEDGER_MERGE_MAX_PART_CHARS,
    LEDGER_MERGE_MAX_REFS,
)
from ..contracts.ledger import BlobRef, LedgerEntry
from ..tokens import estimate_tokens

HEADER = "CONTEXT LEDGER"
NO_REFS = "(no refs)"

_LAST = "￿"


@dataclass(frozen=True, slots=True)
class Run:
    """One rendered line: a single entry, or several merged into one."""

    ids: tuple[str, ...]
    parts: tuple[str, ...]
    refs: tuple[BlobRef, ...]
    flags: tuple[str, ...]
    pinned: bool
    order: int
    """Position in the log. What "oldest" means, kept explicitly because a
    merged run has several ids and the smallest is not always the oldest once
    supersession has renumbered around it."""

    @property
    def merged(self) -> bool:
        return len(self.ids) > 1

    def line(self) -> str:
        label = ",".join(self.ids)
        flags = ",".join(["merged"] if self.merged else self.flags)
        refs = ", ".join(ref.location for ref in self.refs[:LEDGER_MERGE_MAX_REFS])
        extra = f" +{len(self.refs) - LEDGER_MERGE_MAX_REFS}" if (
            len(self.refs) > LEDGER_MERGE_MAX_REFS
        ) else ""
        tail = f"  ({refs}{extra})" if refs else ""
        return f"  {label} [{flags}] {'; '.join(self.parts)}{tail}"


@dataclass(frozen=True, slots=True)
class LedgerBlock:
    """The rendered block, and what the fitting loop had to do to it."""

    text: str = ""
    tokens: int = 0
    entries_shown: int = 0
    merges: int = 0
    over_budget: bool = False
    """True when even the pinned entries do not fit. Reported rather than
    resolved: dropping a pinned entry to make a number look right is the one
    failure this block must not hide."""

    @property
    def is_empty(self) -> bool:
        return not self.text


def render_ledger(entries: Sequence[LedgerEntry], *, soft_tokens: int) -> LedgerBlock:
    """Render ``entries`` as one block, merged down until it fits."""
    runs = [_run(entry, order) for order, entry in enumerate(entries)]
    if not runs:
        return LedgerBlock()

    text = _render(runs)
    merges = 0
    while estimate_tokens(text) > soft_tokens:
        merged = merge_oldest_unpinned(runs)
        if merged is None:
            break
        runs, merges = merged, merges + 1
        text = _render(runs)

    tokens = estimate_tokens(text)
    return LedgerBlock(
        text=text,
        tokens=tokens,
        entries_shown=sum(len(run.ids) for run in runs),
        merges=merges,
        over_budget=tokens > soft_tokens,
    )


def merge_oldest_unpinned(runs: Sequence[Run]) -> list[Run] | None:
    """Splice the oldest unpinned run into its same-file successor."""
    grouped = _group(runs)
    candidates = [
        (run.order, key, index)
        for key, group in grouped
        for index, run in enumerate(group)
        if not run.pinned and index + 1 < len(group)
    ]
    if not candidates:
        return None

    _, key, index = min(candidates)
    rebuilt: list[Run] = []
    for group_key, group in grouped:
        if group_key != key:
            rebuilt.extend(group)
            continue
        merged = _splice(group[index], group[index + 1])
        rebuilt.extend([*group[:index], merged, *group[index + 2 :]])
    return rebuilt


def _run(entry: LedgerEntry, order: int) -> Run:
    flags = [entry.kind, entry.confidence]
    if entry.pinned:
        flags.append("pinned")
    if entry.stale:
        flags.append("stale")
    return Run(
        ids=(entry.id,),
        parts=(" ".join(entry.text.split()),),
        refs=entry.refs,
        flags=tuple(flags),
        pinned=entry.pinned,
        order=order,
    )


def _splice(first: Run, second: Run) -> Run:
    """Two runs into one, capped so the result cannot grow without bound."""
    both = (*first.parts, *second.parts)
    parts = tuple(_clip(part, LEDGER_MERGE_MAX_PART_CHARS) for part in both)
    joined = _clip("; ".join(parts), LEDGER_MERGE_MAX_CHARS)
    refs = tuple({ref.location: ref for ref in (*first.refs, *second.refs)}.values())
    return Run(
        ids=(*first.ids, *second.ids),
        parts=(joined,),
        refs=refs,
        flags=("merged",),
        pinned=False,
        order=min(first.order, second.order),
    )


def _clip(text: str, limit: int) -> str:
    """Trim to ``limit`` including the ellipsis, so the cap is a real cap."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _group(runs: Sequence[Run]) -> list[tuple[str, list[Run]]]:
    """Runs by path, groups ordered, entries within a group ordered."""
    groups: dict[str, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.refs[0].path if run.refs else _LAST, []).append(run)

    return [
        (key, sorted(group, key=lambda run: (not run.pinned, run.order)))
        for key, group in sorted(groups.items())
    ]


def _render(runs: Sequence[Run]) -> str:
    lines = [HEADER]
    for key, group in _group(runs):
        lines.append(NO_REFS if key == _LAST else key)
        lines.extend(run.line() for run in group)
    return "\n".join(lines)


def stub(tool: str, detail: str, entry_id: str) -> str:
    """What an elided tool result leaves behind: ``[read x.py:1-40 → L7]``."""
    head = f"{tool} {detail}".strip()
    return f"[{head} → {entry_id}]" if entry_id else f"[{head}]"
