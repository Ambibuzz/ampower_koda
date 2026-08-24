"""Every way the ledger changes, and the two keys that stop it repeating itself."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..contracts.ledger import (
    BlobRef,
    Confidence,
    Ledger,
    LedgerEntry,
    LedgerKind,
    LedgerSource,
)


def next_id(ledger: Ledger) -> str:
    """``L<n>`` for the next entry, one past the highest ever issued."""
    highest = 0
    for entry in ledger.entries:
        if entry.id.startswith("L") and entry.id[1:].isdigit():
            highest = max(highest, int(entry.id[1:]))
    return f"L{max(highest, len(ledger.entries)) + 1}"


def record(
    ledger: Ledger,
    kind: LedgerKind,
    text: str,
    *,
    refs: Sequence[BlobRef] = (),
    source: LedgerSource = "main",
    confidence: Confidence = "read",
    pinned: bool = False,
) -> tuple[Ledger, str]:
    """Append one entry, or find the one that already says this."""
    candidate = _entry(ledger, kind, text, refs, source, confidence, pinned)
    counts = candidate.kind != "span"
    existing = ledger.id_for(candidate.key)

    if existing:
        seen = ledger.counting(rederived=True) if counts else ledger
        return _repin(seen, existing, pinned), existing

    return ledger.appending(candidate, counted=counts), candidate.id


def _entry(
    ledger: Ledger,
    kind: LedgerKind,
    text: str,
    refs: Sequence[BlobRef],
    source: LedgerSource,
    confidence: Confidence,
    pinned: bool,
) -> LedgerEntry:
    """Build the candidate. Whitespace is collapsed here, at the front door."""
    return LedgerEntry(
        id=next_id(ledger),
        kind=kind,
        text=" ".join(text.split()),
        refs=tuple(refs),
        source=source,
        confidence=confidence,
        pinned=pinned,
    )


def _repin(ledger: Ledger, entry_id: str, pinned: bool) -> Ledger:
    """Pin an existing entry when a deduped offer asked for it."""
    existing = ledger.get(entry_id)
    if existing is None or existing.pinned or not pinned:
        return ledger
    return ledger.replacing(replace(existing, pinned=True))


def record_read(
    ledger: Ledger,
    path: str,
    start: int,
    end: int,
    sha: str,
    *,
    source: LedgerSource = "main",
) -> tuple[Ledger, str]:
    """Record that the session read these lines. One ref, text is the location."""
    ref = BlobRef(path=path, start=start, end=end, sha=sha)
    return record(ledger, "span", ref.location, refs=(ref,), source=source)


def supersede(
    ledger: Ledger,
    entry_id: str,
    text: str,
    *,
    source: LedgerSource = "human",
) -> tuple[Ledger, str]:
    """Append a correction and point the old entry at it."""
    original = ledger.get(entry_id)
    if original is None or not original.is_live:
        return ledger, ""

    correction = _entry(
        ledger,
        original.kind,
        text,
        original.refs,
        source,
        original.confidence,
        original.pinned,
    )
    ledger = ledger.appending(correction, counted=False)
    superseded = ledger.get(entry_id)
    if superseded is None:  # pragma: no cover
        return ledger, correction.id
    return ledger.replacing(replace(superseded, superseded_by=correction.id)), correction.id


def pin(ledger: Ledger, entry_id: str, *, pinned: bool = True) -> Ledger:
    """Set an entry's pin. Core-owned, so it is a flag rather than a new line."""
    entry = ledger.get(entry_id)
    if entry is None or entry.pinned == pinned:
        return ledger
    return ledger.replacing(replace(entry, pinned=pinned))


def mark_stale(ledger: Ledger, entry_id: str) -> Ledger:
    """Latch an entry stale. Never unlatches."""
    entry = ledger.get(entry_id)
    if entry is None or entry.stale:
        return ledger
    return ledger.replacing(replace(entry, stale=True))


def note(ledger: Ledger, text: str, *, pinned: bool = True) -> tuple[Ledger, str]:
    """A human's instruction, pinned by default."""
    return record(ledger, "human_note", text, source="human", confidence="read", pinned=pinned)
