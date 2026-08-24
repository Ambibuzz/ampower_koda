"""Round-robin, scoreless, with the head of the list reserved."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ..constants import FANOUT_RESERVED_SLOTS
from ..contracts.retrieval import Hit


def round_robin(
    original: Sequence[Hit],
    alternatives: Sequence[Sequence[Hit]],
    *,
    limit: int,
    reserved: int = FANOUT_RESERVED_SLOTS,
) -> tuple[Hit, ...]:
    """Interleave ``original`` with each alternative list, best-first, no scores."""
    if limit <= 0:
        return ()

    merged: list[Hit] = []
    seen: set[str] = set()

    for hit in original[: max(0, reserved)]:
        _take(hit, merged, seen)

    lists = [list(original[max(0, reserved) :])] + [list(other) for other in alternatives]
    while len(merged) < limit and any(lists):
        progressed = False
        for source in lists:
            if not source:
                continue
            progressed = True
            _take(source.pop(0), merged, seen)
            if len(merged) >= limit:
                break
        if not progressed:
            break

    return tuple(merged[:limit])


def _take(hit: Hit, merged: list[Hit], seen: set[str]) -> None:
    """Append ``hit`` unless its content is already in the list."""
    digest = hit.chunk.digest
    if digest in seen:
        return
    seen.add(digest)
    merged.append(hit)


def added(merged: Sequence[Hit], original: Sequence[Hit]) -> int:
    """How many hits in ``merged`` the original list did not already have."""
    known = {hit.chunk.digest for hit in original}
    return sum(1 for hit in merged if hit.chunk.digest not in known)


def annotate(hits: Sequence[Hit], reading: str, *, already_found: Sequence[Hit]) -> tuple[Hit, ...]:
    """Stamp ``read as: …`` onto the hits the original question never reached."""
    if not reading:
        return tuple(hits)

    known = {hit.chunk.digest for hit in already_found}
    note = f"read as: {reading}"
    return tuple(
        hit if (hit.chunk.digest in known or hit.note) else replace(hit, note=note)
        for hit in hits
    )
