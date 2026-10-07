"""Stable diversity selection; score adjustments are only for local fallback."""

from __future__ import annotations

from collections.abc import Sequence

from ..constants import MAX_HITS_PER_FILE, PROSE_RESULT_PENALTY, SAME_FILE_DECAY
from ..contracts.retrieval import Hit


def decay_same_file(hits: Sequence[Hit]) -> tuple[Hit, ...]:
    seen: dict[str, int] = {}
    decayed: list[Hit] = []
    for hit in hits:
        count = seen.get(hit.path, 0)
        seen[hit.path] = count + 1
        decayed.append(hit.with_score(hit.score * (SAME_FILE_DECAY ** count)))
    return _resort(decayed)


def _resort(hits: Sequence[Hit]) -> tuple[Hit, ...]:
    return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.location)))


def penalise_prose(hits: Sequence[Hit], prose: frozenset[str]) -> tuple[Hit, ...]:
    return _resort(tuple(hit.with_score(hit.score * PROSE_RESULT_PENALTY)
                         if hit.chunk.digest in prose else hit for hit in hits))


def diversify(hits: Sequence[Hit], *, limit: int,
              per_file: int = MAX_HITS_PER_FILE) -> tuple[Hit, ...]:
    """Take ranked hits in order, without inserting lower-ranked alternatives."""
    if limit <= 0:
        return ()
    kept: list[Hit] = []
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for hit in hits:
        count = counts.get(hit.path, 0)
        if count >= per_file or hit.chunk.digest in seen:
            continue
        seen.add(hit.chunk.digest)
        counts[hit.path] = count + 1
        kept.append(hit)
        if len(kept) >= limit:
            break
    return tuple(kept)


