"""The last pass: floors, counterparts, and diversity."""

from __future__ import annotations

from collections.abc import Sequence

from ..constants import MAX_HITS_PER_FILE, PROSE_RESULT_PENALTY, SAME_FILE_DECAY
from ..contracts.retrieval import Hit

_TEST_MARKERS = ("test_", "_test", "tests", "spec_", "_spec", "specs", "__tests__")


def decay_same_file(hits: Sequence[Hit]) -> tuple[Hit, ...]:
    """Discount each hit by how many from its file already outrank it, and re-sort."""
    seen: dict[str, int] = {}
    decayed: list[Hit] = []
    for hit in hits:
        count = seen.get(hit.path, 0)
        seen[hit.path] = count + 1
        decayed.append(hit.with_score(hit.score * (SAME_FILE_DECAY**count)))
    return _resort(decayed)


def _resort(hits: Sequence[Hit]) -> tuple[Hit, ...]:
    """Descending score, ties broken by location so the order is total."""
    return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.chunk.location)))


def penalise_prose(hits: Sequence[Hit], prose: frozenset[str]) -> tuple[Hit, ...]:
    """Multiply mostly-comment code chunks down."""
    if not prose:
        return tuple(hits)
    return _resort(
        [
            hit.with_score(hit.score * PROSE_RESULT_PENALTY)
            if hit.chunk.digest in prose
            else hit
            for hit in hits
        ]
    )


def preserve_original_window(
    hits: Sequence[Hit],
    original: Sequence[Hit],
    *,
    head_slots: int = 9,
) -> tuple[Hit, ...]:
    """Force the original query's best hit to rank 1, and two more into the head."""
    if not original or not hits:
        return tuple(hits)

    wanted = [
        digest
        for digest in _first_distinct_files(original, 3)
        if digest in {hit.chunk.digest for hit in hits}
    ]
    if not wanted:
        return tuple(hits)

    ordered = list(hits)

    ordered.insert(0, ordered.pop(_position(ordered, wanted[0])))

    for digest in wanted[1:]:
        position = _position(ordered, digest)
        if position >= head_slots:
            ordered.insert(min(head_slots - 1, len(ordered) - 1), ordered.pop(position))

    return tuple(ordered)


def _position(hits: Sequence[Hit], digest: str) -> int:
    return next(index for index, hit in enumerate(hits) if hit.chunk.digest == digest)


def preserve_direct_files(
    hits: Sequence[Hit],
    original: Sequence[Hit],
    *,
    limit: int,
) -> tuple[Hit, ...]:
    """Guarantee every file the query matched directly survives the cut."""
    if not original:
        return tuple(hits[:limit])

    visible = list(hits[:limit])
    present = {hit.path for hit in visible}

    missing: list[Hit] = []
    for hit in original:
        if hit.path not in present and hit.path not in {other.path for other in missing}:
            missing.append(hit)

    for hit in missing:
        if len(visible) < limit:
            visible.append(hit)
            continue
        victim = _most_over_represented(visible)
        if victim is None:
            break
        visible[victim] = hit

    return tuple(visible)


def add_supplemental(hits: Sequence[Hit], pool: Sequence[Hit], *, slot: int = 9) -> tuple[Hit, ...]:
    """Insert exactly one test↔implementation counterpart, at ``slot``."""
    if not hits:
        return tuple(hits)

    visible = list(hits)
    present = {hit.path for hit in visible}

    for hit in visible[:3]:
        counterpart = _counterpart(hit.path, pool, present)
        if counterpart is not None:
            visible.insert(min(slot, len(visible)), counterpart)
            return tuple(visible)

    return tuple(visible)


def diversify(
    hits: Sequence[Hit],
    *,
    limit: int,
    per_file: int = MAX_HITS_PER_FILE,
) -> tuple[Hit, ...]:
    """Greedy cap on hits per file. Runs last, which is why rerank runs before it."""
    kept: list[Hit] = []
    counts: dict[str, int] = {}
    for hit in hits:
        count = counts.get(hit.path, 0)
        if count >= per_file:
            continue
        counts[hit.path] = count + 1
        kept.append(hit)
        if len(kept) >= limit:
            break
    return tuple(kept)


def _first_distinct_files(hits: Sequence[Hit], count: int) -> tuple[str, ...]:
    """Digests of the first ``count`` hits from distinct files."""
    seen: set[str] = set()
    digests: list[str] = []
    for hit in hits:
        if hit.path in seen:
            continue
        seen.add(hit.path)
        digests.append(hit.chunk.digest)
        if len(digests) >= count:
            break
    return tuple(digests)


def _most_over_represented(hits: Sequence[Hit]) -> int | None:
    """Index of the last hit from the file holding the most slots."""
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.path] = counts.get(hit.path, 0) + 1

    worst = max(counts.values(), default=0)
    if worst < 2:
        return None

    for index in range(len(hits) - 1, -1, -1):
        if counts[hits[index].path] == worst:
            return index
    return None


def _counterpart(path: str, pool: Sequence[Hit], present: set[str]) -> Hit | None:
    """A test for an implementation, or an implementation for a test."""
    want_test = not is_test_path(path)
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    core = stem
    for marker in ("test_", "_test", "spec_", "_spec"):
        core = core.replace(marker, "")
    if len(core) < 3:
        return None

    for candidate in pool:
        if candidate.path in present or is_test_path(candidate.path) is not want_test:
            continue
        other = candidate.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if core in other or other.replace("test_", "").replace("_test", "") == core:
            return candidate
    return None


def is_test_path(path: str) -> bool:
    """Whether a path names a test, by the conventions people actually use."""
    lowered = path.lower()
    basename = lowered.rsplit("/", 1)[-1]
    segments = set(lowered.split("/")[:-1])
    return any(marker in basename for marker in _TEST_MARKERS) or bool(
        segments & {"tests", "test", "specs", "spec", "__tests__"}
    )
