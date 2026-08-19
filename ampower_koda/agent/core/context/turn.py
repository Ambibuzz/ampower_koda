"""Turn boundaries — where a transcript may be cut, and where it may not."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import CoreError


@dataclass(frozen=True, slots=True)
class TurnBoundaries:
    """The transcript index each turn began at, plus a monotonic turn counter."""

    starts: tuple[int, ...] = ()
    sequence: int = 0
    """Increments once per turn and never decreases, even across a compaction
    that discards the turns it counted. Checkpoints are named by it, so reusing
    a number would make two different states share a name."""

    @property
    def count(self) -> int:
        return len(self.starts)

    @property
    def current_start(self) -> int | None:
        """Index the in-flight turn began at, or ``None`` before the first one."""
        return self.starts[-1] if self.starts else None


def begin_turn(boundaries: TurnBoundaries, transcript_length: int) -> TurnBoundaries:
    """Record a turn beginning at ``transcript_length``."""
    if transcript_length < 0:
        raise CoreError(f"transcript length cannot be negative: {transcript_length}")
    if boundaries.starts and transcript_length < boundaries.starts[-1]:
        raise CoreError(
            f"turn cannot begin at {transcript_length}, "
            f"before the previous turn at {boundaries.starts[-1]}"
        )
    return TurnBoundaries(
        starts=(*boundaries.starts, transcript_length),
        sequence=boundaries.sequence + 1,
    )


def turn_span(
    boundaries: TurnBoundaries,
    turn_index: int,
    transcript_length: int,
) -> tuple[int, int]:
    """Return the half-open ``[start, end)`` message range of one turn."""
    if not 0 <= turn_index < boundaries.count:
        raise CoreError(f"no turn at index {turn_index}")
    start = boundaries.starts[turn_index]
    end = (
        boundaries.starts[turn_index + 1]
        if turn_index + 1 < boundaries.count
        else transcript_length
    )
    return start, max(start, end)


def safe_cut_index(boundaries: TurnBoundaries, keep_turns: int) -> int:
    """The lowest index a transcript may be truncated to, keeping ``keep_turns``."""
    if keep_turns <= 0:
        raise CoreError("keep_turns must be positive; cutting the whole transcript is not a trim")
    if boundaries.count <= keep_turns:
        return 0
    return boundaries.starts[boundaries.count - keep_turns]


def rebase(boundaries: TurnBoundaries, removed_before: int) -> TurnBoundaries:
    """Shift boundaries after ``removed_before`` messages were cut from the head."""
    if removed_before < 0:
        raise CoreError(f"cannot remove a negative number of messages: {removed_before}")
    if removed_before == 0:
        return boundaries

    shifted = tuple(max(0, start - removed_before) for start in boundaries.starts)
    survivors = tuple(start for start in shifted if start > 0)
    head = (0,) if len(survivors) < len(shifted) else ()
    return TurnBoundaries(starts=head + survivors, sequence=boundaries.sequence)
