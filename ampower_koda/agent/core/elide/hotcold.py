"""In-turn elision: bound raw tool results without rewriting the cache every round."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import AMORTISATION_RATIO, HOTCOLD_HARD_PRESSURE, HOTCOLD_LOW_WATER
from ..contracts.transcript import Transcript
from .collapse import collapse


@dataclass(frozen=True, slots=True)
class Elision:
    """The new transcript and why it looks the way it does."""

    transcript: Transcript
    collapsed: int = 0
    dropped_tokens: int = 0
    skipped_for_amortisation: bool = False
    """True when a collapse *would* have helped this round and was refused
    because it would not pay for itself. Reported rather than silent: a session
    sitting over budget for ten rounds is worth being able to see, and the
    number that explains it is this one."""

    reason: str = ""

    @property
    def changed(self) -> bool:
        return self.collapsed > 0


def hot_cold(
    transcript: Transcript,
    *,
    max_tokens: int,
    max_count: int,
    rounds_left: int = 0,
    evict_undistilled_reads: bool = False,
) -> Elision:
    """Elide the oldest live results until the transcript is back under half."""
    live_tokens = transcript.live_result_tokens()
    live_count = transcript.live_result_count()
    if live_tokens <= max_tokens and live_count <= max_count:
        return Elision(transcript=transcript, reason="under both marks")

    hard = live_tokens > max_tokens * HOTCOLD_HARD_PRESSURE or (
        live_count > max_count * HOTCOLD_HARD_PRESSURE
    )
    proposal = _collapse_to_low_water(
        transcript,
        target_tokens=int(max_tokens * HOTCOLD_LOW_WATER),
        target_count=max(1, int(max_count * HOTCOLD_LOW_WATER)),
        evict_undistilled_reads=evict_undistilled_reads,
    )
    if not proposal.changed:
        return Elision(transcript=transcript, reason="nothing collapsible")

    if not hard and not _pays_back(transcript, proposal, rounds_left):
        return Elision(
            transcript=transcript,
            skipped_for_amortisation=True,
            reason=f"collapse would rewrite {transcript.suffix_tokens(proposal.first_index):,} "
            f"tokens to save {proposal.dropped_tokens:,}",
        )

    return Elision(
        transcript=proposal.transcript,
        collapsed=proposal.collapsed,
        dropped_tokens=proposal.dropped_tokens,
        reason="hard pressure" if hard else "amortised",
    )


@dataclass(frozen=True, slots=True)
class _Proposal:
    """A collapse that has been *run* rather than estimated."""

    transcript: Transcript
    collapsed: int
    dropped_tokens: int
    first_index: int

    @property
    def changed(self) -> bool:
        return self.collapsed > 0


def _collapse_to_low_water(
    transcript: Transcript,
    *,
    target_tokens: int,
    target_count: int,
    evict_undistilled_reads: bool,
) -> _Proposal:
    """Walk newest → oldest, keeping results until the marks are met."""
    blocks = list(transcript.blocks)
    kept_tokens = 0
    kept_count = 0
    collapsed = 0
    dropped = 0
    first_index = len(blocks)

    for index in reversed(range(len(blocks))):
        block = blocks[index]
        if not block.is_result or block.elided:
            continue

        first_live = kept_count == 0
        fits = kept_tokens + block.tokens <= target_tokens and kept_count + 1 <= target_count
        if first_live or fits:
            kept_tokens += block.tokens
            kept_count += 1
            continue

        replacement = collapse(block, evict_undistilled_reads=evict_undistilled_reads)
        if replacement is block or not replacement.elided or replacement.tokens >= block.tokens:
            kept_tokens += block.tokens
            kept_count += 1
            continue

        blocks[index] = replacement
        collapsed += 1
        dropped += block.tokens - replacement.tokens
        first_index = min(first_index, index)

    return _Proposal(
        transcript=transcript.with_blocks(blocks),
        collapsed=collapsed,
        dropped_tokens=dropped,
        first_index=first_index,
    )


def _pays_back(transcript: Transcript, proposal: _Proposal, rounds_left: int) -> bool:
    """``rounds_left × dropped >= 20 × suffix``."""
    if rounds_left <= 0:
        return True
    suffix = transcript.suffix_tokens(proposal.first_index)
    return rounds_left * proposal.dropped_tokens >= AMORTISATION_RATIO * suffix
