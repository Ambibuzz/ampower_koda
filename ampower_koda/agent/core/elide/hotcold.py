"""In-turn elision: replace older tool results with stubs under input pressure."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.transcript import Transcript
from .collapse import collapse


@dataclass(frozen=True, slots=True)
class Elision:
    """The new transcript and why it looks the way it does."""

    transcript: Transcript
    collapsed: int = 0
    dropped_tokens: int = 0
    reason: str = ""

    @property
    def changed(self) -> bool:
        return self.collapsed > 0


def hot_cold(
    transcript: Transcript,
    *,
    max_tokens: int,
    max_count: int,
    target_tokens: int,
    target_count: int,
    evict_undistilled_reads: bool = False,
    protected_call_ids: frozenset[str] = frozenset(),
) -> Elision:
    """Elide older results down to the caller's pressure target."""
    live_tokens = transcript.live_result_tokens()
    live_count = transcript.live_result_count()
    if live_tokens <= max_tokens and live_count <= max_count:
        return Elision(transcript=transcript, reason="under both marks")

    proposal = _collapse_to_low_water(
        transcript,
        target_tokens=max(0, target_tokens),
        target_count=max(1, target_count),
        evict_undistilled_reads=evict_undistilled_reads,
        protected_call_ids=protected_call_ids,
    )
    if not proposal.changed:
        return Elision(transcript=transcript, reason="nothing collapsible")

    return Elision(
        transcript=proposal.transcript,
        collapsed=proposal.collapsed,
        dropped_tokens=proposal.dropped_tokens,
        reason="tool result budget",
    )


@dataclass(frozen=True, slots=True)
class _Proposal:
    """A collapse that has been *run* rather than estimated."""

    transcript: Transcript
    collapsed: int
    dropped_tokens: int

    @property
    def changed(self) -> bool:
        return self.collapsed > 0


def _collapse_to_low_water(
    transcript: Transcript,
    *,
    target_tokens: int,
    target_count: int,
    evict_undistilled_reads: bool,
    protected_call_ids: frozenset[str],
) -> _Proposal:
    """Walk newest → oldest, keeping results until the marks are met."""
    blocks = list(transcript.blocks)
    kept_tokens = 0
    kept_count = 0
    collapsed = 0
    dropped = 0

    for index in reversed(range(len(blocks))):
        block = blocks[index]
        if not block.is_result or block.elided:
            continue

        first_live = kept_count == 0
        fits = kept_tokens + block.tokens <= target_tokens and kept_count + 1 <= target_count
        if block.call_id in protected_call_ids or first_live or fits:
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

    return _Proposal(
        transcript=transcript.with_blocks(blocks),
        collapsed=collapsed,
        dropped_tokens=dropped,
    )
