"""Turn-boundary compaction: the cheap path first, the cliff only if it must."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..constants import (
    COMPACTION_CLIFF_FRACTION,
    COMPACTION_KEEP_PROSE,
    COMPACTION_MAX_OUTPUT_TOKENS,
    COMPACTION_THRASH_WINDOW,
    KEEP_RAW_TURNS,
)
from ..contracts.escalation import SideUsage
from ..contracts.model import UtilityModel
from ..contracts.transcript import Block, Transcript
from ..fold.document import HEADER

TRIMMED_NOTICE = "Dropped older turns that are already in the session state"

CLIFF_SYSTEM = (
    "You compact an engineering session so it can survive losing its transcript. "
    "You never restate content that is stored elsewhere."
)

CLIFF_PROMPT = (
    "Summarize the session so far so it can survive losing its transcript:\n"
    f"{HEADER}\n"
    "INTENT: … · ANSWERED: … · DECISIONS: … · OPEN: … · FILES: …\n"
    "Ledger entries are stored separately — "
    "do NOT restate them; reference ids only."
)

SUMMARY_HEADER = "[compacted session summary]"

PRESSURE_QUESTION = "Context pressure — split session or drop cold entries?"
PRESSURE_OPTIONS: tuple[str, ...] = ("split", "drop-cold")


@dataclass(frozen=True, slots=True)
class ThrashGuard:
    """Turn numbers at which a cliff compaction fired."""

    stamps: tuple[int, ...] = ()

    def allows(self, turn: int, *, window: int = COMPACTION_THRASH_WINDOW) -> bool:
        return not any(turn - stamp < window for stamp in self.stamps)

    def stamped(self, turn: int) -> ThrashGuard:
        return ThrashGuard(stamps=(*self.stamps, turn))


@dataclass(frozen=True, slots=True)
class Compaction:
    """What compaction did, and what it needs from a human if it could not."""

    transcript: Transcript
    guard: ThrashGuard = field(default_factory=ThrashGuard)
    usage: SideUsage = SideUsage()

    trimmed_turns: int = 0
    cliff: bool = False
    paused: bool = False
    """The thrash guard refused a second cliff inside its window. The caller
    must stop and ask; see :data:`PRESSURE_QUESTION`."""

    notices: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.trimmed_turns > 0 or self.cliff


def compact(
    transcript: Transcript,
    *,
    window_tokens: int,
    folded_turns: int = 0,
    turn: int = 0,
    guard: ThrashGuard | None = None,
    summariser: UtilityModel | None = None,
) -> Compaction:
    """Cheap path, then the cliff only if the transcript is still over."""
    stamps = guard or ThrashGuard()
    threshold = int(window_tokens * COMPACTION_CLIFF_FRACTION)

    trimmed, dropped = trim_folded(transcript, folded_turns)
    notices = (TRIMMED_NOTICE,) if dropped else ()

    if trimmed.tokens < threshold:
        return Compaction(transcript=trimmed, guard=stamps, trimmed_turns=dropped, notices=notices)

    if summariser is None:
        return Compaction(
            transcript=trimmed,
            guard=stamps,
            trimmed_turns=dropped,
            notices=(*notices, "over the compaction threshold and no summariser was supplied"),
        )

    if not stamps.allows(turn):
        return Compaction(
            transcript=trimmed,
            guard=stamps.stamped(turn),
            trimmed_turns=dropped,
            paused=True,
            notices=(*notices, PRESSURE_QUESTION),
        )

    return _cliff(trimmed, stamps, turn, summariser, dropped, notices)


def trim_folded(transcript: Transcript, folded_turns: int) -> tuple[Transcript, int]:
    """Drop whole turns the fold has already digested. No model call."""
    starts = transcript.turn_starts
    digested_here = transcript.local_turn(folded_turns)
    cut_turn = min(max(0, digested_here), max(0, len(starts) - KEEP_RAW_TURNS))
    if cut_turn <= 0:
        return transcript, 0
    return transcript.trimmed_from(starts[cut_turn], turns=cut_turn), cut_turn


def _cliff(
    transcript: Transcript,
    guard: ThrashGuard,
    turn: int,
    summariser: UtilityModel,
    trimmed: int,
    notices: tuple[str, ...],
) -> Compaction:
    """One utility call, then keep the summary and the last two prose blocks."""
    completion = summariser.complete(
        CLIFF_SYSTEM,
        f"{CLIFF_PROMPT}\n\n{_render(transcript)}",
        max_tokens=COMPACTION_MAX_OUTPUT_TOKENS,
    )
    if not completion.usable:
        return Compaction(
            transcript=transcript,
            guard=guard,
            usage=completion.usage,
            trimmed_turns=trimmed,
            notices=(*notices, completion.detail or "compaction produced nothing"),
        )

    summary = Block(
        role="user",
        kind="prose",
        text=f"{SUMMARY_HEADER}\n{completion.text.strip()}",
    )
    tail = [block for block in transcript.blocks if block.is_prose][-COMPACTION_KEEP_PROSE:]

    return Compaction(
        transcript=transcript.with_blocks([summary, *tail]),
        guard=guard.stamped(turn),
        usage=completion.usage,
        trimmed_turns=trimmed,
        cliff=True,
        notices=notices,
    )


def _render(transcript: Transcript) -> str:
    """The transcript as the summariser sees it: one line per block."""
    return "\n".join(
        f"{block.role}/{block.kind}: {' '.join(block.text.split())[:400]}"
        for block in transcript.blocks
        if block.text.strip()
    )
