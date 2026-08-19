"""The conversation, in the one shape elision and compaction can safely edit."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from ..errors import CoreError
from ..tokens import estimate_tokens
from .prompt import Message

BlockKind = Literal["prose", "tool_use", "tool_result"]
Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Block:
    """One transcript block: a message, a tool call, or a tool result."""

    role: Role
    kind: BlockKind = "prose"
    text: str = ""

    tool: str = ""
    call_id: str = ""
    """Pairs a ``tool_use`` with its ``tool_result``. Empty on prose."""

    detail: str = ""
    """The call's arguments, already rendered — ``"cache plan"``,
    ``src/x.py:88-120``. Kept alongside the result because a stub is written
    from it *after* the result's bytes are gone."""

    arguments_json: str = ""
    """The call's arguments as JSON, for the driver that has to re-send them.

    Separate from :attr:`detail`, which is lossy on purpose: ``detail`` is a
    200-character human rendering for a stub, and a provider handed it back as a
    tool call would reject the request. Every provider's replay of a prior
    assistant turn needs the arguments *exactly* as the model produced them, and
    a driver that memoised them itself would lose them on the first process
    restart — which is precisely when a background job resumes a session."""

    entry_id: str = ""
    """The ledger handle this result distilled to. What makes an elided result a
    pointer rather than a hole."""

    elided: bool = False
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.kind in ("tool_use", "tool_result") and not self.call_id:
            raise CoreError(f"a {self.kind} block must carry a call id")
        if not self.tokens and self.text:
            object.__setattr__(self, "tokens", estimate_tokens(self.text))

    @property
    def is_result(self) -> bool:
        return self.kind == "tool_result"

    @property
    def is_prose(self) -> bool:
        return self.kind == "prose"

    def elided_to(self, stub: str) -> Block:
        """Return this result with its bytes replaced by ``stub``."""
        return replace(self, text=stub, elided=True, tokens=estimate_tokens(stub))


@dataclass(frozen=True, slots=True)
class Transcript:
    """Blocks in order, plus where each turn begins."""

    blocks: tuple[Block, ...] = ()

    dropped_turns: int = 0
    """Turns the trimmer has already removed from the front.

    Carried because two counters live on different axes and used to be compared
    directly: the fold's ``turns_folded`` counts turns over the *session's*
    lifetime, and ``turn_starts`` lists only the turns still present. Once one
    turn had been trimmed, "digest turn 4" indexed the wrong turn — and then the
    fold stopped finding anything to do while the trimmer kept cutting, which is
    exactly the "drop a turn nothing digested" failure §13 exists to prevent."""

    @property
    def turn_starts(self) -> tuple[int, ...]:
        """Indices where a turn begins **and a cut may land**."""
        return tuple(
            index
            for index, block in enumerate(self.blocks)
            if block.role == "user" and block.is_prose and not self._open_pair_at(index)
        )

    def _open_pair_at(self, index: int) -> bool:
        """Whether a tool call before ``index`` is still awaiting its result."""
        opened = {
            block.call_id for block in self.blocks[:index] if block.kind == "tool_use"
        }
        closed = {block.call_id for block in self.blocks[:index] if block.is_result}
        return bool(opened - closed)

    @property
    def turns(self) -> int:
        return len(self.turn_starts)

    @property
    def tokens(self) -> int:
        return sum(block.tokens for block in self.blocks)

    def results(self) -> tuple[tuple[int, Block], ...]:
        """Every tool result with its index, oldest first."""
        return tuple(
            (index, block) for index, block in enumerate(self.blocks) if block.is_result
        )

    def live_result_tokens(self) -> int:
        """Tokens held by results that have **not** been elided."""
        return sum(block.tokens for _, block in self.results() if not block.elided)

    def live_result_count(self) -> int:
        return sum(1 for _, block in self.results() if not block.elided)

    def suffix_tokens(self, index: int) -> int:
        """Tokens from ``index`` to the end — what a rewrite here re-sends."""
        return sum(block.tokens for block in self.blocks[index:])

    def with_blocks(self, blocks: Iterable[Block]) -> Transcript:
        return replace(self, blocks=tuple(blocks))

    def trimmed_from(self, index: int, *, turns: int) -> Transcript:
        """Cut at ``index`` and record that ``turns`` turns went with it."""
        return replace(
            self.slice_from(index), dropped_turns=self.dropped_turns + max(0, turns)
        )

    def local_turn(self, absolute_turn: int) -> int:
        """A session-lifetime turn number as an index into :attr:`turn_starts`."""
        return absolute_turn - self.dropped_turns

    def slice_from(self, index: int) -> Transcript:
        """Drop everything before ``index``. Refuses a cut that splits a pair."""
        if index <= 0:
            return self
        orphans = {
            block.call_id
            for block in self.blocks[index:]
            if block.is_result
        } - {block.call_id for block in self.blocks[index:] if block.kind == "tool_use"}
        if orphans:
            raise CoreError(
                f"cutting at {index} would orphan tool result(s) {sorted(orphans)}"
            )
        return replace(self, blocks=self.blocks[index:])

    def to_messages(self) -> tuple[Message, ...]:
        """The cache planner's view. Tool blocks are structured, prose is plain."""
        return tuple(
            Message(
                role=block.role,
                text=block.text,
                plain=block.is_prose,
                tokens=block.tokens,
            )
            for block in self.blocks
        )


def transcript_of(blocks: Sequence[Block]) -> Transcript:
    return Transcript(blocks=tuple(blocks))
