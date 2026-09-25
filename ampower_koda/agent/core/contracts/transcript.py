"""The conversation, in the one shape elision and compaction can safely edit."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

from ..errors import CoreError
from ..tokens import estimate_tokens
from .prompt import Message

BlockKind = Literal["prose", "context", "tool_use", "tool_result"]
Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Block:
    """One transcript block: a message, a tool call, or a tool result."""

    role: Role
    kind: BlockKind = "prose"
    """Context snapshots are evidence, not user requests pinned by compaction."""
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
    provider_context_json: str = field(default='', repr=False)
    parallel_call_ids: tuple[str, ...] = ()
    """The first call retains its original assistant round for exact provider replay."""

    def __post_init__(self) -> None:
        if self.kind in ("tool_use", "tool_result") and not self.call_id:
            raise CoreError(f"a {self.kind} block must carry a call id")
        if not self.tokens:
            object.__setattr__(self, "tokens", estimate_tokens(self.text + self.arguments_json + self.provider_context_json))

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

    def with_blocks(self, blocks: Iterable[Block]) -> Transcript:
        return replace(self, blocks=tuple(blocks))

    def to_messages(self) -> tuple[Message, ...]:
        """Text and tool results expose a cacheable text block; tool calls do not."""
        return tuple(
            Message(
                role=block.role,
                text=block.text,
                plain=block.kind in ("prose", "context", "tool_result"),
                tokens=block.tokens,
            )
            for block in self.blocks
        )
