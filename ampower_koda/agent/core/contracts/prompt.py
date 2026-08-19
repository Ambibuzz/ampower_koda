"""The shape of a request, and where its cache boundaries fall."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..errors import CoreError

CacheTtl = Literal["none", "5m", "1h"]
"""How long a breakpoint's prefix should be kept.

``5m`` for the system blocks: they are written once and read for the rest of the
session, and "5 minutes loses nothing at 30 seconds and everything at 400".
``1h`` for the rolling transcript marker, which is the one boundary that moves —
and the one whose expiry costs a whole conversation rewrite."""

BlockRole = Literal["map+memory", "system+tools", "tail"]


@dataclass(frozen=True, slots=True)
class PromptBlock:
    """One addressable region of the request."""

    role: BlockRole
    text: str
    ttl: CacheTtl = "none"

    breakpoint: bool = False
    """Whether a cache boundary is emitted *after* this block. Not every block
    earns one: a boundary below the provider's minimum cacheable width costs a
    write and returns nothing."""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class TranscriptMarker:
    """A rolling cache boundary inside the conversation."""

    index: int
    ttl: CacheTtl = "1h"

    def __post_init__(self) -> None:
        if self.index < 0:
            raise CoreError(f"marker index must be non-negative, got {self.index}")


@dataclass(frozen=True, slots=True)
class CachePlan:
    """Everything a driver needs to build one request with correct boundaries."""

    blocks: tuple[PromptBlock, ...] = ()
    """System blocks, in order. Map first, role prompt second — see
    :mod:`ampower_koda.agent.core.prompt.cache` for why that order is worth two
    thousand tokens per role."""

    marker: TranscriptMarker | None = None
    """The rolling boundary in the transcript, or ``None`` when the conversation
    is too short for one to pay."""

    previous_marker: TranscriptMarker | None = None
    """The marker from the last request, re-emitted. A provider searches
    backwards from a breakpoint for a bounded number of blocks, and a round that
    appends several tool calls can move the marker past that window — at which
    point the previous entry is evicted and the whole prefix is rewritten.
    Pinning the older one first keeps it alive."""

    tail: str = ""
    """The single trailing user message: session state, the ledger, the working
    set. Nothing is cached behind it, so appending to it invalidates nothing."""

    session_id: str = ""
    """The one routing control worth sending. A prompt cache lives on one
    upstream instance, and a load balancer that only enables sticky routing
    *after* observing cache usage decides too late to help the early rounds that
    establish the prefix."""

    @property
    def breakpoints(self) -> int:
        """Prefix boundaries emitted, marker excluded."""
        return sum(1 for block in self.blocks if block.breakpoint)

    def block(self, role: BlockRole) -> PromptBlock | None:
        for block in self.blocks:
            if block.role == role:
                return block
        return None

    @property
    def system_text(self) -> str:
        """The system blocks concatenated, in order."""
        return "\n\n".join(block.text for block in self.blocks if not block.is_empty)


@dataclass(frozen=True, slots=True)
class PromptBudget:
    """The per-region ceilings a plan is assembled against."""

    map_tokens: int = 2000
    memory_tokens: int = 800
    tail_tokens: int = 0
    """0 means unbounded. The tail is uncached and therefore fully paid for, but
    it is also where the ledger and the working set live, and both already carry
    their own allocator-derived caps."""

    reserved_breakpoints: int = 1
    """Boundaries held back from the prefix for the rolling marker. Providers cap
    the total, and spending the last one on a system block would leave the
    transcript uncacheable — which is the expensive half."""


@dataclass(frozen=True, slots=True)
class Message:
    """One transcript entry, reduced to what cache planning needs."""

    role: Literal["user", "assistant"]
    text: str = ""

    plain: bool = True
    """False for a message carrying structured content — a tool call, a tool
    result, an image. The rolling marker may only be pinned on a plain message,
    because a boundary inside a structured block is not addressable."""

    tokens: int = 0
    """Estimated size, filled by the caller. Carried rather than recomputed so
    the planner never has to hold an estimator."""


@dataclass(frozen=True, slots=True)
class ModelCacheLimits:
    """What one model family will actually cache."""

    min_cacheable: int = 1024
    max_breakpoints: int = 4
    families: tuple[str, ...] = field(default_factory=tuple)
