"""The shapes one turn passes through, and the two seams it needs from outside."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from ..budget.allocator import ContextBudget
from ..tokens import estimate_tokens
from .escalation import SideUsage
from .ledger import Ledger
from .prompt import CachePlan, TranscriptMarker
from .repo_map import FileRanks, RepoMap
from .session import SessionContext
from .transcript import Transcript
from .working_set import WorkingSet


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One call the model wants made."""

    id: str
    tool: str
    arguments: Mapping[str, object] = field(default_factory=dict)

    @property
    def detail(self) -> str:
        """The arguments rendered for a stub. Short, ordered, no braces."""
        parts = [f"{value}" for key, value in sorted(self.arguments.items()) if value != ""]
        return " ".join(parts)[:200]


@dataclass(frozen=True, slots=True)
class TurnUsage:
    """What one model call cost, split the way the two budgets need it."""

    input_tokens: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output_tokens: int = 0

    @property
    def processed(self) -> int:
        return self.input_tokens + self.cache_write + self.output_tokens

    @property
    def observed(self) -> int:
        return self.processed + self.cache_read

    def plus(self, other: TurnUsage) -> TurnUsage:
        return TurnUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_write=self.cache_write + other.cache_write,
            cache_read=self.cache_read + other.cache_read,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """One round's answer: what it said, what it wants to call, what it cost."""

    text: str = ""
    calls: tuple[ToolCall, ...] = ()
    usage: TurnUsage = TurnUsage()

    stopped_at_limit: bool = False
    """The output hit ``max_tokens``. The loop offers up to two continuations
    and then says plainly that it was cut off — a truncated answer read as a
    complete one is worse than an error."""

    failed: bool = False
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Everything a driver needs to build one provider request, as one value."""

    plan: CachePlan
    transcript: Transcript
    tools: tuple[str, ...] = ()

    max_tokens: int = 0
    force_terminal: bool = False
    """Send ``tool_choice: none``. The loop has decided this is the last round
    and is asking for prose; offering tools would invite a call it will then
    have to refuse."""

    @property
    def system_text(self) -> str:
        """The system blocks concatenated, for a driver that wants one string."""
        return self.plan.system_text


@runtime_checkable
class ChatModel(Protocol):
    """The conversational model call. The only thing the loop cannot do itself."""

    def respond(self, request: ModelRequest) -> ModelTurn:
        """Send one request. Return what the model said and wants to call."""


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What a tool produced. Never an exception."""

    text: str = ""
    ok: bool = True

    truncated: bool = False
    dropped: int = 0
    """How many rows, files or characters did not make it. Zero when unknown —
    the marker still says *that* something was cut, which is the part the model
    cannot infer, because a tool result is a leaf and can never ask."""

    entry_text: str = ""
    """What the ledger should record, when the caller cannot work it out from
    the rendered text alone — a read's *resolved* span, for one. Empty means
    "distil it normally"."""

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@runtime_checkable
class ToolHost(Protocol):
    """The tools the core cannot implement against a read-only port."""

    def call(self, name: str, arguments: Mapping[str, object]) -> ToolOutcome:
        """Run one tool. Must not raise."""


@dataclass(frozen=True, slots=True)
class Session:
    """Everything one conversation carries between turns."""

    workspace: object
    """The ``Workspace`` port cold start was built from. Carried here because
    :class:`~…contracts.session.SessionContext` deliberately does not hold it —
    a context is a *value*, and a port is a live object. The turn needs both."""

    model_id: str
    """Which model the prompt is assembled for. Only its *cache limits* are
    consulted, so what matters is the family in the id rather than the version."""

    context: SessionContext
    retriever: object
    """The built :class:`~…retrieval.engine.Retriever`. Untyped here only to
    keep this contracts module free of a dependency on the retrieval package."""

    ranks: FileRanks
    repo_map: RepoMap
    budget: ContextBudget

    ledger: Ledger = field(default_factory=Ledger)
    transcript: Transcript = field(default_factory=Transcript)
    state: object = None
    """The fold's :class:`~…fold.document.SessionState`, or ``None`` until the
    fourth turn."""

    marker: TranscriptMarker | None = None
    guard: object = None
    """The compaction :class:`~…elide.compact.ThrashGuard`."""

    edited: tuple[str, ...] = ()
    turn: int = 0
    notes: tuple[str, ...] = ()

    def advanced(self, **changes: object) -> Session:
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TurnResult:
    """One turn, and the session to hand to the next one."""

    answer: str
    session: Session

    rounds: int = 0
    usage: TurnUsage = TurnUsage()
    side_usage: SideUsage = SideUsage()
    """Escalation, fold and compaction calls — charged separately because the
    loop cannot see them in its own request/response pair."""

    calls: tuple[str, ...] = ()
    stop_reason: str = "answered"
    working_set: WorkingSet = field(default_factory=WorkingSet)
    notes: tuple[str, ...] = ()

    @property
    def total_tokens(self) -> int:
        return self.usage.observed + self.side_usage.total_tokens
