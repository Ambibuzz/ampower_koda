"""The block that fixes a specific hole: a chat turn had no question-aware
retrieval at all.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkingSpan:
    """One line of the block: where, how sure, and enough text to recognise it."""

    location: str
    excerpt: str = ""
    symbol: str = ""
    score: float = 0.0
    anchor: str = ""
    """Short content digest. Present only on retrieved spans — the ones actually
    scored against this message — because it is what lets a later turn ask
    whether the quoted line still says what it said."""

    origin: str = "retrieved"
    """``retrieved`` · ``established`` · ``edited``. Rendered, because the three
    mean different things to a reader: one is a guess about this message, one is
    something the session already confirmed, and one is something it changed."""

    def line(self) -> str:
        if self.origin == "established":
            return f"{self.location} — established earlier this session"
        anchor = f"@{self.anchor}" if self.anchor else ""
        score = f" [{self.score:.3f}]" if self.score else ""
        symbol = f" {self.symbol}" if self.symbol else ""
        excerpt = f" — {self.excerpt}" if self.excerpt else ""
        return f"{self.location}{anchor}{score}{symbol}{excerpt}"


@dataclass(frozen=True, slots=True)
class WorkingSet:
    """The rendered block, plus the one thing it changes about the turn."""

    text: str = ""
    spans: tuple[WorkingSpan, ...] = ()
    tokens: int = 0
    coverage: float = 0.0
    """Share of the message's IDF mass the top retrieved span carries. Below the
    weak floor the block warns about itself, in the block, where a reader will
    see it — a confidence number that only appears in a log is a confidence
    number nobody acts on."""

    truncated: bool = False
    """The tail did not fit. *Truncated*, not evicted: spending line by line and
    stopping keeps the highest-ranked spans, where dropping whole sources to fit
    would discard the retrieved tier — the only one scored against the actual
    message."""

    broadened: bool = False
    """A non-empty working set pre-sets ``searchScope.broadened`` for the turn.

    This is a real side effect, returned as a value rather than performed: it
    disarms two turn-level interceptors — the refusal of a first search scoped
    to one file, and the automatic upgrade of a first grep to hybrid mode. Both
    exist to stop a model from starting narrow when it has seen nothing. Once
    this block has put eight scored spans in front of it, it has seen something,
    and the interceptors are correcting a problem that no longer exists."""

    @property
    def is_empty(self) -> bool:
        return not self.text
