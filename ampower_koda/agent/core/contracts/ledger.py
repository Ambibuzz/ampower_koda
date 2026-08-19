"""What the session has established, and the refs that prove it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

from ..errors import CoreError

LedgerKind = Literal[
    "fact",
    "span",
    "symbol",
    "invariant",
    "gotcha",
    "decision",
    "human_note",
    "human_edit",
    "hits",
    "probe",
    "shape",
    "state",
]

LEDGER_KINDS: tuple[str, ...] = (
    "fact",
    "span",
    "symbol",
    "invariant",
    "gotcha",
    "decision",
    "human_note",
    "human_edit",
    "hits",
    "probe",
    "shape",
    "state",
)

LedgerSource = Literal["scout", "main", "human", "verifier"]

Confidence = Literal["read", "inferred"]


@dataclass(frozen=True, slots=True)
class BlobRef:
    """A pointer into the code, checkable against the repository."""

    path: str
    start: int = 0
    end: int = 0
    sha: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            raise CoreError("a blob ref must name a path")
        if self.start < 0 or self.end < 0:
            raise CoreError(f"blob ref {self.path} has a negative line number")
        if self.end and self.end < self.start:
            raise CoreError(f"blob ref {self.path} ends before it starts")

    @property
    def location(self) -> str:
        if not self.start:
            return self.path
        return f"{self.path}:{self.start}-{self.end}" if self.end else f"{self.path}:{self.start}"

    @property
    def read_key(self) -> str:
        """``path:start-end`` — the idempotence key for reads."""
        return self.location


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One thing this session established."""

    id: str
    kind: LedgerKind
    text: str
    refs: tuple[BlobRef, ...] = ()
    source: LedgerSource = "main"
    confidence: Confidence = "read"

    pinned: bool = False
    """Core-owned. A pinned entry is never merged, never dropped, and claimed
    before anything is scored — which is what makes a plan redirect outlive the
    compaction that would otherwise erase the turn it was said in."""

    stale: bool = False
    """Core-owned, and latching. Set when a ref's blob sha no longer matches."""

    superseded_by: str = ""
    """The id of the entry that replaced this one. Set on the *old* entry, which
    stays in the log: append-only means a correction is a new line plus a
    back-pointer, never an edit."""

    def __post_init__(self) -> None:
        if not self.id:
            raise CoreError("a ledger entry must have an id")
        if self.kind not in LEDGER_KINDS:
            raise CoreError(f"unknown ledger kind: {self.kind!r}")
        if not self.text.strip():
            raise CoreError(f"ledger entry {self.id} has no text")

    @property
    def is_live(self) -> bool:
        return not self.superseded_by

    @property
    def evidence_key(self) -> str:
        """``kind:text`` — the idempotence key for everything that is not a read."""
        return f"{self.kind}:{' '.join(self.text.split())}"

    @property
    def key(self) -> str:
        """Whichever idempotence key applies to this kind."""
        return self.refs[0].read_key if self.kind == "span" and self.refs else self.evidence_key

    @property
    def path(self) -> str:
        """The path this entry groups under. Empty when it has no refs."""
        return self.refs[0].path if self.refs else ""


@dataclass(frozen=True, slots=True)
class Ledger:
    """The whole append-only log, plus the two counters that fall out of it."""

    entries: tuple[LedgerEntry, ...] = ()

    distilled: int = 0
    """Evidence entries offered to the ledger, whether or not they were new."""

    rederived: int = 0
    """Offers that hit an existing evidence key. See :attr:`rederivation_rate`."""

    _by_key: Mapping[str, str] = field(default_factory=dict, repr=False)
    """Idempotence key → the id already holding it. Maintained on append rather
    than recomputed on read, because it is also what makes the re-derivation
    count free."""

    def __post_init__(self) -> None:
        if not self._by_key and self.entries:
            index = {entry.key: entry.id for entry in self.entries}
            object.__setattr__(self, "_by_key", index)
        object.__setattr__(self, "_by_key", MappingProxyType(dict(self._by_key)))

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def rederivation_rate(self) -> float:
        """``rederived / distilled`` — how much of this session was rediscovery."""
        return self.rederived / self.distilled if self.distilled else 0.0

    def get(self, entry_id: str) -> LedgerEntry | None:
        return next((entry for entry in self.entries if entry.id == entry_id), None)

    def id_for(self, key: str) -> str:
        """The id currently holding ``key``, **live entries only**."""
        entry_id = self._by_key.get(key, "")
        found = self.get(entry_id) if entry_id else None
        return entry_id if found is not None and found.is_live else ""

    def live(self) -> tuple[LedgerEntry, ...]:
        return tuple(entry for entry in self.entries if entry.is_live)

    def pinned(self) -> tuple[LedgerEntry, ...]:
        return tuple(entry for entry in self.live() if entry.pinned)

    def recent(self, count: int) -> tuple[LedgerEntry, ...]:
        """The newest ``count`` live entries, newest first."""
        return tuple(reversed(self.live()))[: max(0, count)]

    def appending(self, entry: LedgerEntry, *, counted: bool, rederived: bool = False) -> Ledger:
        """Return this ledger with ``entry`` on the end and its key indexed."""
        return replace(
            self,
            entries=(*self.entries, entry),
            distilled=self.distilled + (1 if counted else 0),
            rederived=self.rederived + (1 if rederived else 0),
            _by_key={**dict(self._by_key), entry.key: entry.id},
        )

    def counting(self, *, rederived: bool) -> Ledger:
        """Count an offer that produced no new entry. See :attr:`rederivation_rate`."""
        return replace(
            self,
            distilled=self.distilled + 1,
            rederived=self.rederived + (1 if rederived else 0),
        )

    def replacing(self, entry: LedgerEntry) -> Ledger:
        """Return this ledger with ``entry`` swapped in by id, order preserved."""
        current = self.get(entry.id)
        if current is None:
            return self
        if (current.kind, current.text) != (entry.kind, entry.text):
            raise CoreError(
                f"ledger entry {entry.id} cannot be rewritten in place — "
                "supersede it instead (append-only)"
            )
        return replace(
            self,
            entries=tuple(
                entry if existing.id == entry.id else existing for existing in self.entries
            ),
        )
