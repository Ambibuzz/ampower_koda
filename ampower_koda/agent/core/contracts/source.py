"""Where things are, and what a file looked like when we read it."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import CoreError


@dataclass(frozen=True, slots=True, order=True)
class Span:
    """An inclusive, 1-based range of lines."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1:
            raise CoreError(f"span start must be 1-based, got {self.start}")
        if self.end < self.start:
            raise CoreError(f"span end {self.end} precedes start {self.start}")

    @property
    def line_count(self) -> int:
        return self.end - self.start + 1

    def as_slice(self) -> slice:
        """Return the slice selecting these lines from a 0-based line list."""
        return slice(self.start - 1, self.end)

    def contains(self, other: Span) -> bool:
        """True when ``other`` lies wholly inside this span."""
        return self.start <= other.start and other.end <= self.end

    def __str__(self) -> str:
        return f"{self.start}-{self.end}" if self.start != self.end else str(self.start)


def split_lines(text: str) -> tuple[str, ...]:
    """Split ``text`` into lines the way the parsers count them."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return tuple(lines)


@dataclass(frozen=True, slots=True)
class FileStat:
    """The cheap facts about a file, used as a cache fast path only."""

    size: int
    mtime_ns: int

    def key(self) -> str:
        return f"{self.size}:{self.mtime_ns}"


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One file's path and decoded text, as the indexer sees it."""

    path: str
    """Workspace-relative, forward-slashed. Never absolute, on any platform."""

    text: str

    source_hash: str = ""
    """sha256 of the file's raw *bytes*, not of ``text``.

    Hashing before decoding keeps the hash independent of the decoder: if it
    were taken over the decoded string, a change to how BOMs or line endings are
    handled would look like an edit to every file in the repository and
    invalidate the entire chunk cache."""

    stat: FileStat | None = None

    @property
    def lines(self) -> tuple[str, ...]:
        """The text split into lines, without terminators."""
        return split_lines(self.text)


@dataclass(frozen=True, slots=True)
class Overlay:
    """In-memory content that supersedes what is on disk."""

    path: str
    text: str
    origin: str = "buffer"
    """``"buffer"`` for an unsaved editor document, ``"applied"`` for an edit
    this session made. Only used to resolve collisions and to explain them."""
