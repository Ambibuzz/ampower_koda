"""What one file yields, and why one file might yield nothing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .chunks import Chunk
from .source import FileStat
from .symbols import Definition, Reference

SkipReason = Literal["redacted", "too_large", "binary", "unreadable", "parse_failed"]


@dataclass(frozen=True, slots=True)
class FileAnalysis:
    """Everything the index knows about one file."""

    path: str
    language: str
    source_hash: str
    definitions: tuple[Definition, ...] = field(default_factory=tuple)
    references: tuple[Reference, ...] = field(default_factory=tuple)
    chunks: tuple[Chunk, ...] = field(default_factory=tuple)
    stat: FileStat | None = None

    recovered: bool = False
    """The file parsed, but with syntax errors the grammar recovered from."""

@dataclass(frozen=True, slots=True)
class SkippedFile:
    """A file discovery found and the index deliberately does not hold."""

    path: str
    reason: SkipReason
    detail: str = ""
