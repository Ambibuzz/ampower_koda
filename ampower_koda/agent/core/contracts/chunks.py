"""The unit of retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..constants import NON_INDEXABLE_ROLES
from ..errors import CoreError
from ..identity import content_hash
from .source import Span
from .symbols import SymbolRole

ChunkKind = Literal["symbol", "lines"]


@dataclass(frozen=True, slots=True)
class Chunk:
    """One addressable, content-hashed region of one file."""

    path: str
    kind: ChunkKind
    span: Span
    body: str

    identity: str = ""
    """The qualified symbol name for a ``symbol`` chunk; empty for ``lines``.
    Part of the digest, so two byte-identical accessors on different classes
    remain distinct chunks."""

    role: SymbolRole | None = None

    indexable: bool = True
    """False excludes the chunk from document-frequency statistics while
    leaving it fully addressable and quotable. Set for field definitions: a
    DocType with two hundred ``Field()`` lines would otherwise drag the IDF of
    every term those lines contain toward zero, quietly making the whole
    repository's ranking worse."""

    digest: str = ""
    """Content hash over ``(path, span, identity, body)``. Computed here rather
    than by the builder so no code path can construct a chunk without one."""

    def __post_init__(self) -> None:
        if not self.path:
            raise CoreError("chunk path cannot be empty")
        if self.kind == "symbol" and not self.identity:
            raise CoreError(f"symbol chunk at {self.path}:{self.span} has no identity")
        if self.kind == "lines" and self.identity:
            raise CoreError(f"lines chunk at {self.path}:{self.span} carries an identity")
        if not self.digest:
            object.__setattr__(
                self,
                "digest",
                content_hash(self.path, self.span.start, self.span.end, self.identity, self.body),
            )

    @property
    def location(self) -> str:
        """``src/cart.py:88-120`` — the form a human and an editor both accept."""
        return f"{self.path}:{self.span}"

    @property
    def label(self) -> str:
        """The chunk's most useful one-line name for a ranked list."""
        return f"{self.identity} ({self.location})" if self.identity else self.location


def is_indexable_role(role: SymbolRole | None) -> bool:
    """Whether a chunk with this role contributes to corpus statistics."""
    return role not in NON_INDEXABLE_ROLES
