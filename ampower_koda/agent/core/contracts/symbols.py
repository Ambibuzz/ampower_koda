"""Definitions and references — the only two facts a parser may report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

from ..errors import CoreError
from .source import Span

SymbolRole = Literal[
    "function",
    "class",
    "method",
    "constant",
    "type",
    "enum",
    "field",
]

ReferenceKind = Literal["call", "class", "type"]

SYMBOL_ROLES: tuple[str, ...] = get_args(SymbolRole)
REFERENCE_KINDS: tuple[str, ...] = get_args(ReferenceKind)


@dataclass(frozen=True, slots=True)
class DefinitionSite:
    """A definition as a parser reports it: flat, with no container chain."""

    name: str
    role: SymbolRole
    extent: Span
    """Every line the definition owns, including its decorators."""

    name_line: int
    """The line the identifier appears on. Line-level, not column-level: no
    consumer needs a column, and carrying one would be a precision this system
    never verifies."""

    def __post_init__(self) -> None:
        if not self.name:
            raise CoreError("definition name cannot be empty")
        if self.role not in SYMBOL_ROLES:
            raise CoreError(f"unknown definition role {self.role!r}")
        if not self.extent.start <= self.name_line <= self.extent.end:
            raise CoreError(
                f"name line {self.name_line} falls outside extent {self.extent}"
            )


@dataclass(frozen=True, slots=True)
class Definition:
    """A definition with its container chain resolved."""

    name: str
    role: SymbolRole
    extent: Span
    name_line: int
    container: tuple[str, ...] = ()
    """Enclosing definition names, outermost first. ``("Cart",)`` for a method
    of ``Cart``; empty at module level."""

    @property
    def qualified_name(self) -> str:
        """``Cart.total`` rather than bare ``total``."""
        return ".".join((*self.container, self.name))

    @property
    def depth(self) -> int:
        return len(self.container)


@dataclass(frozen=True, slots=True)
class Reference:
    """One site where a name is used rather than defined."""

    name: str
    kind: ReferenceKind
    line: int

    def __post_init__(self) -> None:
        if not self.name:
            raise CoreError("reference name cannot be empty")
        if self.kind not in REFERENCE_KINDS:
            raise CoreError(f"unknown reference kind {self.kind!r}")
        if self.line < 1:
            raise CoreError(f"reference line must be 1-based, got {self.line}")


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Everything a parser produces for one file."""

    language: str
    definitions: tuple[DefinitionSite, ...] = field(default_factory=tuple)
    references: tuple[Reference, ...] = field(default_factory=tuple)

    recovered: bool = False
    """True when the grammar hit a syntax error and carried on.

    Error recovery is the main reason a parser generator beats a language's own
    parser here: a file caught mid-edit still yields most of its symbols, where
    ``ast.parse`` would yield none. But a recovered parse is a weaker claim than
    a clean one, so it is recorded rather than smoothed over — a caller that
    wants to distrust those symbols can, and cold start reports the count."""
