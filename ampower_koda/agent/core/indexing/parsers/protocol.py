"""What a language backend must provide, and what it must not."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ...contracts.symbols import ParseResult


@runtime_checkable
class LanguageParser(Protocol):
    """Extracts definitions and references from one language's source text."""

    @property
    def language(self) -> str:
        """Stable language name, e.g. ``"python"``. Recorded on every analysis."""

    @property
    def identity(self) -> str:
        """Version tag for *what this parser extracts*."""

    @property
    def extensions(self) -> tuple[str, ...]:
        """Lowercase file extensions, with the dot: ``(".py", ".pyi")``."""

    def parse(self, path: str, text: str) -> ParseResult:
        """Extract from ``text``."""
