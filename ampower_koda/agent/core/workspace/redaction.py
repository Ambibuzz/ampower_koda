"""Redaction, applied at the source."""

from __future__ import annotations

from collections.abc import Sequence

from ..globs import GlobMatcher, compile_globs

RedactionMatcher = GlobMatcher
"""Takes a relative path; returns the pattern that redacts it, or ``None``."""


def compile_redaction(patterns: Sequence[str]) -> RedactionMatcher:
    """Compile redaction globs into a matcher."""
    return compile_globs(patterns)
