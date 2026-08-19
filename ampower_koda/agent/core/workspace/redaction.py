"""Redaction, applied at the source."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..errors import RedactedFileError
from ..globs import GlobMatcher, compile_globs

RedactionMatcher = GlobMatcher
"""Takes a relative path; returns the pattern that redacts it, or ``None``."""


def compile_redaction(patterns: Sequence[str]) -> RedactionMatcher:
    """Compile redaction globs into a matcher."""
    return compile_globs(patterns)


def refuse_if_redacted(matcher: RedactionMatcher, path: str) -> None:
    """Raise if ``path`` is redacted."""
    pattern = matcher(path)
    if pattern is not None:
        raise RedactedFileError(path, pattern)


def partition(
    matcher: RedactionMatcher,
    paths: Iterable[str],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Split ``paths`` into ``(allowed, [(redacted_path, pattern), …])``."""
    allowed: list[str] = []
    redacted: list[tuple[str, str]] = []
    for path in paths:
        pattern = matcher(path)
        if pattern is None:
            allowed.append(path)
        else:
            redacted.append((path, pattern))
    return tuple(allowed), tuple(redacted)
