"""A glob matcher, hand-rolled, with semantics that are stated rather than inherited."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

GlobMatcher = Callable[[str], str | None]
"""Takes a relative path; returns the first pattern that matched, or ``None``."""

_CLASS_SPECIAL = "\\]^"


def glob_to_regex(pattern: str) -> str:
    """Translate one glob into an anchored regular expression."""
    out: list[str] = []
    index = 0
    length = len(pattern)

    while index < length:
        char = pattern[index]

        if char == "*":
            if pattern.startswith("**", index):
                index += 2
                if pattern.startswith("/", index):
                    index += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
            else:
                index += 1
                out.append("[^/]*")

        elif char == "?":
            index += 1
            out.append("[^/]")

        elif char == "[":
            close = _class_end(pattern, index)
            if close == -1:
                index += 1
                out.append(re.escape("["))
            else:
                body = pattern[index + 1 : close]
                index = close + 1
                negated = body[:1] in ("!", "^")
                members = _escape_class(body[1:] if negated else body)
                out.append(f"[{'^' if negated else ''}{members}]")

        else:
            index += 1
            out.append(re.escape(char))

    return f"(?s:{''.join(out)})\\Z"


def compile_globs(patterns: Sequence[str]) -> GlobMatcher:
    """Compile globs into a matcher."""
    compiled: list[tuple[str, re.Pattern[str], bool]] = []

    for pattern in patterns:
        cleaned = pattern.strip()
        if not cleaned:
            continue
        compiled.append((cleaned, re.compile(glob_to_regex(cleaned)), "/" not in cleaned))

    def matcher(path: str) -> str | None:
        normalised = path.replace("\\", "/").removeprefix("./").lstrip("/")
        basename = normalised.rsplit("/", 1)[-1]
        for source, regex, basename_only in compiled:
            if regex.match(basename if basename_only else normalised):
                return source
        return None

    return matcher


def _class_end(pattern: str, start: int) -> int:
    """Index of the ``]`` closing the class opened at ``start``, or ``-1``."""
    cursor = start + 1
    if cursor < len(pattern) and pattern[cursor] in "!^":
        cursor += 1
    if cursor < len(pattern) and pattern[cursor] == "]":
        cursor += 1
    return pattern.find("]", cursor)


def _escape_class(body: str) -> str:
    """Escape a character-class body while leaving ranges intact."""
    return "".join(f"\\{char}" if char in _CLASS_SPECIAL else char for char in body)
