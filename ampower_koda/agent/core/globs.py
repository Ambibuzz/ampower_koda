"""A glob matcher, hand-rolled, with semantics that are stated rather than inherited."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

GlobMatcher = Callable[[str], str | None]
"""Takes a relative path; returns the first pattern that matched, or ``None``."""

_CLASS_SPECIAL = "\\]^"


def glob_to_regex(pattern: str) -> str:
    """Translate one glob into an anchored regular expression."""
    return f"(?s:{_translate(pattern)})\\Z"


def _translate(pattern: str) -> str:
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

        elif char == "{":
            close, commas = _brace_end(pattern, index)
            if close == -1 or not commas:
                index += 1
                out.append(re.escape("{"))
            else:
                bounds = [index, *commas, close]
                alternatives = (_translate(pattern[left + 1 : right]) for left, right in zip(bounds, bounds[1:]))
                out.append(f"(?:{'|'.join(alternatives)})")
                index = close + 1

        else:
            index += 1
            out.append(re.escape(char))

    return "".join(out)


def compile_globs(patterns: Sequence[str], *, anchored: bool = True) -> GlobMatcher:
    """Compile globs; ``*`` stays in one directory, ``**/`` spans any, ``{a,b}`` alternates.

    A pattern without ``/`` matches the file name; one with ``/`` the whole path or, unless
    ``anchored``, the path below any directory (``page/*.py`` matches ``mod/page/a.py``).
    """
    compiled: list[tuple[str, re.Pattern[str], bool]] = []

    for pattern in patterns:
        cleaned = pattern.strip()
        if not cleaned:
            continue
        basename_only = "/" not in cleaned
        regex = glob_to_regex(cleaned)
        if not (anchored or basename_only):
            regex = "(?s:.*/)?" + regex
        compiled.append((cleaned, re.compile(regex), basename_only))

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


def _brace_end(pattern: str, start: int) -> tuple[int, list[int]]:
    """Index of the ``}`` closing the group opened at ``start`` (``-1`` if none) and of its top-level commas."""
    depth, commas, index = 0, [], start
    while index < len(pattern):
        char = pattern[index]
        if char == "[":
            close = _class_end(pattern, index)
            index = index if close == -1 else close
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if not depth:
                return index, commas
        elif char == "," and depth == 1:
            commas.append(index)
        index += 1
    return -1, commas


def _escape_class(body: str) -> str:
    """Escape a character-class body while leaving ranges intact."""
    return "".join(f"\\{char}" if char in _CLASS_SPECIAL else char for char in body)
