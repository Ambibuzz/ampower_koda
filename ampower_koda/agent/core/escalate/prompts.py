"""The two prompts, and the parsers that refuse to trust their answers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from ..constants import FANOUT_MAX, TRANSLATE_MAX_NAMES, TRANSLATE_MAX_OUTPUT_TOKENS
from ..contracts.escalation import Rewrite

TRANSLATE_SYSTEM = (
    "You translate a question about a codebase into the identifiers that "
    "codebase actually uses. You are given a map of its files and definitions. "
    "Choose ONLY names that appear in the map. You never explain, never answer "
    "the question, and never invent a name."
)

FANOUT_ANGLES: tuple[str, ...] = ("symbol", "path", "behavior", "data_flow")

FANOUT_SYSTEM = (
    "You rewrite a question about a codebase into at most "
    f"{FANOUT_MAX} search queries, along these angles: "
    + " · ".join(FANOUT_ANGLES)
    + ". You are given a map of its files and definitions. Write at most one "
    "query per angle, and omit any angle the map gives you no support for — "
    "fewer, grounded queries beat four guesses. Answer with one line per query, "
    "formatted `angle: query`. You never explain and never answer the question."
)

TRANSLATE_MAX_TOKENS = TRANSLATE_MAX_OUTPUT_TOKENS

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

_ANGLED_LINE = re.compile(
    r"^[^A-Za-z]{0,16}(symbol|path|behaviou?r|data[ _-]?flow)[*_`]{0,3}\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)

_ANGLE_ALIASES: Mapping[str, str] = {
    "behaviour": "behavior",
    "dataflow": "data_flow",
    "data_flow": "data_flow",
}


def parse_translation(
    text: str,
    known: Mapping[str, int] | Iterable[str],
    *,
    limit: int = TRANSLATE_MAX_NAMES,
) -> tuple[str, ...]:
    """Identifier-shaped tokens in ``text`` that the repository actually defines."""
    if limit <= 0:
        return ()

    vocabulary = set(known)
    seen: dict[str, None] = {}

    for match in _NAME.finditer(text):
        for candidate in _candidates(match.group(0)):
            if candidate in vocabulary:
                seen.setdefault(candidate, None)
                break
        if len(seen) >= limit:
            break

    return tuple(seen)


def _candidates(token: str) -> tuple[str, ...]:
    """The token, then its suffixes, then its prefixes — longest first."""
    if "." not in token:
        return (token,)

    parts = token.split(".")
    suffixes = [".".join(parts[start:]) for start in range(len(parts))]
    prefixes = [".".join(parts[:end]) for end in range(len(parts) - 1, 0, -1)]
    return tuple(dict.fromkeys([*suffixes, *prefixes]))


def parse_fanout(text: str, original: str, *, limit: int = FANOUT_MAX) -> tuple[Rewrite, ...]:
    """``angle: query`` lines, deduplicated against the original and each other."""
    taken = {_normalise(original)}
    rewrites: list[Rewrite] = []

    for line in text.split("\n"):
        match = _ANGLED_LINE.match(line)
        if not match:
            continue
        query = match.group(2).strip(" `\"'*_")
        key = _normalise(query)
        if not key or key in taken:
            continue
        taken.add(key)
        rewrites.append(Rewrite(text=query, angle=_angle_of(match.group(1))))
        if len(rewrites) >= limit:
            break

    return tuple(rewrites)


def _angle_of(label: str) -> str:
    """One of :data:`FANOUT_ANGLES`, from whatever the model spelled."""
    folded = re.sub(r"[ \-]", "_", label.lower())
    return _ANGLE_ALIASES.get(folded, folded)


def _normalise(text: str) -> str:
    """Casefolded, whitespace-collapsed — what "the same rewrite" means here."""
    return " ".join(text.lower().split())
