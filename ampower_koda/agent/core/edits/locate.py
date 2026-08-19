"""Finding the string to replace. Exact, or not at all."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Match:
    """Where a located string sits, in characters."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Located:
    """The outcome of one lookup: a match, or a reason there is none."""

    match: Match | None = None
    matches: int = 0
    error: str = ""
    eol: str = "\n"
    """The file's dominant line ending, so the caller can re-spell the
    *replacement* to match. Carried rather than recomputed because the caller
    that inserts the new text is not the one that decided what the file uses."""

    normalized_eol: bool = False
    """The match was found only after re-spelling both strings to the file's
    dominant line ending. Reported, always — a silent normalisation is
    indistinguishable from a fuzzy match to anyone reading the result."""

    @property
    def ok(self) -> bool:
        return self.match is not None


def dominant_eol(content: str) -> str:
    """``"\r\n"`` or ``"\n"``, whichever the file mostly uses."""
    crlf = content.count("\r\n")
    lf = content.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def locate(content: str, needle: str, *, occurrence: int = 0) -> Located:
    """Find ``needle`` in ``content``. Exact, then EOL-normalised, then fail."""
    if not needle:
        return Located(error="old_string is empty — use write to replace a whole file")

    eol = dominant_eol(content)
    found = _all(content, needle)
    if found:
        return replace(_select(found, occurrence), eol=eol)

    respelled = _respell(needle, eol)
    if respelled != needle:
        found = _all(content, respelled)
        if found:
            selected = _select(found, occurrence)
            return Located(
                match=selected.match,
                matches=selected.matches,
                error=selected.error,
                eol=eol,
                normalized_eol=True,
            )

    hint = _crlf_hint(content, needle)
    return Located(matches=0, eol=eol, error=f"no match for old_string{hint}")


def _select(found: list[Match], occurrence: int) -> Located:
    """One match, or an error naming how many there were."""
    if occurrence:
        if occurrence < 1 or occurrence > len(found):
            return Located(
                matches=len(found),
                error=f"occurrence {occurrence} of {len(found)} match(es) does not exist",
            )
        if occurrence == 1 and len(found) > 1:
            return Located(
                matches=len(found),
                error=(
                    f"ambiguous, {len(found)} matches — occurrence 1 is a default, not a "
                    "choice. Extend old_string until it is unique, or name which match."
                ),
            )
        return Located(match=found[occurrence - 1], matches=len(found))

    if len(found) > 1:
        return Located(
            matches=len(found),
            error=(
                f"ambiguous, {len(found)} matches — pass occurrence to choose, "
                "or extend old_string until it is unique"
            ),
        )
    return Located(match=found[0], matches=1)


def _all(content: str, needle: str) -> list[Match]:
    matches: list[Match] = []
    start = content.find(needle)
    while start != -1:
        matches.append(Match(start=start, end=start + len(needle)))
        start = content.find(needle, start + 1)
    return matches


def _respell(text: str, eol: str) -> str:
    """Rewrite every line ending in ``text`` to ``eol``. Idempotent."""
    return text.replace("\r\n", "\n").replace("\n", eol)


def _crlf_hint(content: str, needle: str) -> str:
    """Name CRLF in the failure when it is plausibly the cause."""
    if "\r\n" in content and "\r\n" not in needle and "\n" in needle:
        return " — the file uses CRLF line endings and old_string uses LF"
    return ""
