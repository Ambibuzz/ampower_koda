"""One retrieval pass per user message, spent line by line."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..constants import (
    LEDGER_RECENT_WINDOW,
    WORKING_SET_EXCERPT_CHARS,
    WORKING_SET_MAX_EDITED,
    WORKING_SET_MAX_SPANS,
    WORKING_SET_SEARCH_LIMIT,
    WORKING_SET_WEAK_COVERAGE,
)
from ..contracts.ledger import Ledger
from ..contracts.retrieval import Hit
from ..contracts.working_set import WorkingSet, WorkingSpan
from ..identity import anchor_id
from ..retrieval.engine import Retriever, search
from ..tokens import estimate_tokens

HEADER = "WORKING SET (most relevant spans for this message)"


def working_set_for(
    message: str,
    retriever: Retriever,
    *,
    ledger: Ledger | None = None,
    edited: Sequence[str] = (),
    max_tokens: int = 0,
) -> WorkingSet:
    """Build the block for one user message. One search, no model call."""
    if not message.strip():
        return WorkingSet()

    result = search(retriever, message, limit=WORKING_SET_SEARCH_LIMIT)
    spans = _collect(result.hits, ledger)
    if not spans:
        return WorkingSet(coverage=result.confidence)

    return _render(spans, edited, coverage=result.confidence, max_tokens=max_tokens)


def _collect(hits: Sequence[Hit], ledger: Ledger | None) -> list[WorkingSpan]:
    """The two *span* tiers, deduplicated by location, priority order preserved."""
    spans: list[WorkingSpan] = []
    seen: set[str] = set()

    for span in (*_retrieved(hits), *_established(ledger)):
        if span.location in seen:
            continue
        seen.add(span.location)
        spans.append(span)
        if len(spans) >= WORKING_SET_MAX_SPANS:
            break

    return spans


def _retrieved(hits: Sequence[Hit]) -> Iterable[WorkingSpan]:
    """Tier one: scored against this message, with anchors."""
    for hit in hits:
        excerpt = _excerpt(hit.chunk.body)
        yield WorkingSpan(
            location=hit.location,
            excerpt=excerpt,
            symbol=hit.symbol,
            score=hit.score,
            anchor=anchor_id(hit.path, excerpt),
            origin="retrieved",
        )


def _established(ledger: Ledger | None) -> Iterable[WorkingSpan]:
    """Tier two: refs from the last twelve live entries, newest first."""
    if ledger is None:
        return
    for entry in ledger.recent(LEDGER_RECENT_WINDOW):
        for ref in entry.refs:
            if ref.start:
                yield WorkingSpan(location=ref.location, origin="established")


def _render(
    spans: Sequence[WorkingSpan],
    edited: Sequence[str],
    *,
    coverage: float,
    max_tokens: int,
) -> WorkingSet:
    """Header, optional warning, spans, edited clause — spent line by line."""
    lines = [HEADER]
    warning = _warning(coverage)
    truncated = False

    def fits(candidate: Sequence[str]) -> bool:
        return not max_tokens or estimate_tokens("\n".join(candidate)) <= max_tokens

    if warning:
        if not fits([*lines, warning]):
            return WorkingSet(coverage=coverage, truncated=True)
        lines.append(warning)

    kept: list[WorkingSpan] = []
    for span in spans:
        line = span.line()
        if not fits([*lines, line]):
            truncated = True
            break
        lines.append(line)
        kept.append(span)

    changed = _changed_clause(edited)
    if changed and fits([*lines, changed]):
        lines.append(changed)
    elif changed:
        truncated = True

    if not kept:
        return WorkingSet(coverage=coverage, truncated=truncated)

    text = "\n".join(lines)
    return WorkingSet(
        text=text,
        spans=tuple(kept),
        tokens=estimate_tokens(text),
        coverage=coverage,
        truncated=truncated,
        broadened=True,
    )


def _warning(coverage: float) -> str:
    """The self-report, below the same floor the escalation ladder calls weak."""
    if coverage >= WORKING_SET_WEAK_COVERAGE:
        return ""
    return (
        f"[weak automatic retrieval: {coverage:.0%} query coverage; "
        "verify before relying on these spans]"
    )


def _changed_clause(edited: Sequence[str]) -> str:
    if not edited:
        return ""
    names = list(dict.fromkeys(edited))[:WORKING_SET_MAX_EDITED]
    return "changed this session: " + ", ".join(names)


def _excerpt(body: str) -> str:
    """The first non-blank line, clipped."""
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped:
            if len(stripped) > WORKING_SET_EXCERPT_CHARS:
                return stripped[: WORKING_SET_EXCERPT_CHARS - 1] + "…"
            return stripped
    return ""
