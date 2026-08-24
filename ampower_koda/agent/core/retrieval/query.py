"""Reading the question: what kind it is, and how many ways to ask it."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log1p

from ..constants import (
    IDENTIFIER_MAX_SITES,
    ISSUE_QUERY_MIN_CHARS,
    VIEW_WEIGHTS,
)
from .bm25 import LexicalIndex
from .tokenize import is_code_shaped, tokenize

Route = str

_FENCE = re.compile(r"```")
_QUOTED = re.compile(r"[`'\"]([A-Za-z_][\w.]{2,})[`'\"]")
_PATH_LIKE = re.compile(r"[\w./-]*[\w-]+\.[A-Za-z]{1,5}\b")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,}")
_CODE_MARKERS = ("traceback", "expected output", "expected behavior", "expected behaviour")

MAX_VIEWS = 4


@dataclass(frozen=True, slots=True)
class QueryView:
    """One way of asking the same question."""

    kind: str
    text: str
    weight: float


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """Everything decided about a query before any scoring happens."""

    original: str
    route: Route
    views: tuple[QueryView, ...]
    identifiers: tuple[str, ...] = ()
    exact_symbol: str = ""
    """The symbol this query names outright, if it names one. Non-empty is what
    makes the route ``exact``, and it is carried rather than recomputed because
    the structural leg wants it too."""

    @property
    def is_issue_report(self) -> bool:
        return len(self.views) > 1

    @property
    def anchored(self) -> bool:
        """Whether the query names something the repository could contain."""
        return bool(self.exact_symbol or self.identifiers)


def plan_query(
    query: str,
    index: LexicalIndex,
    *,
    known_symbols: Mapping[str, int] | None = None,
) -> QueryPlan:
    """Decide how to ask ``query``."""
    cleaned = query.strip()
    symbols = known_symbols if known_symbols is not None else _symbol_sites(index)

    exact = _exact_symbol(cleaned, symbols)
    identifiers = _identifiers(cleaned, symbols)

    views = [QueryView(kind="original", text=cleaned, weight=VIEW_WEIGHTS["original"])]
    if not exact and _looks_like_issue_report(cleaned):
        views.extend(_issue_views(cleaned, identifiers))

    return QueryPlan(
        original=cleaned,
        route="exact" if exact else "hybrid",
        views=tuple(views),
        identifiers=identifiers,
        exact_symbol=exact,
    )


def _exact_symbol(query: str, symbols: Mapping[str, int]) -> str:
    """The symbol this query *is*, if the whole query names one."""
    candidate = query.strip().strip("`'\"")
    if not candidate or " " in candidate:
        return ""
    return candidate if candidate in symbols else ""


def _symbol_sites(index: LexicalIndex) -> Mapping[str, int]:
    """Symbol name → how many chunks carry it, bare and qualified."""
    sites: dict[str, int] = {}
    for document in index.documents:
        identity = document.chunk.identity
        if not identity:
            continue
        for name in {identity, identity.rsplit(".", 1)[-1]}:
            sites[name] = sites.get(name, 0) + 1
    return sites


def _looks_like_issue_report(query: str) -> bool:
    """Long, fenced, or carrying a traceback or an expected/actual section."""
    if len(query) >= ISSUE_QUERY_MIN_CHARS:
        return True
    lowered = query.lower()
    return bool(_FENCE.search(query)) or any(marker in lowered for marker in _CODE_MARKERS)


def _issue_views(query: str, identifiers: Sequence[str]) -> list[QueryView]:
    """The four derived views, strongest weight last to build."""
    views: list[QueryView] = []

    title = _title_line(query)
    if title and title != query:
        views.append(QueryView(kind="title", text=title, weight=VIEW_WEIGHTS["title"]))

    if identifiers:
        views.append(
            QueryView(
                kind="identifiers",
                text=" ".join(identifiers),
                weight=VIEW_WEIGHTS["identifiers"],
            )
        )
        for name in identifiers[:2]:
            views.append(QueryView(kind="anchor", text=name, weight=VIEW_WEIGHTS["anchor"]))

    path = _first_known_path(query)
    if path:
        views.append(QueryView(kind="path", text=path, weight=VIEW_WEIGHTS["path"]))

    return sorted(views, key=lambda view: -view.weight)[:MAX_VIEWS]


def _title_line(query: str) -> str:
    """The first non-blank, non-comment line, capped."""
    for line in query.split("\n"):
        stripped = line.strip().lstrip("#").strip()
        if stripped and not stripped.startswith(("```", ">")):
            return stripped[:240]
    return ""


def _identifiers(query: str, symbols: Mapping[str, int]) -> tuple[str, ...]:
    """Names in the query that the repository actually defines, best first."""
    quoted = {match.group(1) for match in _QUOTED.finditer(query)}
    in_code = {
        match.group(0)
        for block in _fenced_blocks(query)
        for match in _IDENTIFIER.finditer(block)
    }

    scored: dict[str, float] = {}
    for match in _IDENTIFIER.finditer(query):
        word = match.group(0)
        sites = symbols.get(word) or symbols.get(word.rsplit(".", 1)[-1])
        if not sites or sites > IDENTIFIER_MAX_SITES:
            continue

        score = 1.0 / (1.0 + log1p(sites))
        if word in quoted:
            score += 5.0
        if word in in_code:
            score += 4.0
        if is_code_shaped(word):
            score += 3.0
        scored[word] = max(scored.get(word, 0.0), score)

    return tuple(sorted(scored, key=lambda name: (-scored[name], name))[:6])


def _fenced_blocks(query: str) -> list[str]:
    """The contents of every fenced block. Odd fences are ignored, not repaired."""
    parts = _FENCE.split(query)
    return parts[1::2] if len(parts) >= 3 else []


def _first_known_path(query: str) -> str:
    """The first path-shaped token in the query. Empty when there is none."""
    for match in _PATH_LIKE.finditer(query):
        candidate = match.group(0)
        if "/" in candidate or candidate.count(".") == 1:
            return candidate
    return ""


def merged_weight(weight: float, rank: int, decay: float) -> float:
    """``weight / (1 + decay × rank)`` — the multi-view merge."""
    return weight / (1.0 + decay * rank)


def query_terms(query: str) -> tuple[str, ...]:
    """The query's tokens as the scorer sees them. Shared with the reranker."""
    return tokenize(query, is_query=True)
