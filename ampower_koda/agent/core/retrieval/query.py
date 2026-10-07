"""Reading the question: what kind it is, and how many ways to ask it."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log1p

from ..constants import (
    IDENTIFIER_MAX_SITES,
    ISSUE_QUERY_MIN_CHARS,
    TYPO_MAX_DISTANCE,
    TYPO_MIN_LENGTH,
    VIEW_WEIGHTS,
)
from .bm25 import LexicalIndex
from .tokenize import is_code_shaped, split_words, stem

Route = str

_FENCE = re.compile(r"```")
_QUOTED = re.compile(r"[`'\"]([A-Za-z_][\w.]{2,})[`'\"]")
_PATH_LIKE = re.compile(r"[\w./-]*[\w-]+\.[A-Za-z]{1,5}\b")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{2,}")
_CODE_MARKERS = ("traceback", "expected output", "expected behavior", "expected behaviour")

MAX_VIEWS = 4


def plain_query(query: str) -> str:
    """Remove a Frappe rich-text wrapper, retaining code in ordinary queries."""
    if not re.match(r'''\s*<div\b[^>]*\bclass=["'][^"']*\bql-editor\b''', query):
        return query.strip()

    class Text(HTMLParser):
        def handle_data(self, data: str) -> None:
            parts.append(data)

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag in {"p", "div", "br", "li", "pre"}:
                parts.append("\n")

        def handle_endtag(self, tag: str) -> None:
            if tag in {"p", "div", "li", "pre"}:
                parts.append("\n")

    parts: list[str] = []
    parser = Text()
    parser.feed(query)
    parser.close()
    return "\n".join(line.strip() for line in "".join(parts).splitlines() if line.strip())


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


def plan_query(
    query: str,
    index: LexicalIndex,
    *,
    known_symbols: Mapping[str, int] | None = None,
) -> QueryPlan:
    """Decide how to ask ``query``."""
    cleaned = corrected_query(plain_query(query), index.document_frequency)
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


_PROSE_WORD = re.compile(r"(?<![\w./-])[a-z]+(?![\w./-])")


def corrected_query(query: str, vocabulary: Mapping[str, int]) -> str:
    """Replace misspelled prose words with the repository term they nearly spell.

    Only unknown lowercase prose words are touched; identifiers, paths and
    dotted names are left alone.
    """
    if not vocabulary:
        return query

    def fix(match: re.Match) -> str:
        word = match.group(0)
        if len(word) < TYPO_MIN_LENGTH or word in vocabulary or stem(word) in vocabulary:
            return word
        return _nearest_term(word, vocabulary) or word

    return _PROSE_WORD.sub(fix, query)


_TypoIndex = tuple[Mapping[str, int], dict[tuple[str, int], list[tuple[str, int, int]]], dict[str, str]]
_TYPO_INDEXES: dict[int, _TypoIndex] = {}


def _letters(word: str) -> int:
    mask = 0
    for char in word:
        mask |= 1 << (ord(char) - 97)
    return mask


def _typo_index(vocabulary: Mapping[str, int]) -> _TypoIndex:
    """Candidate terms by (first letter, length) with a letter-set mask, built once per index.

    The vocabulary itself is kept so a reused ``id`` never matches another index.
    """
    cached = _TYPO_INDEXES.get(id(vocabulary))
    if cached is not None and cached[0] is vocabulary:
        return cached
    buckets: dict[tuple[str, int], list[tuple[str, int, int]]] = {}
    for term, frequency in vocabulary.items():
        if len(term) >= TYPO_MIN_LENGTH and term.isascii() and term.isalpha():
            buckets.setdefault((term[0], len(term)), []).append((term, frequency, _letters(term)))
    if len(_TYPO_INDEXES) >= 4:
        _TYPO_INDEXES.clear()
    built = (vocabulary, buckets, {})
    _TYPO_INDEXES[id(vocabulary)] = built
    return built


def _nearest_term(word: str, vocabulary: Mapping[str, int]) -> str:
    if not word.isascii():
        return ""
    _, buckets, memo = _typo_index(vocabulary)
    if word in memo:
        return memo[word]
    mask = _letters(word)
    best, best_key = "", None
    for size in range(len(word) - TYPO_MAX_DISTANCE, len(word) + TYPO_MAX_DISTANCE + 1):
        for term, frequency, letters in buckets.get((word[0], size), ()):
            # Each edit adds or removes at most one letter from the set, so two
            # edits cannot change more than four: a cheap exact pre-filter.
            if (mask ^ letters).bit_count() > 2 * TYPO_MAX_DISTANCE:
                continue
            distance = _edit_distance(word, term, TYPO_MAX_DISTANCE)
            if distance > TYPO_MAX_DISTANCE:
                continue
            key = (distance, -frequency, term)
            if best_key is None or key < best_key:
                best, best_key = term, key
    memo[word] = best
    return best


def _edit_distance(left: str, right: str, bound: int) -> int:
    """Levenshtein distance with adjacent transposition, abandoned past ``bound``."""
    previous_previous: list[int] = []
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            cost = 0 if a == b else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if i > 1 and j > 1 and a == right[j - 2] and left[i - 2] == b:
                value = min(value, previous_previous[j - 2] + 1)
            current.append(value)
        if min(current) > bound:
            return bound + 1
        previous_previous, previous = previous, current
    return previous[-1]


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
        # Ordinary prose such as "files" or "ranking" is not a named anchor
        # just because a field elsewhere happens to have that name.
        if word not in quoted and word not in in_code and not is_code_shaped(word):
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


_SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    ("py", "pyi", "js", "jsx", "mjs", "cjs", "ts", "tsx", "vue", "json", "html", "htm", "css",
     "scss", "md", "rst", "txt", "yaml", "yml", "toml", "ini", "cfg", "sql", "sh", "csv", "xml")
)


def _first_known_path(query: str) -> str:
    """The first path-shaped token in the query. Empty when there is none.

    A bare ``word.word`` only counts when the suffix is a file extension:
    ``frappe.call`` or ``self.name`` is an attribute access, and treating it as
    a path gave a view of two common words the highest weight in the plan.
    """
    for match in _PATH_LIKE.finditer(query):
        candidate = match.group(0)
        if "/" in candidate:
            return candidate
        if candidate.count(".") == 1 and candidate.rsplit(".", 1)[1].lower() in _SOURCE_EXTENSIONS:
            return candidate
    return ""


def merged_weight(weight: float, rank: int, decay: float) -> float:
    """``weight / (1 + decay × rank)`` — the multi-view merge."""
    return weight / (1.0 + decay * rank)


def named_paths(query: str, paths: Sequence[str]) -> tuple[str, ...]:
    """Normalize a potentially long request once for all candidate filenames."""
    query = query.replace("\\", "/")
    explicit = {match.group(0).lower() for match in _PATH_LIKE.finditer(query)}
    partial_paths = tuple(candidate for candidate in explicit if "/" in candidate)
    words = " " + " ".join(split_words(query)) + " "
    matched = []
    for path in paths:
        normalized_path = path.lower()
        basename = path.rsplit("/", 1)[-1]
        exact = (normalized_path in explicit or basename.lower() in explicit
                 or any(normalized_path.endswith("/" + candidate) for candidate in partial_paths))
        phrase = " ".join(split_words(basename.rsplit(".", 1)[0]))
        if exact or (len(phrase) >= 6 and " " + phrase + " " in words):
            matched.append(path)
    return tuple(matched)
