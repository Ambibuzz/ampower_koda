"""A dependency-free BM25 index over the cached syntax chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import log
from types import MappingProxyType

from ..constants import (
    BM25_B,
    BM25_K1,
    BM25_PATH_BONUS,
    BM25_PATH_REPEAT,
    BM25_SYMBOL_BONUS,
    BM25_SYMBOL_REPEAT,
    BM25_TEXT_BONUS,
    PROSE_COMMENT_SHARE,
)
from ..contracts.chunks import Chunk
from ..contracts.repository import RepositoryIndex, iter_chunks
from .tokenize import counts, tokenize

PROSE_EXEMPT_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".rst", ".txt"})

_COMMENT_PREFIXES = ("#", "//", "/*", "*", "--", "<!--", '"""', "'''")


@dataclass(frozen=True, slots=True)
class Document:
    """One indexed chunk, reduced to what scoring needs."""

    chunk: Chunk
    terms: Mapping[str, int]
    length: int
    prose: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", MappingProxyType(dict(self.terms)))


@dataclass(frozen=True, slots=True)
class LexicalIndex:
    """A scored corpus: documents, postings, and the statistics over them."""

    documents: tuple[Document, ...] = ()
    postings: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    document_frequency: Mapping[str, int] = field(default_factory=dict)
    average_length: float = 0.0
    counted: int = 0
    """How many documents contributed to ``document_frequency``. Not
    ``len(documents)``: fields are retrievable but do not vote, so the IDF
    denominator is smaller than the corpus and has to be tracked separately or
    every IDF is quietly wrong."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "postings", MappingProxyType(dict(self.postings)))
        object.__setattr__(
            self, "document_frequency", MappingProxyType(dict(self.document_frequency))
        )

    def __len__(self) -> int:
        return len(self.documents)

    def idf(self, term: str) -> float:
        """Lucene's BM25 IDF: ``ln(1 + (N − df + 0.5) / (df + 0.5))``."""
        frequency = self.document_frequency.get(term, 0)
        total = max(self.counted, 1)
        return log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))

    def rarity(self, term: str) -> float:
        """``1 / (1 + ln(1 + sites))`` — the reranker's rarity feature."""
        return 1.0 / (1.0 + log(1.0 + self.document_frequency.get(term, 0)))


def build_lexical_index(index: RepositoryIndex) -> LexicalIndex:
    """Build the scored corpus from an analysed repository."""
    documents: list[Document] = []
    postings: dict[str, list[int]] = {}
    frequency: dict[str, int] = {}
    total_length = 0
    counted = 0

    for position, chunk in enumerate(iter_chunks(index)):
        terms = _terms(chunk)
        document = Document(
            chunk=chunk,
            terms=terms,
            length=sum(terms.values()),
            prose=is_prose(chunk),
        )
        documents.append(document)
        total_length += document.length

        for term in terms:
            postings.setdefault(term, []).append(position)

        if chunk.indexable:
            counted += 1
            for term in terms:
                frequency[term] = frequency.get(term, 0) + 1

    return LexicalIndex(
        documents=tuple(documents),
        postings={term: tuple(positions) for term, positions in postings.items()},
        document_frequency=frequency,
        average_length=(total_length / len(documents)) if documents else 0.0,
        counted=counted,
    )


def _terms(chunk: Chunk) -> dict[str, int]:
    """Term counts for one chunk, with path and symbol weighted by repetition."""
    frequencies = counts(tokenize(chunk.body))

    for token in tokenize(chunk.path.replace("/", " ")):
        frequencies[token] = frequencies.get(token, 0) + BM25_PATH_REPEAT
    for token in tokenize(chunk.identity.replace(".", " ")):
        frequencies[token] = frequencies.get(token, 0) + BM25_SYMBOL_REPEAT

    return frequencies


def is_prose(chunk: Chunk) -> bool:
    """Whether a *code* chunk is mostly comment."""
    suffix = chunk.path.rsplit(".", 1)
    if len(suffix) == 2 and f".{suffix[1].lower()}" in PROSE_EXEMPT_EXTENSIONS:
        return False

    lines = [line.strip() for line in chunk.body.split("\n") if line.strip()]
    if not lines:
        return False

    commented = sum(1 for line in lines if line.startswith(_COMMENT_PREFIXES))
    return commented / len(lines) >= PROSE_COMMENT_SHARE


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    position: int
    score: float
    matched: tuple[str, ...]
    """Which query terms this document carried. Feeds the reranker's term
    coverage feature and the confidence calculation, both of which need to know
    *which* terms matched rather than only how well."""


def score(
    index: LexicalIndex,
    query: str,
    *,
    weights: Mapping[str, float] | None = None,
    limit: int = 200,
) -> tuple[ScoredDocument, ...]:
    """Score the corpus against ``query``, best first."""
    terms = counts(tokenize(query, is_query=True))
    if not terms or not index.documents:
        return ()

    weights = weights or {}
    normalised = query.strip().lower()
    accumulated: dict[int, float] = {}
    matched: dict[int, set[str]] = {}

    for term, query_frequency in terms.items():
        positions = index.postings.get(term)
        if not positions:
            continue

        idf = index.idf(term)
        weight = weights.get(term, 1.0) * query_frequency

        for position in positions:
            document = index.documents[position]
            frequency = document.terms.get(term, 0)
            if frequency <= 0:
                continue
            accumulated[position] = accumulated.get(position, 0.0) + weight * idf * _tf(
                frequency, document.length, index.average_length
            )
            matched.setdefault(position, set()).add(term)

    for position in list(accumulated):
        accumulated[position] += _exact_bonus(index.documents[position].chunk, normalised)

    ranked = sorted(
        accumulated.items(),
        key=lambda item: (-item[1], item[0]),
    )[:limit]

    return tuple(
        ScoredDocument(
            position=position,
            score=value,
            matched=tuple(sorted(matched.get(position, ()))),
        )
        for position, value in ranked
    )


def _tf(frequency: int, length: int, average_length: float) -> float:
    """Saturating term frequency with length normalisation."""
    if average_length <= 0:
        return 0.0
    normalised = BM25_K1 * (1.0 - BM25_B + BM25_B * (length / average_length))
    return frequency * (BM25_K1 + 1.0) / (frequency + normalised)


def _exact_bonus(chunk: Chunk, query: str) -> float:
    """Additive credit for containing the query verbatim."""
    if not query:
        return 0.0

    bonus = 0.0
    if chunk.identity and chunk.identity.lower() == query:
        bonus += BM25_SYMBOL_BONUS
    if query in chunk.path.lower():
        bonus += BM25_PATH_BONUS
    if query in chunk.body.lower():
        bonus += BM25_TEXT_BONUS
    return bonus


def symbols(index: LexicalIndex) -> Sequence[str]:
    """Every distinct symbol name in the corpus, in first-seen order."""
    seen: dict[str, None] = {}
    for document in index.documents:
        if document.chunk.identity:
            seen.setdefault(document.chunk.identity, None)
    return tuple(seen)
