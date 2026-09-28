"""Which files to open first: whole-file ranking, then one reranked pick of files and one of definitions.

Measured on sixteen long ERPNext requests (2026-09-28): ranking whole files put an
acceptable file in the top three for 11/16, against 5/16 for the chunk pipeline;
one dedicated rerank call over file cards raised that to 14/16, and a second call
over the definitions inside those files named the exact function for 10/16.
"""

from __future__ import annotations
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import log
from types import MappingProxyType
from ..constants import BM25_B, BM25_K1, CHUNK_CHARS, CHUNK_LONG_LINE_STRIDE
from ..contracts.chunks import Chunk
from ..contracts.mirrors import MirrorSet
from ..contracts.repository import RepositoryIndex
from .tokenize import counts, tokenize
INDEXED_SUFFIXES = (".py", ".js", ".json", ".html", ".vue", ".ts")
NOISE_DIRECTORIES = frozenset({"tests", "test", "patches", "fixtures", "translations", "locale", "node_modules"})
UNIT_KINDS = ("doctype", "report", "page", "print_format", "dashboard_chart", "workspace", "web_form")
PATH_REPEAT = 3
POOL = 40
TOP_FILES = 3
SPANS_PER_FILE = 2
MAX_SPAN_DOCUMENTS = 100
CARD_CHARS = 1900
QUERY_CHARS = 4000
NO_MIRRORS = MirrorSet()
HEADER = "STARTING POINTS (ranked from the request before any model call; confirm before relying on them)"

_UI_STRING = re.compile(r"""\b_{1,2}\(\s*f?(["'])(.{3,200}?)(?<!\\)\1""")
_JSON_TEXT = re.compile(r'"(?:label|description|options)"\s*:\s*"([^"]{3,200})"')


def is_noise(path: str) -> bool:
    """Tests, migrations, fixtures and translations are never where a request starts."""
    parts = path.split("/")
    return (not path.endswith(INDEXED_SUFFIXES) or parts[-1].startswith("test_")
            or any(part in NOISE_DIRECTORIES for part in parts[:-1]))


def unit_of(path: str) -> tuple[str, str, str] | None:
    """``(kind, name, directory)`` for a file inside a DocType/report/page folder."""
    parts = path.split("/")
    for kind in UNIT_KINDS:
        if kind in parts[:-1]:
            at = parts.index(kind)
            if at + 1 < len(parts) - 1:
                return kind, parts[at + 1], "/".join(parts[: at + 2])
    return None


@dataclass(frozen=True, slots=True)
class FileIndex:
    """BM25 over whole files: one document per source file, path words repeated."""

    terms: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    lengths: Mapping[str, int] = field(default_factory=dict)
    postings: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    average_length: float = 0.0
    signatures: Mapping[str, str] = field(default_factory=dict)
    """Content hash per file, so a byte-identical copy ranks once."""
    folders: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    """Directory to the source files directly inside it."""

    def __post_init__(self) -> None:
        for name in ("terms", "lengths", "postings", "signatures", "folders"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def score(self, query: str) -> dict[str, float]:
        wanted = counts(tokenize(query, is_query=True))
        total = max(1, len(self.terms))
        scores: dict[str, float] = {}
        for term, query_frequency in wanted.items():
            paths = self.postings.get(term, ())
            if not paths:
                continue
            idf = log(1.0 + (total - len(paths) + 0.5) / (len(paths) + 0.5))
            for path in paths:
                frequency = self.terms[path][term]
                norm = BM25_K1 * (1.0 - BM25_B + BM25_B * self.lengths[path] / max(self.average_length, 1.0))
                scores[path] = scores.get(path, 0.0) + query_frequency * idf * frequency * (BM25_K1 + 1.0) / (frequency + norm)
        return scores


def build_file_index(index: RepositoryIndex) -> FileIndex:
    terms: dict[str, dict[str, int]] = {}
    lengths: dict[str, int] = {}
    postings: dict[str, list[str]] = {}
    signatures: dict[str, str] = {}
    folders: dict[str, list[str]] = {}
    for path, analysis in index.files.items():
        if is_noise(path):
            continue
        folders.setdefault(path.rpartition("/")[0], []).append(path)
        text = file_text(analysis.chunks)
        bag = counts(tokenize(text))
        for token in tokenize(path.rsplit(".", 1)[0].replace("/", " ").replace("_", " ")):
            bag[token] = bag.get(token, 0) + PATH_REPEAT
        if not bag:
            continue
        terms[path] = bag
        lengths[path] = sum(bag.values())
        signatures[path] = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        for token in bag:
            postings.setdefault(token, []).append(path)
    return FileIndex(
        terms=terms,
        lengths=lengths,
        postings={token: tuple(paths) for token, paths in postings.items()},
        average_length=(sum(lengths.values()) / len(lengths)) if lengths else 0.0,
        signatures=signatures,
        folders={folder: tuple(sorted(paths)) for folder, paths in folders.items()},
    )


def file_text(chunks: Sequence[Chunk]) -> str:
    """A file's source reassembled from its chunks by line number (windows overlap).

    A body longer than the character cap was indexed as overlapping pieces sharing
    one span, in no useful order; they are joined back before placing their lines.
    """
    pieces: dict[tuple[int, int, str, str], list[str]] = {}
    for chunk in chunks:
        pieces.setdefault((chunk.span.start, chunk.span.end, chunk.kind, chunk.identity), []).append(chunk.body)
    lines: dict[int, str] = {}
    loose: list[str] = []
    for (start, end, _, _), bodies in pieces.items():
        whole = _rejoin(bodies)
        body = whole.split("\n")
        if len(body) != end - start + 1:
            loose.append(whole)
            continue
        for number, text in zip(range(start, end + 1), body):
            lines.setdefault(number, text)
    return "\n".join([lines[number] for number in sorted(lines)] + loose)


def _rejoin(bodies: list[str]) -> str:
    """One body from its long-line pieces: each starts ``CHUNK_LONG_LINE_STRIDE`` into the
    one before, so a piece follows the one whose tail it begins with."""
    if len(bodies) == 1:
        return bodies[0]
    overlap = CHUNK_CHARS - CHUNK_LONG_LINE_STRIDE
    remaining = list(dict.fromkeys(bodies))
    follows = {i: j for i, a in enumerate(remaining) for j, b in enumerate(remaining)
               if i != j and len(a) == CHUNK_CHARS and b.startswith(a[CHUNK_LONG_LINE_STRIDE:][:overlap])}
    successors = set(follows.values())
    first = next((i for i in range(len(remaining)) if i not in successors), 0)
    order, seen = [first], {first}
    while order[-1] in follows and follows[order[-1]] not in seen:
        order.append(follows[order[-1]])
        seen.add(order[-1])
    if len(order) != len(remaining):
        return "\n".join(remaining)  # ambiguous (repeated text): every piece, unordered
    return "".join(remaining[i][:CHUNK_LONG_LINE_STRIDE] for i in order[:-1]) + remaining[order[-1]]


# ---------------------------------------------------------------- ranking


def homes(files: FileIndex, directory: str) -> list[str]:
    """Where edits to a unit land: its controller, then its client script."""
    inside = files.folders.get(directory, ())
    name = directory.rsplit("/", 1)[-1]
    found = [path for path in (f"{directory}/{name}.py", f"{directory}/{name}.js") if path in inside]
    return found or [path for path in inside if not path.endswith(".json")][:2]


def rank_files(index: RepositoryIndex, files: FileIndex, query: str, *,
               mirrors: MirrorSet = NO_MIRRORS, limit: int = POOL) -> list[str]:
    """Files by whole-file BM25; unit metadata speaks for its controller; copies rank once."""
    scores = files.score(query)
    ordered = sorted(scores, key=lambda path: (mirrors.contains(path), -scores[path], path))
    out: list[str] = []
    seen_paths: set[str] = set()
    seen_content: set[str] = set()
    for path in ordered:
        unit = unit_of(path)
        targets = homes(files, unit[2]) if path.endswith(".json") and unit else [path]
        for target in targets:
            signature = files.signatures.get(target, target)
            if target in seen_paths or signature in seen_content or is_noise(target) or target not in index.files:
                continue
            seen_paths.add(target)
            seen_content.add(signature)
            out.append(target)
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------- cards and the brief


