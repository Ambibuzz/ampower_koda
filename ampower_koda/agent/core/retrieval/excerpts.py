"""Small, query-aware source windows shared by ranking and its consumers."""

from __future__ import annotations

from ..contracts.chunks import Chunk
from ..contracts.repository import RepositoryIndex
from .tokenize import tokenize

_NOISE = ("#", "//", "/*", "*", "<!--", '"""', "'''", "@")


def excerpt(body: str, query: str = "", *, max_chars: int = 600) -> str:
    """A contiguous source window around matching code, rather than a banner."""
    lines = body.splitlines()
    if not lines or max_chars <= 0:
        return ""
    terms = set(tokenize(query, is_query=True))
    useful = [i for i, line in enumerate(lines)
              if line.strip() and not line.lstrip().startswith(_NOISE)]
    candidates = useful or [i for i, line in enumerate(lines) if line.strip()]
    if not candidates:
        return ""
    best = max(candidates, key=lambda i: (len(terms & set(tokenize(lines[i]))), -i))
    start = max(candidates[0], best - 2)
    # Centering on a long matching line still exposes its match when possible.
    text = "\n".join(lines[start:best + 9]).strip()
    return text[:max_chars].rstrip()


def best_chunks(index: RepositoryIndex, path: str, query: str = "", *,
                symbol: str = "", limit: int = 2) -> tuple[Chunk, ...]:
    """Pick evidence within a related file instead of its first comment chunk."""
    analysis = index.files.get(path)
    if analysis is None:
        return ()
    terms = set(tokenize(query, is_query=True))
    reference_lines = {ref.line for ref in analysis.references if ref.name == symbol}

    def key(chunk: Chunk) -> tuple:
        matched_symbol = bool(symbol) and (
            chunk.identity == symbol or chunk.identity.endswith("." + symbol)
        )
        call_site = any(chunk.span.start <= line <= chunk.span.end for line in reference_lines)
        overlap = len(terms & set(tokenize(chunk.body)))
        return (-int(matched_symbol or call_site), -overlap,
                -int(chunk.kind == "symbol"), chunk.span.start, chunk.digest)

    meaningful = [chunk for chunk in analysis.chunks if any(char.isalnum() for char in chunk.body)]
    return tuple(sorted(meaningful, key=key)[:limit])
