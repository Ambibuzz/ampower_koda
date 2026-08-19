"""One file in, one :class:`FileAnalysis` out."""

from __future__ import annotations

from ..contracts.analysis import FileAnalysis
from ..contracts.source import SourceFile
from ..contracts.symbols import ParseResult
from .chunking import build_chunks
from .containers import resolve_definitions
from .parsers.registry import ParserRegistry

PLAIN_TEXT = "text"


def analyze(source: SourceFile, registry: ParserRegistry) -> FileAnalysis:
    """Analyse one file. Pure: same source and registry give the same result."""
    parser = registry.for_path(source.path)
    parsed = parser.parse(source.path, source.text) if parser else ParseResult(language=PLAIN_TEXT)

    definitions = resolve_definitions(parsed.definitions)
    chunks = build_chunks(source.path, source.text, definitions)

    return FileAnalysis(
        path=source.path,
        language=parsed.language,
        source_hash=source.source_hash,
        definitions=definitions,
        references=parsed.references,
        chunks=chunks,
        stat=source.stat,
        recovered=parsed.recovered,
    )
