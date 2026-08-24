"""The chunk cache, and the paranoia it is built on."""

from __future__ import annotations

import json
from typing import Any

from ..constants import (
    CACHE_ENTRY_VERSION,
    CHUNK_CHARS,
    CHUNK_LINES,
    CHUNK_LONG_LINE_STRIDE,
    CHUNK_OVERLAP_LINES,
    CHUNKLESS_ROLES,
    COMMENT_MAX_LINES,
    NON_INDEXABLE_ROLES,
)
from ..contracts.analysis import FileAnalysis
from ..contracts.chunks import Chunk
from ..contracts.source import FileStat, Span
from ..contracts.symbols import Definition, Reference
from ..errors import CoreError
from ..identity import cache_key
from ..identity import fingerprint as _fingerprint
from ..workspace.ports import Workspace
from .parsers.registry import ParserRegistry, registry_identity


def schema_fingerprint(registry: ParserRegistry) -> str:
    """Digest every input that changes what an analysis looks like."""
    return _fingerprint(
        "koda-index",
        CACHE_ENTRY_VERSION,
        CHUNK_LINES,
        CHUNK_CHARS,
        CHUNK_LONG_LINE_STRIDE,
        CHUNK_OVERLAP_LINES,
        COMMENT_MAX_LINES,
        ",".join(CHUNKLESS_ROLES),
        ",".join(NON_INDEXABLE_ROLES),
        registry_identity(registry),
    )


def load(
    workspace: Workspace,
    path: str,
    *,
    fingerprint: str,
    source_hash: str,
    stat: FileStat | None,
) -> tuple[FileAnalysis | None, bool]:
    """Probe the cache for ``path``."""
    raw = workspace.read_cache(cache_key(path))
    if raw is None:
        return None, False

    try:
        entry = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, False

    if not isinstance(entry, dict):
        return None, False
    if entry.get("version") != CACHE_ENTRY_VERSION:
        return None, False
    if entry.get("fingerprint") != fingerprint:
        return None, False
    if entry.get("source_hash") != source_hash:
        return None, False

    try:
        analysis = decode_analysis(entry["analysis"], stat=stat)
    except (KeyError, TypeError, ValueError, CoreError):
        return None, False

    return analysis, entry.get("stat_key") != (stat.key() if stat else None)


def store(workspace: Workspace, analysis: FileAnalysis, *, fingerprint: str) -> None:
    """Write ``analysis`` to the cache. Never raises."""
    entry = {
        "version": CACHE_ENTRY_VERSION,
        "fingerprint": fingerprint,
        "source_hash": analysis.source_hash,
        "stat_key": analysis.stat.key() if analysis.stat else None,
        "analysis": encode_analysis(analysis),
    }
    payload = json.dumps(entry, separators=(",", ":"), sort_keys=True).encode("utf-8")
    workspace.write_cache(cache_key(analysis.path), payload)


def encode_analysis(analysis: FileAnalysis) -> dict[str, Any]:
    return {
        "path": analysis.path,
        "language": analysis.language,
        "source_hash": analysis.source_hash,
        "recovered": analysis.recovered,
        "definitions": [
            {
                "name": definition.name,
                "role": definition.role,
                "start": definition.extent.start,
                "end": definition.extent.end,
                "name_line": definition.name_line,
                "container": list(definition.container),
            }
            for definition in analysis.definitions
        ],
        "references": [
            {"name": reference.name, "kind": reference.kind, "line": reference.line}
            for reference in analysis.references
        ],
        "chunks": [
            {
                "kind": chunk.kind,
                "start": chunk.span.start,
                "end": chunk.span.end,
                "identity": chunk.identity,
                "role": chunk.role,
                "indexable": chunk.indexable,
                "body": chunk.body,
                "digest": chunk.digest,
            }
            for chunk in analysis.chunks
        ],
    }


def decode_analysis(payload: dict[str, Any], *, stat: FileStat | None) -> FileAnalysis:
    """Rebuild an analysis from a cache entry."""
    path = payload["path"]

    definitions = tuple(
        Definition(
            name=item["name"],
            role=item["role"],
            extent=Span(item["start"], item["end"]),
            name_line=item["name_line"],
            container=tuple(item["container"]),
        )
        for item in payload["definitions"]
    )
    references = tuple(
        Reference(name=item["name"], kind=item["kind"], line=item["line"])
        for item in payload["references"]
    )
    chunks = tuple(
        Chunk(
            path=path,
            kind=item["kind"],
            span=Span(item["start"], item["end"]),
            body=item["body"],
            identity=item["identity"],
            role=item["role"],
            indexable=item["indexable"],
            digest=item["digest"],
        )
        for item in payload["chunks"]
    )

    return FileAnalysis(
        path=path,
        language=payload["language"],
        source_hash=payload["source_hash"],
        definitions=definitions,
        references=references,
        chunks=chunks,
        stat=stat,
        recovered=payload["recovered"],
    )
