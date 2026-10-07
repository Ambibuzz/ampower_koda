"""Turning a repository into a searchable index."""

from __future__ import annotations

from .analysis import PLAIN_TEXT, analyze
from .build import BuildResult, BuildStats, build_index
from .cache import schema_fingerprint
from .chunking import build_chunks
from .containers import resolve_definitions
from .incremental import apply_overlays
from .parsers import (
    LANGUAGES,
    FrappeJsonParser,
    LanguageParser,
    LanguageSpec,
    ParserRegistry,
    TreeSitterParser,
    build_default_registry,
    build_registry,
    default_registry,
    load_grammars,
    registry_identity,
)

__all__ = [
    "PLAIN_TEXT",
    "BuildResult",
    "BuildStats",
    "LANGUAGES",
    "FrappeJsonParser",
    "LanguageParser",
    "LanguageSpec",
    "ParserRegistry",
    "TreeSitterParser",
    "analyze",
    "apply_overlays",
    "build_chunks",
    "build_index",
    "build_default_registry",
    "build_registry",
    "default_registry",
    "load_grammars",
    "registry_identity",
    "resolve_definitions",
    "schema_fingerprint",
]
