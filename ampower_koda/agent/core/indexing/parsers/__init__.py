"""Language backends: one protocol, one registry, nine grammars, one exception."""

from __future__ import annotations

from .frappe_json import FrappeJsonParser
from .grammars import JSON_SPEC, LANGUAGES, Grammar, GrammarLoad, LanguageSpec, load_grammars
from .protocol import LanguageParser
from .registry import (
    ParserRegistry,
    build_default_registry,
    build_registry,
    default_registry,
    registry_identity,
)
from .treesitter import TreeSitterParser, build_parser

__all__ = [
    "JSON_SPEC",
    "LANGUAGES",
    "FrappeJsonParser",
    "Grammar",
    "GrammarLoad",
    "LanguageParser",
    "LanguageSpec",
    "ParserRegistry",
    "TreeSitterParser",
    "build_default_registry",
    "build_parser",
    "build_registry",
    "default_registry",
    "load_grammars",
    "registry_identity",
]
