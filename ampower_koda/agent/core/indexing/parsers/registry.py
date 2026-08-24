"""Choosing a parser for a file, and pinning what that choice implies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from ...errors import CoreError
from .frappe_json import FrappeJsonParser
from .grammars import JSON_SPEC, LANGUAGES, GrammarLoad, LanguageSpec, load_grammars
from .protocol import LanguageParser
from .treesitter import TreeSitterParser, build_parser

_INJECTION_TARGETS = ("javascript", "css")


@dataclass(frozen=True, slots=True)
class ParserRegistry:
    """An immutable extension → parser mapping."""

    by_extension: Mapping[str, LanguageParser]

    unavailable: tuple[tuple[str, str], ...] = ()
    """``(language, reason)`` for grammars that could not be loaded. Carried on
    the registry rather than logged where they happen, so cold start can report
    the whole picture once — "yaml and vue are not indexed on this bench" is a
    useful sentence; five scattered warnings are not."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_extension", MappingProxyType(dict(self.by_extension)))

    def for_path(self, path: str) -> LanguageParser | None:
        """Return the parser claiming this path, or ``None``."""
        _, dot, suffix = path.rpartition(".")
        if not dot:
            return None
        return self.by_extension.get(f".{suffix.lower()}")

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(sorted({parser.language for parser in self.by_extension.values()}))


def build_registry(
    parsers: Iterable[LanguageParser],
    *,
    unavailable: tuple[tuple[str, str], ...] = (),
) -> ParserRegistry:
    """Build a registry, refusing two parsers that claim the same extension."""
    table: dict[str, LanguageParser] = {}
    for parser in parsers:
        for extension in parser.extensions:
            key = extension.lower()
            existing = table.get(key)
            if existing is not None and existing is not parser:
                raise CoreError(
                    f"{key!r} is claimed by both {existing.language!r} and {parser.language!r}"
                )
            table[key] = parser
    return ParserRegistry(by_extension=table, unavailable=unavailable)


@lru_cache(maxsize=1)
def default_registry() -> ParserRegistry:
    """The registry this app ships with, built once per process."""
    return build_default_registry()


def build_default_registry(
    *,
    specs: Sequence[LanguageSpec] = LANGUAGES,
    load: GrammarLoad | None = None,
) -> ParserRegistry:
    """Build a registry from scratch, with no memo."""
    grammars = load if load is not None else load_grammars((*specs, JSON_SPEC))
    by_name = {spec.name: spec for spec in specs}

    built: dict[str, TreeSitterParser] = {}
    failed = list(grammars.missing)

    ordered = [name for name in _INJECTION_TARGETS if name in by_name]
    ordered += [spec.name for spec in specs if spec.name not in ordered]

    for name in ordered:
        grammar = grammars.grammars.get(name)
        if grammar is None:
            continue
        try:
            built[name] = build_parser(by_name[name], grammar, built=built)
        except Exception as exc:  # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))

    parsers: list[LanguageParser] = [built[spec.name] for spec in specs if spec.name in built]

    json_grammar = grammars.grammars.get("json")
    if json_grammar is not None:
        parsers.append(FrappeJsonParser(_grammar=json_grammar))

    return build_registry(parsers, unavailable=tuple(failed))


def registry_identity(registry: ParserRegistry) -> str:
    """A stable string naming every parser and version in the registry."""
    return "|".join(
        f"{extension}={registry.by_extension[extension].identity}"
        for extension in sorted(registry.by_extension)
    )
