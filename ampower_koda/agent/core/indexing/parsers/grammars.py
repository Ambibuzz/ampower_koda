"""Which grammars exist, where they come from, and what identifies them."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path

from ...identity import fingerprint

QUERY_DIRECTORY = Path(__file__).parent / "queries"

_LANGUAGE_PACK = "tree_sitter_language_pack"


@dataclass(frozen=True, slots=True)
class LanguageSpec:
    """Everything needed to build one parser."""

    name: str
    """Stable language name. Also the query filename stem and the pack key."""

    extensions: tuple[str, ...]
    """Lowercase, with the dot. Two specs may not claim the same extension."""

    module: str | None = None
    """Dedicated grammar module, e.g. ``tree_sitter_javascript``."""

    symbol: str = "language"
    """Function in that module returning the grammar. ``tree-sitter-typescript``
    ships two (``language_typescript`` and ``language_tsx``), which is why this
    is a field rather than a constant."""

    distribution: str | None = None
    """PyPI name, for reading the installed version. Defaults to ``module`` with
    underscores turned into hyphens."""

    query_name: str | None = None
    """Query file stem, when several specs share one. ``tsx`` reuses
    ``typescript-tags.scm`` because the differences do not affect tags."""

    injections: tuple[tuple[str, str], ...] = ()
    """``(capture_suffix, language_name)`` pairs. An HTML ``<script>`` body is
    one node of raw text to its own grammar; re-parsing that text with the
    JavaScript grammar is the only way its symbols are ever seen."""

    def query_path(self) -> Path:
        return QUERY_DIRECTORY / f"{self.query_name or self.name}-tags.scm"


LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec("python", (".py", ".pyi"), module="tree_sitter_python"),
    LanguageSpec("javascript", (".js", ".jsx", ".mjs", ".cjs"), module="tree_sitter_javascript"),
    LanguageSpec(
        "typescript",
        (".ts", ".mts", ".cts"),
        module="tree_sitter_typescript",
        symbol="language_typescript",
    ),
    LanguageSpec(
        "tsx",
        (".tsx",),
        module="tree_sitter_typescript",
        symbol="language_tsx",
        query_name="typescript",
    ),
    LanguageSpec("yaml", (".yaml", ".yml"), module="tree_sitter_yaml"),
    LanguageSpec("css", (".css",), module="tree_sitter_css"),
    LanguageSpec("scss", (".scss",), module="tree_sitter_scss"),
    LanguageSpec(
        "html",
        (".html", ".htm"),
        module="tree_sitter_html",
        injections=(("javascript", "javascript"), ("css", "css")),
    ),
    LanguageSpec(
        "vue",
        (".vue",),
        module=None,
        injections=(("javascript", "javascript"), ("css", "css")),
    ),
)

JSON_SPEC = LanguageSpec("json", (".json",), module="tree_sitter_json")


@dataclass(frozen=True, slots=True)
class Grammar:
    """A loaded grammar plus the string that identifies its exact version."""

    name: str
    language: object
    """A ``tree_sitter.Language``. Typed as ``object`` so this module imports
    without tree-sitter present, which is what lets the absence of the whole
    dependency be a degraded index rather than an ImportError at startup."""

    version: str


@dataclass(frozen=True, slots=True)
class GrammarLoad:
    """The result of trying to load every grammar in the table."""

    grammars: dict[str, Grammar] = field(default_factory=dict)
    missing: tuple[tuple[str, str], ...] = ()
    """``(language, reason)`` for each grammar that could not be loaded. Carried
    rather than logged, so cold start can report it once and the caller decides
    where it goes."""


def load_grammars(specs: Sequence[LanguageSpec] = LANGUAGES) -> GrammarLoad:
    """Load every grammar that is installed, and record why the others are not."""
    loaded: dict[str, Grammar] = {}
    missing: list[tuple[str, str]] = []

    for spec in specs:
        try:
            loaded[spec.name] = _load_one(spec)
        except Exception as exc:  # noqa: BLE001
            missing.append((spec.name, f"{type(exc).__name__}: {exc}"))

    return GrammarLoad(grammars=loaded, missing=tuple(missing))


def _load_one(spec: LanguageSpec) -> Grammar:
    """Load one grammar from its dedicated module, or from the language pack."""
    from tree_sitter import Language

    if spec.module:
        module = import_module(spec.module)
        capsule = getattr(module, spec.symbol)()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            language = Language(capsule)
        return Grammar(name=spec.name, language=language, version=_distribution_version(spec))

    pack = import_module(_LANGUAGE_PACK)
    return Grammar(
        name=spec.name,
        language=pack.get_language(spec.name),
        version=f"pack-{_version_of(_LANGUAGE_PACK.replace('_', '-'))}",
    )


def _distribution_version(spec: LanguageSpec) -> str:
    distribution = spec.distribution or (spec.module or "").replace("_", "-")
    return _version_of(distribution)


def _version_of(distribution: str) -> str:
    """The installed version, or a marker."""
    try:
        return importlib_metadata.version(distribution)
    except Exception:  # noqa: BLE001
        return "unknown"


def read_query(spec: LanguageSpec) -> str:
    """Read a language's tags query from disk."""
    return spec.query_path().read_text(encoding="utf-8")


def parser_identity(spec: LanguageSpec, grammar: Grammar, query_source: str) -> str:
    """The version tag folded into the chunk cache fingerprint."""
    return f"{spec.name}/{grammar.version}/{fingerprint(query_source, length=8)}"
