"""Building the repository index, once per session."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import CoreConfig
from ..contracts.analysis import FileAnalysis, SkippedFile
from ..contracts.repository import RepositoryIndex
from ..contracts.source import SourceFile
from ..errors import ParseError
from ..workspace.discovery import Discovery, discover
from ..workspace.ports import Workspace
from . import cache
from .analysis import analyze
from .parsers.registry import ParserRegistry, default_registry


@dataclass(frozen=True, slots=True)
class BuildStats:
    """What the build did, for the log line that follows it."""

    discovered: int = 0
    parsed: int = 0
    cached: int = 0
    rewritten: int = 0
    skipped: int = 0

    recovered: int = 0
    """Files that parsed with syntax errors the grammar recovered from. Their
    symbols are in the index and are a weaker claim than the rest — which is
    worth a number, because "the agent misread that file" and "that file does
    not compile" look identical from the outside."""

    def summary(self) -> str:
        return (
            f"{self.discovered} files: {self.parsed} parsed, {self.cached} cached"
            f"{f', {self.rewritten} stat-refreshed' if self.rewritten else ''}"
            f"{f', {self.recovered} with syntax errors' if self.recovered else ''}"
            f"{f', {self.skipped} skipped' if self.skipped else ''}"
        )


@dataclass(frozen=True, slots=True)
class BuildResult:
    index: RepositoryIndex
    stats: BuildStats


def build_index(
    workspace: Workspace,
    config: CoreConfig,
    *,
    registry: ParserRegistry | None = None,
    discovery: Discovery | None = None,
) -> BuildResult:
    """Index the workspace."""
    registry = registry or default_registry()
    found = discovery if discovery is not None else discover(workspace, config)
    fingerprint = cache.schema_fingerprint(registry)

    files: dict[str, FileAnalysis] = {}
    skipped: dict[str, SkippedFile] = {skip.path: skip for skip in found.skipped}
    parsed = cached = rewritten = recovered = 0

    for source in found.candidates:
        outcome = _analyse_one(workspace, source, config, registry, fingerprint)
        if outcome.analysis is None:
            skipped[source.path] = SkippedFile(source.path, "parse_failed", outcome.reason)
            continue

        files[source.path] = outcome.analysis
        parsed += 0 if outcome.from_cache else 1
        cached += 1 if outcome.from_cache else 0
        rewritten += 1 if outcome.stat_refreshed else 0
        recovered += 1 if outcome.analysis.recovered else 0

    return BuildResult(
        index=RepositoryIndex(files=files, skipped=skipped, fingerprint=fingerprint),
        stats=BuildStats(
            discovered=len(found.candidates),
            parsed=parsed,
            cached=cached,
            rewritten=rewritten,
            recovered=recovered,
            skipped=len(skipped),
        ),
    )


@dataclass(frozen=True, slots=True)
class _Outcome:
    """One file's result, plus how it was obtained."""

    analysis: FileAnalysis | None
    from_cache: bool = False
    stat_refreshed: bool = False
    reason: str = ""


def _analyse_one(
    workspace: Workspace,
    source: SourceFile,
    config: CoreConfig,
    registry: ParserRegistry,
    fingerprint: str,
) -> _Outcome:
    if config.indexing.use_cache:
        hit, needs_rewrite = cache.load(
            workspace,
            source.path,
            fingerprint=fingerprint,
            source_hash=source.source_hash,
            stat=source.stat,
        )
        if hit is not None:
            if needs_rewrite:
                cache.store(workspace, hit, fingerprint=fingerprint)
            return _Outcome(analysis=hit, from_cache=True, stat_refreshed=needs_rewrite)

    try:
        analysis = analyze(source, registry)
    except ParseError as exc:
        return _Outcome(analysis=None, reason=exc.reason)

    if config.indexing.use_cache:
        cache.store(workspace, analysis, fingerprint=fingerprint)
    return _Outcome(analysis=analysis)
