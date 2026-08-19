"""Cold start: building the world, once per session."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..budget.allocator import ContextBudget, allocate
from ..config.merge import merge_config, parse_toml
from ..config.schema import CoreConfig
from ..contracts.session import CoChangeMemory, RepoMemory, SessionContext
from ..contracts.source import Overlay
from ..errors import ConfigError, CoreError
from ..history.cochange import build_cochange, empty_memory, git_log_arguments, parse_git_log
from ..indexing.build import BuildStats, build_index
from ..indexing.incremental import apply_overlays
from ..indexing.parsers.registry import ParserRegistry, default_registry
from ..memory.repo_memory import read_repo_memory
from ..repomap.build import MapBuild, build_map
from ..retrieval.engine import Retriever, build_retriever
from ..workspace.discovery import discover
from ..workspace.local import SystemClock
from ..workspace.ports import Clock, Workspace

CONFIG_PATH = ".koda/config.toml"


@dataclass(frozen=True, slots=True)
class Bootstrap:
    """A built session context, plus what building it cost."""

    context: SessionContext
    stats: BuildStats

    budget: ContextBudget
    """Every ceiling this session runs under, derived from its window."""

    retriever: Retriever
    """The search engine, built once. Not on the context because a
    :class:`~ampower_koda.agent.core.contracts.session.SessionContext` is a
    contract — data with no behaviour — and a retriever holds a scored corpus
    and knows how to walk a graph."""

    ranking: MapBuild
    """The map's ranking machinery: the code graph, the unpersonalized file
    ranks, and the mirror set. Three consumers need these and all three are
    expensive; computing them once here is the difference between a few
    milliseconds of cold start and a few milliseconds per query."""

    notes: tuple[str, ...] = ()
    """Non-fatal things a developer would want to know: a config file that
    failed to parse, history that could not be read, memory that was truncated.
    Collected rather than logged, so the caller decides where they surface."""


def build_context(
    workspace: Workspace,
    *,
    overrides: dict | None = None,
    registry: ParserRegistry | None = None,
    overlays: tuple[Overlay, ...] = (),
    clock: Clock | None = None,
) -> Bootstrap:
    """Build everything a session needs before it can answer anything."""
    registry = registry or default_registry()
    clock = clock or SystemClock()
    notes: list[str] = []

    config = _resolve_config(workspace, overrides, notes)
    if registry.unavailable:
        notes.append(
            "not indexed by symbol: "
            + ", ".join(f"{language} ({reason})" for language, reason in registry.unavailable)
        )

    discovery = discover(workspace, config)
    build = build_index(workspace, config, registry=registry, discovery=discovery)
    if build.stats.recovered:
        notes.append(f"{build.stats.recovered} file(s) parsed with syntax errors")

    memory = _read_memory(workspace, config, notes)
    cochange = _read_cochange(workspace, config, clock, notes)

    context = apply_overlays(
        SessionContext(
            root=workspace.root,
            config=config,
            index=build.index,
            memory=memory,
            cochange=cochange,
        ),
        overlays,
        registry=registry,
    )

    ranking = build_map(context.index, max_tokens=config.context.map_tokens)
    context = replace(context, repo_map=ranking.map)

    if ranking.map.degraded:
        notes.append("repo map degraded to a directory tree: no parseable definitions")
    if ranking.mirrors.roots:
        notes.append("vendored copies demoted: " + ", ".join(sorted(ranking.mirrors.roots)))

    return Bootstrap(
        context=context,
        stats=build.stats,
        budget=allocate(
            config.context.window_tokens,
            ledger_override=config.context.ledger_soft_tokens,
            map_tokens=config.context.map_tokens,
            memory_tokens=config.context.memory_tokens,
        ),
        retriever=build_retriever(
            context.index,
            ranking.graph,
            ranking.ranks,
            mirrors=ranking.mirrors,
            cochange=cochange,
        ),
        ranking=ranking,
        notes=tuple(notes),
    )


def _resolve_config(
    workspace: Workspace,
    overrides: dict | None,
    notes: list[str],
) -> CoreConfig:
    """Resolve ``defaults < .koda/config.toml < overrides``."""
    from_file: dict | None = None
    if workspace.stat(CONFIG_PATH) is not None:
        try:
            from_file = parse_toml(workspace.read_bytes(CONFIG_PATH).decode("utf-8"))
        except (CoreError, UnicodeDecodeError) as exc:
            notes.append(f"{CONFIG_PATH} ignored: {exc}")

    try:
        return merge_config(from_file, overrides)
    except ConfigError:
        if from_file is None:
            raise
        notes.append(f"{CONFIG_PATH} ignored: contains an invalid value")
        return merge_config(overrides)


def _read_memory(workspace: Workspace, config: CoreConfig, notes: list[str]) -> RepoMemory:
    memory = read_repo_memory(workspace, max_tokens=config.context.memory_tokens)
    if memory.truncated:
        notes.append(
            f"repository memory truncated to {config.context.memory_tokens} tokens "
            f"({', '.join(memory.sources) or 'no file fit'})"
        )
    return memory


def _read_cochange(
    workspace: Workspace,
    config: CoreConfig,
    clock: Clock,
    notes: list[str],
) -> CoChangeMemory:
    if not config.history.enabled:
        return empty_memory()

    output = workspace.run_git(git_log_arguments(config.history))
    if output is None:
        notes.append("co-change memory unavailable: git log could not be read")
        return empty_memory()

    return build_cochange(parse_git_log(output), config.history, now=clock.now())
