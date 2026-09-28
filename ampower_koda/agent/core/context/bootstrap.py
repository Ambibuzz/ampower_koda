"""Cold start: building the world, once per session."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.load import CONFIG_PATH, resolve_config  # noqa: F401 - CONFIG_PATH re-exported
from ..config.schema import CoreConfig
from ..contracts.session import CoChangeMemory, RepoMemory, SessionContext
from ..contracts.source import Overlay
from ..history.cochange import build_cochange, empty_memory, git_log_arguments, parse_git_log
from ..indexing.build import build_index
from ..indexing.incremental import apply_overlays
from ..indexing.parsers.registry import ParserRegistry, default_registry
from ..memory.repo_memory import read_repo_memory
from ..graph import build_graph, detect_mirrors
from ..retrieval.engine import Retriever, build_retriever
from ..workspace.discovery import discover
from ..workspace.local import SystemClock
from ..workspace.ports import Clock, Workspace


@dataclass(frozen=True, slots=True)
class Bootstrap:
    """A built session context and its retriever."""

    context: SessionContext

    retriever: Retriever
    """The search engine, built once. Not on the context because a
    :class:`~ampower_koda.agent.core.contracts.session.SessionContext` is a
    contract — data with no behaviour — and a retriever holds a scored corpus
    and knows how to walk the code graph, which is built once per cold start."""

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

    config = resolve_config(workspace, overrides, notes)
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

    graph = build_graph(context.index)
    mirrors = detect_mirrors(context.index.paths)
    if mirrors.roots:
        notes.append("vendored copies demoted: " + ", ".join(sorted(mirrors.roots)))

    return Bootstrap(
        context=context,
        retriever=build_retriever(
            context.index,
            graph,
            mirrors=mirrors,
            cochange=cochange,
            config=config.retrieval,
            rerank_config=config.rerank,
        ),
        notes=tuple(notes),
    )


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
