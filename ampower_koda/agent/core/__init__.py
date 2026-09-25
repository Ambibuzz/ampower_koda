"""Koda's retrieval and context core."""

from __future__ import annotations

from .agent import ROLE_PROMPT, open_session, run_turn
from .budget import ContextBudget, TokenCalibrator, allocate
from .config import CoreConfig, config_defaults, merge_config
from .constants import CHARS_PER_TOKEN
from .context import Bootstrap, build_context
from .contracts import (
    BlobRef,
    CachePlan,
    Chunk,
    CoChangeMemory,
    Definition,
    FileAnalysis,
    FileRanks,
    Hit,
    Ledger,
    LedgerEntry,
    MirrorSet,
    ModelRequest,
    ModelTurn,
    Overlay,
    Reference,
    RepoMap,
    RepoMemory,
    RepositoryIndex,
    SearchResult,
    Session,
    SessionContext,
    SideUsage,
    SkippedFile,
    SourceFile,
    Span,
    ToolCall,
    ToolHost,
    ToolOutcome,
    Transcript,
    TurnResult,
    TurnUsage,
    UtilityModel,
    WorkingSet,
    iter_chunks,
)
from .elide import compact, hot_cold
from .errors import ConfigError, CoreError, ParseError, WorkspaceError
from .fold import SessionState, fold_turn
from .graph import CodeGraph, build_graph, pagerank
from .ledger import distil_into, record, record_read, rehydrate, render_ledger
from .loop import TurnMeters
from .prompt import assemble, build_prefix
from .repomap import MapBuild, build_map
from .retrieval import Retriever, build_retriever, search
from .tokens import estimate_tokens
from .tools import TOOL_NAMES
from .tools.run import NullHost, run_tool
from .workingset import working_set_for
from .workspace import LocalWorkspace, Workspace

__all__ = [
    "BlobRef",
    "Block",
    "Bootstrap",
    "CHARS_PER_TOKEN",
    "CachePlan",
    "ChatModel",
    "Chunk",
    "CoChangeMemory",
    "CodeGraph",
    "Completion",
    "ConfigError",
    "ContextBudget",
    "CoreConfig",
    "CoreError",
    "Definition",
    "FileAnalysis",
    "FileRanks",
    "Hit",
    "Ledger",
    "LedgerEntry",
    "LocalWorkspace",
    "MapBuild",
    "MirrorSet",
    "ModelRequest",
    "ModelTurn",
    "NullHost",
    "Overlay",
    "ParseError",
    "ROLE_PROMPT",
    "Reference",
    "RepoMap",
    "RepoMemory",
    "RepositoryIndex",
    "Retriever",
    "SearchResult",
    "Session",
    "SessionContext",
    "SessionState",
    "SideUsage",
    "SkippedFile",
    "SourceFile",
    "Span",
    "TOOL_NAMES",
    "TokenCalibrator",
    "ToolCall",
    "ToolHost",
    "ToolOutcome",
    "Transcript",
    "TurnMeters",
    "TurnResult",
    "TurnUsage",
    "UtilityModel",
    "WorkingSet",
    "Workspace",
    "WorkspaceError",
    "allocate",
    "assemble",
    "build_context",
    "build_graph",
    "build_map",
    "build_prefix",
    "build_retriever",
    "compact",
    "config_defaults",
    "distil_into",
    "estimate_tokens",
    "fold_turn",
    "hot_cold",
    "iter_chunks",
    "merge_config",
    "open_session",
    "pagerank",
    "record",
    "record_read",
    "rehydrate",
    "render_ledger",
    "run_tool",
    "run_turn",
    "search",
    "working_set_for",
]
