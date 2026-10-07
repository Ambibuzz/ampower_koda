"""Data shapes only. No I/O, no policy, no behaviour beyond validation."""

from __future__ import annotations

from .agent import (
    ChatModel,
    ModelRequest,
    ModelTurn,
    Session,
    ToolCall,
    ToolHost,
    ToolOutcome,
    TurnResult,
    TurnUsage,
)
from .analysis import FileAnalysis, SkippedFile, SkipReason
from .chunks import Chunk, ChunkKind, is_indexable_role
from .escalation import SideUsage
from .ledger import (
    LEDGER_KINDS,
    BlobRef,
    Confidence,
    Ledger,
    LedgerEntry,
    LedgerKind,
    LedgerSource,
)
from .model import Completion, UtilityModel
from .prompt import (
    CachePlan,
    CacheTtl,
    Message,
    ModelCacheLimits,
    PromptBlock,
    PromptBudget,
    TranscriptMarker,
)
from .mirrors import MirrorSet
from .repository import (
    RepositoryIndex,
    definitions_by_name,
    files_referencing,
    iter_chunks,
    iter_definitions,
    with_file,
)
from .retrieval import LEG_TRUST, UNKNOWN_LEG_TRUST, Hit, LegResult, SearchResult
from .session import CoChangeMemory, RepoMemory, SessionContext
from .source import FileStat, Overlay, SourceFile, Span
from .symbols import (
    REFERENCE_KINDS,
    SYMBOL_ROLES,
    Definition,
    DefinitionSite,
    ParseResult,
    Reference,
    ReferenceKind,
    SymbolRole,
)
from .transcript import Block, BlockKind, Transcript
from .working_set import WorkingSet, WorkingSpan

__all__ = [
    "BlobRef",
    "Block",
    "BlockKind",
    "CachePlan",
    "CacheTtl",
    "ChatModel",
    "Chunk",
    "ChunkKind",
    "CoChangeMemory",
    "Completion",
    "Confidence",
    "Definition",
    "DefinitionSite",
    "FileAnalysis",
    "FileStat",
    "Hit",
    "LEDGER_KINDS",
    "LEG_TRUST",
    "Ledger",
    "LedgerEntry",
    "LedgerKind",
    "LedgerSource",
    "LegResult",
    "Message",
    "MirrorSet",
    "ModelCacheLimits",
    "ModelRequest",
    "ModelTurn",
    "Overlay",
    "ParseResult",
    "PromptBlock",
    "PromptBudget",
    "REFERENCE_KINDS",
    "Reference",
    "ReferenceKind",
    "RepoMemory",
    "RepositoryIndex",
    "SYMBOL_ROLES",
    "SearchResult",
    "Session",
    "SessionContext",
    "SideUsage",
    "SkipReason",
    "SkippedFile",
    "SourceFile",
    "Span",
    "SymbolRole",
    "ToolCall",
    "ToolHost",
    "ToolOutcome",
    "Transcript",
    "TranscriptMarker",
    "TurnResult",
    "TurnUsage",
    "UNKNOWN_LEG_TRUST",
    "UtilityModel",
    "WorkingSet",
    "WorkingSpan",
    "definitions_by_name",
    "files_referencing",
    "is_indexable_role",
    "iter_chunks",
    "iter_definitions",
    "with_file",
]
