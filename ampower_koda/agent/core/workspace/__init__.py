"""The boundary: what the core may read, and what it may never write."""

from __future__ import annotations

from .discovery import Discovery, decode_source, discover, excluded_directories, is_excluded_path
from .local import FixedClock, LocalWorkspace, MemoryWorkspace, SystemClock, text_workspace
from .ports import Clock, Workspace
from .redaction import RedactionMatcher, compile_redaction, partition, refuse_if_redacted

__all__ = [
    "Clock",
    "Discovery",
    "FixedClock",
    "LocalWorkspace",
    "MemoryWorkspace",
    "RedactionMatcher",
    "SystemClock",
    "Workspace",
    "compile_redaction",
    "decode_source",
    "discover",
    "excluded_directories",
    "is_excluded_path",
    "partition",
    "refuse_if_redacted",
    "text_workspace",
]
