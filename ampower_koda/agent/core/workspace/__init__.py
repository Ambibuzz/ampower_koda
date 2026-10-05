"""The boundary: what the core may read, and what it may never write."""

from __future__ import annotations

from .discovery import Discovery, decode_source, discover, excluded_directories, is_excluded_path
from .local import LocalWorkspace, SystemClock
from .ports import Clock, Workspace
from .redaction import RedactionMatcher, compile_redaction

__all__ = [
    "Clock",
    "Discovery",
    "LocalWorkspace",
    "RedactionMatcher",
    "SystemClock",
    "Workspace",
    "compile_redaction",
    "decode_source",
    "discover",
    "excluded_directories",
    "is_excluded_path",
]
