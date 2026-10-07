"""Redaction, applied at the source."""

from __future__ import annotations

import threading
from collections.abc import Sequence

from ..config.load import CONFIG_PATH, resolve_config
from ..config.schema import CoreConfig
from ..constants import DEFAULT_REDACT_GLOBS
from ..globs import GlobMatcher, compile_globs
from .ports import Workspace

RedactionMatcher = GlobMatcher
"""Takes a relative path; returns the pattern that redacts it, or ``None``."""


def compile_redaction(patterns: Sequence[str]) -> RedactionMatcher:
    """Compile redaction globs into a matcher."""
    return compile_globs(patterns)


def redaction_globs(config: CoreConfig) -> tuple[str, ...]:
    """Built-in patterns plus the configured ones: a config adds, never replaces."""
    return DEFAULT_REDACT_GLOBS + config.security.redact_globs


_REDACTION: dict[str, tuple[str, RedactionMatcher]] = {}
"""Root → (config file stamp, matcher). One entry per app root, so it stays small."""
_REDACTION_LOCK = threading.Lock()


def redaction_matcher(workspace: Workspace) -> RedactionMatcher:
    """The redaction discovery applies, for tools that read a file by path.

    Same defaults, same ``.koda/config.toml``: the index is not the only way a
    file reaches the model. Cached per root and config-file stamp, because
    every read asks and the file rarely changes.
    """
    stat = workspace.stat(CONFIG_PATH)
    stamp = stat.key() if stat is not None else ""
    with _REDACTION_LOCK:
        cached = _REDACTION.get(workspace.root)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    matcher = compile_redaction(redaction_globs(resolve_config(workspace, None, [])))
    with _REDACTION_LOCK:
        _REDACTION[workspace.root] = (stamp, matcher)
    return matcher
