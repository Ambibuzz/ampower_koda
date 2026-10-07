"""Exception types raised by the core."""

from __future__ import annotations


class CoreError(Exception):
    """Base class for every error the core raises deliberately."""


class ConfigError(CoreError):
    """A configuration value is missing, malformed, or out of range."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(f"config {key!r}: {reason}")
        self.key = key
        self.reason = reason


class WorkspaceError(CoreError):
    """The workspace port could not satisfy a read."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path!r}: {reason}")
        self.path = path
        self.reason = reason


class ParseError(CoreError):
    """A parser refused a source file."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path!r}: {reason}")
        self.path = path
        self.reason = reason
