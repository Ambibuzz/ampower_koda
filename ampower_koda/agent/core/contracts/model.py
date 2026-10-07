"""The one seam through which this package can cause a model call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .escalation import SideUsage


@dataclass(frozen=True, slots=True)
class Completion:
    """What one utility call produced, and whether it worked."""

    text: str = ""
    usage: SideUsage = SideUsage()
    failed: bool = False
    detail: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def usable(self) -> bool:
        return not self.failed and not self.is_empty


@runtime_checkable
class UtilityModel(Protocol):
    """One call, one string back. No streaming, no tools, no conversation."""

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        """Return the model's text. Must not raise; see the module docstring."""
