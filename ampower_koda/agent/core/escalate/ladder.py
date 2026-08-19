"""One pure function that decides whether to spend a model call, and on what."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import EscalationConfig
from ..contracts.escalation import Rung


@dataclass(frozen=True, slots=True)
class Attempted:
    """Which rungs this question has already used up."""

    translate: bool = False
    expand: bool = False
    explore: bool = False

    def with_rung(self, rung: Rung) -> Attempted:
        """Return this record with ``rung`` marked spent."""
        if rung == "translate":
            return Attempted(True, self.expand, self.explore)
        if rung == "expand":
            return Attempted(self.translate, True, self.explore)
        if rung == "explore":
            return Attempted(self.translate, self.expand, True)
        return self


def decide(
    confidence: float,
    margin: float,
    attempted: Attempted,
    config: EscalationConfig,
    *,
    can_translate: bool = True,
) -> Rung:
    """The next rung to climb, or ``"none"``."""
    if not config.enabled or confidence >= config.confident:
        return "none"

    if confidence < config.weak:
        return _weak_band(attempted, config, can_translate=can_translate)

    if config.fan_out and margin < config.mid_margin and not attempted.expand:
        return "expand"
    return "none"


def _weak_band(attempted: Attempted, config: EscalationConfig, *, can_translate: bool) -> Rung:
    """Translate, then fan out, then explore — each at most once."""
    if can_translate and not attempted.translate:
        return "translate"
    if not config.fan_out:
        return "none"
    if not attempted.expand:
        return "expand"
    return "none" if attempted.explore else "explore"


def band(confidence: float, config: EscalationConfig) -> str:
    """``"confident"`` | ``"mid"`` | ``"weak"`` — for notes and for tests."""
    if confidence >= config.confident:
        return "confident"
    return "mid" if confidence >= config.weak else "weak"
