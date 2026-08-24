"""One formulation per question per session — failures included."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..contracts.escalation import Formulation, SideUsage

_SEPARATOR = "\0"


@dataclass(slots=True)
class FormulationCache:
    """Rung + question → the one formulation this session will ever get for it."""

    entries: dict[str, Formulation] = field(default_factory=dict)

    calls_saved: int = 0
    """Hits served from the cache. A free metric, and a real one: a session with
    a high number here is a session asking the same question repeatedly, which
    usually means the loop is not learning from the answer it already got."""

    def key(self, rung: str, question: str) -> str:
        """Rung and question, whitespace-normalised, joined on a NUL."""
        return f"{rung}{_SEPARATOR}{' '.join(question.split())}"

    def get(self, rung: str, question: str) -> Formulation | None:
        entry = self.entries.get(self.key(rung, question))
        if entry is not None:
            self.calls_saved += 1
        return entry

    def put(self, rung: str, question: str, formulation: Formulation) -> Formulation:
        self.entries[self.key(rung, question)] = formulation
        return formulation


@dataclass(frozen=True, slots=True)
class Formulated:
    """A formulation, and whether producing it actually cost anything."""

    value: Formulation
    fresh: bool

    @property
    def usage(self) -> SideUsage:
        """What to charge: the call's usage, or nothing at all."""
        return self.value.usage if self.fresh else SideUsage()


def formulate(
    cache: FormulationCache,
    rung: str,
    question: str,
    call: Callable[[], Formulation],
) -> Formulated:
    """Return the cached formulation for ``question``, or make the one call."""
    cached = cache.get(rung, question)
    if cached is not None:
        return Formulated(value=cached, fresh=False)

    try:
        result = call()
    except Exception as error:  # noqa: BLE001
        result = Formulation(failed=True, detail=f"{rung} raised {type(error).__name__}")
    return Formulated(value=cache.put(rung, question, result), fresh=True)
