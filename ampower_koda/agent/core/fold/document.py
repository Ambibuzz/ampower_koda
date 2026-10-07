"""SESSION STATE — the session's memory of itself, in five sections."""

from __future__ import annotations

from dataclasses import dataclass

HEADER = "SESSION STATE"

SECTIONS: tuple[str, ...] = ("INTENT", "ANSWERED", "DECISIONS", "OPEN", "FILES")

DROPPABLE: frozenset[str] = frozenset({"ANSWERED", "DECISIONS", "FILES"})

_DESCRIPTIONS: dict[str, str] = {
    "INTENT": "what the developer is trying to achieve, one line",
    "ANSWERED": "one line per question already answered, citing ledger ids (L7)",
    "DECISIONS": "choices made and the reason, one line each",
    "OPEN": "questions raised and not yet resolved",
    "FILES": "path — the one-line conclusion reached about it",
}


@dataclass(frozen=True, slots=True)
class SessionState:
    """The document, parsed into its sections."""

    sections: tuple[tuple[str, tuple[str, ...]], ...] = ()
    turns_folded: int = 0

    @property
    def is_empty(self) -> bool:
        return not any(items for _, items in self.sections)

    def items(self, name: str) -> tuple[str, ...]:
        return next((items for section, items in self.sections if section == name), ())

    def render(self) -> str:
        """Header, then every section with a name, in declared order."""
        if self.is_empty:
            return ""
        lines = [HEADER]
        for name in SECTIONS:
            lines.append(f"{name}:")
            lines.extend(f"  {item}" for item in self.items(name))
        return "\n".join(lines)


def blank() -> SessionState:
    """An empty document with every section present, ready to be filled."""
    return SessionState(sections=tuple((name, ()) for name in SECTIONS))


def parse(text: str) -> SessionState:
    """Read a rendered document back."""
    names = SECTIONS
    collected: dict[str, list[str]] = {name: [] for name in names}
    current = ""

    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line == HEADER:
            continue
        heading = _heading(line, names)
        if heading:
            current = heading
            remainder = line[len(heading) + 1 :].strip()
            if remainder:
                collected[current].append(remainder)
            continue
        if current:
            collected[current].append(line.lstrip("-• ").strip())

    return SessionState(
        sections=tuple((name, tuple(collected[name])) for name in names)
    )


def _heading(line: str, names: tuple[str, ...]) -> str:
    upper = line.upper()
    return next((name for name in names if upper.startswith(f"{name}:")), "")


def template() -> str:
    """The empty document, with each section's description as its placeholder."""
    lines = [HEADER]
    for name in SECTIONS:
        lines.append(f"{name}: {_DESCRIPTIONS.get(name, '')}")
    return "\n".join(lines)
