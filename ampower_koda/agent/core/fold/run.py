"""One turn digested per turn, proactively, so the cliff stays unreachable."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..constants import (
    FOLD_ARGS_CLIP_CHARS,
    FOLD_MAX_OUTPUT_TOKENS,
    FOLD_RESULT_CLIP_CHARS,
    KEEP_RAW_TURNS,
)
from ..contracts.escalation import SideUsage
from ..contracts.model import UtilityModel
from ..contracts.transcript import Transcript
from ..tokens import estimate_tokens, truncate_to_tokens
from .document import DROPPABLE, SessionState, blank, parse, template

FOLD_SYSTEM = (
    "You maintain a running SESSION STATE document for an engineering session. "
    "You output the whole document and nothing else."
)

FOLD_RULES = (
    "Never restate file contents, tool output, or ledger entries. Cite ids.\n"
    "Anything in the current state that the latest turn did not change must be "
    "carried over unchanged. This document is the session's memory of itself.\n"
    "Keep the whole document under {max_tokens} tokens. When it would exceed "
    "that, drop the oldest resolved items, never the open ones."
)


@dataclass(frozen=True, slots=True)
class Fold:
    """The state after one fold attempt, and whether it advanced."""

    state: SessionState
    usage: SideUsage = SideUsage()
    folded: bool = False
    """True only when a turn was actually digested. This is what the trimmer
    reads through :attr:`SessionState.turns_folded`, and it is why a failure
    must not set it."""

    notes: tuple[str, ...] = ()


def fold_turn(
    transcript: Transcript,
    state: SessionState | None,
    model: UtilityModel,
    *,
    max_tokens: int,
) -> Fold:
    """Digest the next undigested turn into the session state."""
    current = state or blank()
    turn = _next_turn(transcript, current.turns_folded)
    if turn is None:
        return Fold(state=current, notes=("nothing to fold yet",))

    completion = model.complete(
        FOLD_SYSTEM,
        _prompt(current, turn, max_tokens=max_tokens),
        max_tokens=FOLD_MAX_OUTPUT_TOKENS,
    )
    if not completion.usable:
        return Fold(
            state=current,
            usage=completion.usage,
            notes=(completion.detail or "fold produced nothing",),
        )

    digested = parse(completion.text)
    if digested.is_empty:
        return Fold(state=current, usage=completion.usage, notes=("fold returned no sections",))

    return Fold(
        state=_fit(digested, current.turns_folded + 1, max_tokens),
        usage=completion.usage,
        folded=True,
    )


def _next_turn(transcript: Transcript, folded: int) -> str | None:
    """The rendered text of the next turn to digest, or ``None``."""
    starts = transcript.turn_starts
    local = transcript.local_turn(folded)
    if local < 0 or len(starts) <= KEEP_RAW_TURNS or local >= len(starts) - KEEP_RAW_TURNS:
        return None

    start = starts[local]
    end = starts[local + 1] if local + 1 < len(starts) else len(transcript.blocks)
    return _render(transcript, start, end)


def _render(transcript: Transcript, start: int, end: int) -> str:
    """One turn, one line per block, with the mechanics clipped hard."""
    lines: list[str] = []
    for block in transcript.blocks[start:end]:
        text = " ".join(block.text.split())
        if block.kind == "tool_use":
            detail = block.detail[:FOLD_ARGS_CLIP_CHARS]
            lines.append(f"{block.role} calls {block.tool}({detail})")
        elif block.is_result:
            lines.append(f"{block.tool} → {text[:FOLD_RESULT_CLIP_CHARS]}")
        elif text:
            lines.append(f"{block.role}: {text}")
    return "\n".join(lines)


def _prompt(state: SessionState, turn: str, *, max_tokens: int) -> str:
    """Current document, the rules, the shape, then the turn."""
    current = state.render() or template()
    return (
        f"CURRENT STATE:\n{current}\n\n"
        f"RULES:\n{FOLD_RULES.format(max_tokens=max_tokens)}\n\n"
        f"LATEST TURN:\n{turn}\n\n"
        "Output the updated SESSION STATE document."
    )


def _fit(state: SessionState, turns_folded: int, max_tokens: int) -> SessionState:
    """Enforce the budget the prompt asked for, oldest resolved items first."""
    sections = {name: list(items) for name, items in state.sections}
    order = [name for name, _ in state.sections]

    def rendered() -> str:
        return SessionState(
            sections=tuple((name, tuple(sections[name])) for name in order),
            turns_folded=turns_folded,
        ).render()

    while estimate_tokens(rendered()) > max_tokens:
        droppable = [name for name in order if name in DROPPABLE and sections[name]]
        if not droppable:
            break
        sections[droppable[0]].pop(0)

    final = SessionState(
        sections=tuple((name, tuple(sections[name])) for name in order),
        turns_folded=turns_folded,
    )
    if estimate_tokens(final.render()) <= max_tokens:
        return final

    trimmed = parse(truncate_to_tokens(final.render(), max_tokens))
    return replace(trimmed, turns_folded=turns_folded)
