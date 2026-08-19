"""Where the cache boundaries go, and why they go there."""

from __future__ import annotations

from collections.abc import Sequence

from ..constants import MAX_TOTAL_BREAKPOINTS
from ..contracts.prompt import (
    CachePlan,
    Message,
    PromptBlock,
    PromptBudget,
    TranscriptMarker,
)
from ..contracts.repo_map import RepoMap
from ..contracts.session import RepoMemory
from ..tokens import estimate_tokens, truncate_to_tokens
from .models import cache_limits, routing_key


def build_prefix(
    repo_map: RepoMap,
    memory: RepoMemory,
    role_prompt: str,
    *,
    model: str,
    budget: PromptBudget | None = None,
) -> tuple[PromptBlock, ...]:
    """Assemble the two system blocks, in cache order."""
    budget = budget or PromptBudget()
    limits = cache_limits(model)

    blocks = [
        PromptBlock(role="map+memory", text=_map_and_memory(repo_map, memory, budget), ttl="5m"),
        PromptBlock(role="system+tools", text=role_prompt.strip(), ttl="5m"),
    ]

    allowed = max(0, MAX_TOTAL_BREAKPOINTS - budget.reserved_breakpoints)
    return tuple(_mark_breakpoints(blocks, min_cacheable=limits.min_cacheable, allowed=allowed))


def _map_and_memory(repo_map: RepoMap, memory: RepoMemory, budget: PromptBudget) -> str:
    """The first cached block: what the repository is, then what it asks of you."""
    sections: list[str] = []
    if not repo_map.is_empty:
        sections.append(truncate_to_tokens(repo_map.text, budget.map_tokens))
    if not memory.is_empty:
        sections.append(truncate_to_tokens(memory.text, budget.memory_tokens))
    return "\n\n".join(sections)


def _mark_breakpoints(
    blocks: Sequence[PromptBlock],
    *,
    min_cacheable: int,
    allowed: int,
) -> list[PromptBlock]:
    """Emit a boundary after each block once the prefix is wide enough to cache."""
    marked: list[PromptBlock] = []
    width = 0
    emitted = 0

    for block in blocks:
        width += estimate_tokens(block.text)
        wanted = bool(block.text.strip()) and width >= min_cacheable and emitted < allowed
        marked.append(
            PromptBlock(role=block.role, text=block.text, ttl=block.ttl, breakpoint=wanted)
        )
        emitted += 1 if wanted else 0

    return marked


def place_marker(
    transcript: Sequence[Message],
    *,
    model: str,
    prefix_tokens: int = 0,
) -> TranscriptMarker | None:
    """Pin the rolling boundary on the last plain-string message."""
    limits = cache_limits(model)
    width = prefix_tokens

    last_plain: int | None = None
    for index, message in enumerate(transcript):
        width += message.tokens
        if message.plain and width >= limits.min_cacheable:
            last_plain = index

    return TranscriptMarker(index=last_plain) if last_plain is not None else None


def assemble(
    *,
    blocks: Sequence[PromptBlock],
    transcript: Sequence[Message],
    tail: str,
    model: str,
    session_id: str = "",
    previous_marker: TranscriptMarker | None = None,
) -> CachePlan:
    """Build the whole plan."""
    prefix_tokens = sum(estimate_tokens(block.text) for block in blocks)
    marker = place_marker(transcript, model=model, prefix_tokens=prefix_tokens)

    return CachePlan(
        blocks=tuple(blocks),
        marker=marker,
        previous_marker=_keep_alive(previous_marker, marker, transcript),
        tail=tail,
        session_id=routing_key(session_id),
    )


def _keep_alive(
    previous: TranscriptMarker | None,
    current: TranscriptMarker | None,
    transcript: Sequence[Message],
) -> TranscriptMarker | None:
    """Whether the previous marker is still worth re-emitting."""
    if previous is None or current is None:
        return None
    if not previous.index < current.index < len(transcript) + 1:
        return None
    if previous.index >= len(transcript) or not transcript[previous.index].plain:
        return None
    return previous


def system_plan_mismatch(system_prompt: str, plan: CachePlan) -> str | None:
    """Detect a driver about to send two role prompts, or the wrong one."""
    wanted = system_prompt.strip()
    if not wanted:
        return None

    carried = plan.system_text
    if not carried:
        return "cache plan carries no system text while a system prompt was supplied"
    if wanted in carried:
        return None
    return (
        "system prompt is not the one in the cache plan — "
        "sending both would run the wrong role"
    )
