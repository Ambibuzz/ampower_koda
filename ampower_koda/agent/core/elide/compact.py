"""Summarize an older prefix of the history when input pressure requires it."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..constants import COMPACTION_MAX_OUTPUT_TOKENS
from ..contracts.escalation import SideUsage
from ..contracts.model import UtilityModel
from ..contracts.transcript import Block, Transcript
from ..tokens import estimate_tokens, truncate_to_tokens

CLIFF_SYSTEM = (
    "You compact an engineering session so it can survive losing its transcript. "
    "You never restate content that is stored elsewhere."
)

SUMMARY_HEADER = "[compacted session summary]"


def compact_for_request(
    transcript: Transcript, *, summariser: UtilityModel | None,
    keep_tokens: int, protected_call_ids: frozenset[str] = frozenset(),
    max_input_tokens: int,
) -> Compaction:
    """Checkpoint an older balanced prefix, retaining requests and recent evidence."""
    if summariser is None:
        return Compaction(transcript=transcript)
    blocks = transcript.blocks
    cut, retained = len(blocks), 0
    while cut and retained + blocks[cut - 1].tokens <= keep_tokens:
        cut -= 1
        retained += blocks[cut].tokens
    for index, block in enumerate(blocks):
        if block.call_id in protected_call_ids:
            cut = min(cut, index)
    # A boundary may never separate the assistant's call from its result.
    tail_calls = {b.call_id for b in blocks[cut:] if b.is_result}
    for index, block in enumerate(blocks[:cut]):
        if block.kind == "tool_use" and block.call_id in tail_calls:
            cut = min(cut, index)
    head = blocks[:cut]
    if not any(b.role == "assistant" or b.is_result for b in head):
        return Compaction(transcript=transcript)
    prompt = (
        "Summarize completed activity for continued work. Preserve verified findings, "
        "paths, changes already made, test outcomes, unresolved questions and next steps. "
        "Do not claim unverified success. User requests will also be retained verbatim.\n\n"
        + json.dumps([{"role": b.role, "kind": b.kind, "tool": b.tool,
                       "arguments": b.arguments_json, "text": b.text} for b in head],
                     ensure_ascii=False)
    )
    if estimate_tokens(CLIFF_SYSTEM + prompt) + 32 > max_input_tokens:
        return Compaction(transcript=transcript, notices=("history is too large to summarize safely",))
    completion = summariser.complete(CLIFF_SYSTEM, prompt, max_tokens=COMPACTION_MAX_OUTPUT_TOKENS)
    if not completion.usable:
        return Compaction(transcript=transcript, usage=completion.usage,
                          notices=(completion.detail or "context summary unavailable",))
    summary = Block(role="assistant", text=SUMMARY_HEADER + "\n" +
                    truncate_to_tokens(completion.text.strip(), COMPACTION_MAX_OUTPUT_TOKENS))
    requests = [b for b in head if b.role == "user" and b.is_prose]
    candidate = transcript.with_blocks([*requests, summary, *blocks[cut:]])
    if candidate.tokens >= transcript.tokens:
        return Compaction(transcript=transcript, usage=completion.usage,
                          notices=("context summary did not reduce history",))
    return Compaction(transcript=candidate, usage=completion.usage, cliff=True,
                      notices=(f"compacted history: {transcript.tokens:,} → {candidate.tokens:,} tokens",))


@dataclass(frozen=True, slots=True)
class Compaction:
    """The transcript after summarizing, what it cost, and what happened."""

    transcript: Transcript
    usage: SideUsage = SideUsage()
    cliff: bool = False
    notices: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.cliff
