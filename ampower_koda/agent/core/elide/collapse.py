"""What a collapsed tool result leaves behind."""

from __future__ import annotations

import re

from ..constants import ROADMAP_MAX_COORDINATES
from ..contracts.transcript import Block
from ..ledger.render import stub

ROADMAP_TOOLS: frozenset[str] = frozenset({"search", "grep", "explore", "refs", "ast_search"})

READ_TOOLS: frozenset[str] = frozenset({"read", "view", "open"})

_COORDINATE = re.compile(
    r"^\s*([\w./-]+\.[A-Za-z0-9]{1,6}:\d+(?:-\d+)?(?:@\w+)?(?:\s*\[[\d.]+\])?)"
)

ELIDED = "[excerpt elided]"


def collapse(block: Block, *, evict_undistilled_reads: bool = False) -> Block:
    """Return ``block`` with its bytes replaced by the right kind of stub."""
    if not block.is_result or block.elided:
        return block

    if block.tool in READ_TOOLS:
        if not block.entry_id and not evict_undistilled_reads:
            return block
        return block.elided_to(stub("read", block.detail, block.entry_id))

    if block.tool in ROADMAP_TOOLS:
        return block.elided_to(roadmap(block))

    if block.entry_id:
        return block.elided_to(f"[{block.detail or block.tool} → {block.entry_id}]")
    return block.elided_to(f"[{block.tool} result elided]")


def roadmap(block: Block) -> str:
    """Coordinates only, excerpts replaced by a marker."""
    kept: list[str] = []
    for line in block.text.split("\n"):
        match = _COORDINATE.match(line)
        if not match:
            continue
        kept.append(f"{match.group(1).strip()} — {ELIDED}")
        if len(kept) >= ROADMAP_MAX_COORDINATES:
            break

    if not kept:
        return stub(block.tool, block.detail, block.entry_id)

    total = sum(1 for line in block.text.split("\n") if line.strip())
    dropped = max(0, total - len(kept))
    tail = f"\n[{dropped} more elided]" if dropped > 0 else ""
    handle = f" → {block.entry_id}" if block.entry_id else ""
    return f"[{block.tool} {block.detail}{handle}]\n" + "\n".join(kept) + tail
