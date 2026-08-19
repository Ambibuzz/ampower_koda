"""What turns the ledger from "everything, by path" into "what matters now"."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..contracts.ledger import LedgerEntry
from ..retrieval.bm25 import LexicalIndex
from ..retrieval.tokenize import tokenize
from ..tokens import estimate_tokens


@dataclass(frozen=True, slots=True)
class Claim:
    """One entry and why it was claimed, so a trace can explain the block."""

    entry: LedgerEntry
    score: float
    tier: str
    """``pinned`` · ``scored`` · ``recent``."""


def rank_entries(
    entries: Sequence[LedgerEntry],
    message: str,
    index: LexicalIndex,
    *,
    soft_tokens: int,
) -> tuple[Claim, ...]:
    """Claim entries for one conversational turn, in the three tiers."""
    live = [entry for entry in entries if entry.is_live]
    scores = {entry.id: coverage(entry, message, index) for entry in live}

    claims: list[Claim] = [
        Claim(entry=entry, score=scores[entry.id], tier="pinned")
        for entry in live
        if entry.pinned
    ]
    spent = sum(_cost(claim.entry) for claim in claims)
    taken = {claim.entry.id for claim in claims}

    def claim(entry: LedgerEntry, tier: str) -> None:
        """Take ``entry`` if it fits, and skip it if it does not."""
        nonlocal spent
        cost = _cost(entry)
        if entry.id in taken or spent + cost > soft_tokens:
            return
        claims.append(Claim(entry=entry, score=scores[entry.id], tier=tier))
        taken.add(entry.id)
        spent += cost

    by_score = sorted(
        (entry for entry in live if scores[entry.id] > 0),
        key=lambda entry: (-scores[entry.id], entry.id),
    )
    for entry in by_score:
        claim(entry, "scored")

    for entry in reversed(live):
        claim(entry, "recent")

    return tuple(claims)


def coverage(entry: LedgerEntry, message: str, index: LexicalIndex) -> float:
    """The share of the message's IDF mass this entry accounts for."""
    terms = set(tokenize(message, is_query=True))
    if not terms:
        return 0.0

    total = sum(index.idf(term) for term in terms)
    if total <= 0:
        return 0.0

    body = " ".join([entry.text, *(ref.path for ref in entry.refs)])
    present = set(tokenize(body)) & terms
    return sum(index.idf(term) for term in present) / total


def _cost(entry: LedgerEntry) -> int:
    """What one entry costs rendered, near enough to spend a budget against."""
    refs = " ".join(ref.location for ref in entry.refs)
    return estimate_tokens(f"  {entry.id} [{entry.kind},{entry.confidence}] {entry.text}  ({refs})")
