"""Merging legs that are not equally believable."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..constants import FUSION_RANK_DECAY, RRF_K, SOURCE_LIMIT, UNION_LIMIT
from ..contracts.retrieval import LEG_TRUST, UNKNOWN_LEG_TRUST, Hit, LegResult


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One hit, with the prior that fusion assigned it."""

    hit: Hit
    prior: float


def leg_trust(leg: str) -> float:
    """How much a leg's own ordering is believed."""
    return LEG_TRUST.get(leg, UNKNOWN_LEG_TRUST)


def fuse(legs: Sequence[LegResult], *, limit: int = UNION_LIMIT) -> tuple[FusedHit, ...]:
    """Merge every leg's hits into one ranked candidate pool."""
    priors: dict[str, float] = {}
    merged: dict[str, Hit] = {}

    for result in legs:
        trust = leg_trust(result.leg)
        for rank, hit in enumerate(result.hits[:SOURCE_LIMIT]):
            digest = hit.chunk.digest
            prior = trust / (1.0 + FUSION_RANK_DECAY * rank)

            existing = merged.get(digest)
            merged[digest] = (existing or hit).claimed_by(result.leg, rank)
            priors[digest] = max(priors.get(digest, 0.0), prior)

    ordered = sorted(
        merged.values(),
        key=lambda hit: (-priors[hit.chunk.digest], hit.chunk.location),
    )
    return tuple(
        FusedHit(hit=hit.with_score(priors[hit.chunk.digest]), prior=priors[hit.chunk.digest])
        for hit in ordered[:limit]
    )


def reciprocal_rank_fusion(
    legs: Sequence[LegResult],
    *,
    limit: int,
    k: int = RRF_K,
) -> tuple[Hit, ...]:
    """Classic RRF. Kept for seed selection, and only for that."""
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}

    for result in legs:
        for rank, hit in enumerate(result.hits):
            digest = hit.chunk.digest
            scores[digest] = scores.get(digest, 0.0) + 1.0 / (k + rank + 1)
            hits.setdefault(digest, hit)

    ordered = sorted(hits.values(), key=lambda hit: (-scores[hit.chunk.digest], hit.chunk.location))
    return tuple(ordered[:limit])


def normalise(priors: Mapping[str, float]) -> dict[str, float]:
    """Min-max the priors into ``[0, 1]``."""
    if not priors:
        return {}
    values = list(priors.values())
    lowest, highest = min(values), max(values)
    if highest - lowest <= 0:
        return dict.fromkeys(priors, 1.0)
    return {digest: (value - lowest) / (highest - lowest) for digest, value in priors.items()}
