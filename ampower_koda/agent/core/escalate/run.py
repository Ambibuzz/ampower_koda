"""The loop that climbs the ladder, and everything it refuses to do."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from ..config.schema import EscalationConfig
from ..constants import DEFAULT_SEARCH_LIMIT, IDENTIFIER_MAX_SITES, MAX_SEARCH_LIMIT
from ..contracts.escalation import Escalation, Formulation, Formulator, Rung, SideUsage
from ..contracts.retrieval import Hit, SearchResult
from ..retrieval.engine import Retriever, search
from ..retrieval.tokenize import STOP_WORDS
from .ladder import Attempted, band, decide
from .memo import Formulated, FormulationCache, formulate
from .merge import added, annotate, round_robin
from .prompts import parse_fanout, parse_translation


@dataclass(frozen=True, slots=True)
class _Climb:
    """The state of one ascent. Every rung takes one and returns the next."""

    result: SearchResult
    attempted: Attempted = field(default_factory=Attempted)
    rungs: tuple[Rung, ...] = ()
    usage: SideUsage = SideUsage()
    readings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    done: bool = False
    """Set by a rung that cannot be followed by another — today only the
    unimplemented ``explore``. Distinct from "the ladder decided ``none``",
    which is the normal exit and is decided in the loop."""

    def spend(self, rung: Rung) -> _Climb:
        return replace(self, attempted=self.attempted.with_rung(rung), rungs=(*self.rungs, rung))

    def paid(self, formulated: Formulated) -> _Climb:
        """Charge a formulation — nothing at all when the memo served it."""
        return replace(self, usage=self.usage.plus(formulated.usage))

    def reading(self, result: SearchResult, substitution: str) -> _Climb:
        """Adopt a rung's result and record the substitution it was read under."""
        return replace(self, result=result, readings=(*self.readings, substitution))

    def noting(self, *notes: str) -> _Climb:
        return replace(self, notes=(*self.notes, *notes))

    def halted(self, note: str) -> _Climb:
        return replace(self, notes=(*self.notes, note), done=True)


def escalated_search(
    retriever: Retriever,
    question: str,
    *,
    formulator: Formulator | None = None,
    repo_map: str = "",
    config: EscalationConfig | None = None,
    cache: FormulationCache | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> Escalation:
    """Search, and climb the ladder if the answer does not stand on its own."""
    settings = config or EscalationConfig()
    settings.validate()
    memo = cache if cache is not None else FormulationCache()
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))

    original = search(retriever, question, limit=limit)
    climb = _Climb(result=original)
    grounded = formulator is not None and bool(repo_map.strip())

    while grounded and not climb.done:
        rung = decide(
            climb.result.confidence,
            climb.result.margin,
            climb.attempted,
            settings,
            can_translate=True,
        )
        if rung == "none":
            break
        climb = _climb(
            rung,
            climb.spend(rung),
            retriever=retriever,
            question=question,
            formulator=formulator,  # type: ignore[arg-type]
            repo_map=repo_map,
            config=settings,
            memo=memo,
            limit=limit,
        )

    ungrounded = () if grounded else ("no formulator or repo map; ladder not climbed",)
    return Escalation(
        result=climb.result,
        rungs=climb.rungs,
        usage=climb.usage,
        readings=climb.readings,
        original_confidence=original.confidence,
        notes=(
            *original.notes,
            *climb.notes,
            *ungrounded,
            f"band: {band(original.confidence, settings)}",
        ),
    )


def _climb(
    rung: Rung,
    climb: _Climb,
    *,
    retriever: Retriever,
    question: str,
    formulator: Formulator,
    repo_map: str,
    config: EscalationConfig,
    memo: FormulationCache,
    limit: int,
) -> _Climb:
    """Dispatch one rung. The rung is already marked spent by the caller."""
    if rung == "explore":
        return climb.halted("explore rung is decided but not implemented")

    if rung == "translate":
        return _translate(
            climb,
            retriever=retriever,
            question=question,
            formulator=formulator,
            repo_map=repo_map,
            memo=memo,
            limit=limit,
        )

    return _fan_out(
        climb,
        retriever=retriever,
        question=question,
        formulator=formulator,
        repo_map=repo_map,
        config=config,
        memo=memo,
        limit=limit,
    )


def _translate(
    climb: _Climb,
    *,
    retriever: Retriever,
    question: str,
    formulator: Formulator,
    repo_map: str,
    memo: FormulationCache,
    limit: int,
) -> _Climb:
    """Question → identifiers → one more search, on the identifiers alone."""
    formulated = formulate(
        memo, "translate", question, lambda: formulator.translate(question, repo_map)
    )
    climb = climb.paid(formulated)

    names = _grounded_names(retriever, formulated.value)
    if not names:
        return climb.noting(
            formulated.value.detail or "translation produced no name this repository defines"
        )

    reading = " ".join(names)
    retried = search(retriever, reading, limit=limit)
    if retried.is_empty:
        return climb.noting(f"translation '{reading}' retrieved nothing")

    return _adopt(climb, [retried.hits], reading, limit=limit, label=f"translated to '{reading}'")


def _fan_out(
    climb: _Climb,
    *,
    retriever: Retriever,
    question: str,
    formulator: Formulator,
    repo_map: str,
    config: EscalationConfig,
    memo: FormulationCache,
    limit: int,
) -> _Climb:
    """Question → ≤4 rewrites → one search each → round-robin merge."""
    formulated = formulate(
        memo, "formulate", question, lambda: formulator.fan_out(question, repo_map)
    )
    climb = climb.paid(formulated)

    rewrites = parse_fanout(_text_of(formulated.value), question, limit=config.max_rewrites)
    if not rewrites:
        return climb.noting(formulated.value.detail or "fan-out produced no usable rewrite")

    alternatives: list[Sequence[Hit]] = []
    for rewrite in rewrites:
        found = search(retriever, rewrite.text, limit=limit)
        if found.hits:
            alternatives.append(found.hits)

    if not alternatives:
        return climb.noting("every rewrite searched empty")

    reading = " | ".join(rewrite.text for rewrite in rewrites)
    return _adopt(
        climb,
        alternatives,
        reading,
        limit=limit,
        label=f"fanned out {len(alternatives)} of {len(rewrites)} rewrites",
    )


def _adopt(
    climb: _Climb,
    alternatives: Sequence[Sequence[Hit]],
    reading: str,
    *,
    limit: int,
    label: str,
) -> _Climb:
    """Merge a rung's hits into the question's, and claim only what it added."""
    merged = round_robin(climb.result.hits, alternatives, limit=limit)
    fresh = added(merged, climb.result.hits)
    note = f"{label}, {fresh} new"
    if not fresh:
        return climb.noting(note)

    return climb.reading(
        _with_hits(climb.result, annotate(merged, reading, already_found=climb.result.hits)),
        reading,
    ).noting(note)


def _grounded_names(retriever: Retriever, formulation: Formulation) -> tuple[str, ...]:
    """The translation's names, filtered to ones this index actually defines."""
    sites: dict[str, int] = {}
    for document in retriever.lexical.documents:
        identity = document.chunk.identity
        if not identity:
            continue
        for name in {identity, identity.rsplit(".", 1)[-1]}:
            sites[name] = sites.get(name, 0) + 1

    vocabulary = {
        name
        for name, count in sites.items()
        if count <= IDENTIFIER_MAX_SITES and name.lower() not in STOP_WORDS
    }
    return parse_translation(_text_of(formulation), vocabulary)


def _text_of(formulation: Formulation) -> str:
    """A formulation's rewrites as one block of text, for the parsers."""
    return "\n".join(rewrite.text for rewrite in formulation.rewrites)


def _with_hits(result: SearchResult, hits: tuple[Hit, ...]) -> SearchResult:
    return replace(result, hits=hits)
