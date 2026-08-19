"""One function. Everything §1–§17 does, in the order a turn does it."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .budget.allocator import allocate
from .constants import DEFAULT_ARCHITECT_MODEL, MAX_OUTPUT_TOKENS, MAX_ROUNDS
from .context.bootstrap import build_context
from .contracts.agent import (
    ChatModel,
    ModelRequest,
    ModelTurn,
    Session,
    ToolCall,
    ToolHost,
    TurnResult,
    TurnUsage,
)
from .contracts.escalation import Formulator, SideUsage
from .contracts.ledger import Ledger
from .contracts.model import UtilityModel
from .contracts.prompt import PromptBudget
from .contracts.transcript import Block, Transcript
from .contracts.working_set import WorkingSet
from .elide.collapse import READ_TOOLS
from .elide.compact import ThrashGuard, compact
from .elide.hotcold import hot_cold
from .escalate.memo import FormulationCache
from .fold.document import SessionState
from .fold.run import fold_turn
from .ledger.distill import distil_into
from .ledger.recall import ref_for
from .ledger.render import render_ledger
from .ledger.write import record_read
from .loop import dedupe, gates, leaks
from .prompt.cache import assemble, build_prefix
from .tools.catalogue import TOOL_NAMES
from .tools.run import BUILT_IN, NullHost, run_tool
from .workingset.build import working_set_for
from .workspace.ports import Workspace

COVERAGE_CENTRAL_FILES = 5

ROLE_PROMPT = """You are a software engineer working inside one repository.

Answer from what you have actually read. When you have not read something, say
so rather than inferring it — a confident guess costs more to undo than a short
answer costs to extend.

Prefer outline over read. Prefer one search over three greps. Cite what you
found as path:line so it can be checked."""


def open_session(
    workspace: Workspace,
    *,
    overrides: dict | None = None,
    model: str = DEFAULT_ARCHITECT_MODEL,
) -> Session:
    """Cold start. Expensive, and it happens exactly once per conversation."""
    bootstrap = build_context(workspace, overrides=overrides)
    context = bootstrap.context
    return Session(
        workspace=workspace,
        model_id=model,
        context=context,
        retriever=bootstrap.retriever,
        ranks=bootstrap.ranking.ranks,
        repo_map=context.repo_map,
        budget=allocate(
            context.config.context.window_tokens,
            ledger_override=context.config.context.ledger_soft_tokens,
            map_tokens=context.config.context.map_tokens,
            memory_tokens=context.config.context.memory_tokens,
        ),
        guard=ThrashGuard(),
        notes=bootstrap.notes,
    )


def run_turn(
    question: str,
    *,
    workspace: Workspace | None = None,
    session: Session | None = None,
    model: ChatModel,
    host: ToolHost | None = None,
    utility: UtilityModel | None = None,
    formulator: Formulator | None = None,
    role_prompt: str = ROLE_PROMPT,
    model_id: str = DEFAULT_ARCHITECT_MODEL,
    overrides: dict | None = None,
    retrieval_query: str = "",
    max_rounds: int = MAX_ROUNDS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> TurnResult:
    """Run one turn end to end and return the answer plus the next session."""
    if session is None:
        if workspace is None:
            raise ValueError("run_turn needs either a workspace to open or a session to continue")
        session = open_session(workspace, overrides=overrides, model=model_id)
    model_id = session.model_id

    working = working_set_for(
        (retrieval_query or question).strip(),
        session.retriever,  # type: ignore[arg-type]
        ledger=session.ledger,
        edited=session.edited,
        max_tokens=session.budget.working_set,
    )

    state = _State(
        session=session,
        transcript=_append(session.transcript, "user", question),
        ledger=session.ledger,
        meters=gates.TurnMeters(
            max_rounds=max(1, int(max_rounds)),
            max_observed=session.budget.observed_turn,
        ),
    )

    outcome = _rounds(
        state,
        working=working,
        model=model,
        model_id=model_id,
        host=host or NullHost(),
        formulator=formulator,
        role_prompt=role_prompt,
        max_output_tokens=max(1, int(max_output_tokens)),
    )

    folded, side, notes = _close(outcome, utility)

    return TurnResult(
        answer=outcome.answer,
        session=folded,
        rounds=outcome.meters.round_index,
        usage=outcome.usage,
        side_usage=outcome.side.plus(side),
        calls=outcome.calls,
        stop_reason=outcome.stop_reason,
        working_set=working,
        notes=(*session.notes, *outcome.notes, *notes),
    )


class _State:
    """The loop's working set. Mutable, and scoped to one call of ``run_turn``."""

    __slots__ = (
        "answer", "calls", "ledger", "memo", "meters", "notes", "seen",
        "session", "side", "stop_reason", "transcript", "usage",
    )

    def __init__(self, session: Session, transcript: Transcript, ledger: Ledger,
                 meters: gates.TurnMeters) -> None:
        self.session = session
        self.transcript = transcript
        self.ledger = ledger
        self.meters = meters
        self.memo = dedupe.Memo()
        self.seen: set[str] = set()
        self.usage = TurnUsage()
        self.side = SideUsage()
        self.answer = ""
        self.calls: tuple[str, ...] = ()
        self.notes: tuple[str, ...] = ()
        self.stop_reason = "answered"


def _rounds(state, *, working, model, model_id, host, formulator, role_prompt,
            max_output_tokens):  # noqa: ANN001, PLR0913
    """The loop. One model call per iteration, tools in emission order."""
    forced = False
    pending_nudge = None
    last_text = ""

    for _ in range(state.meters.max_rounds):
        decision = gates.check(state.meters)
        if decision.nudge is not None and not forced:
            state.transcript = _append(state.transcript, "user", decision.nudge.text)
            state.notes = (*state.notes, f"nudge: {decision.reason}")
            forced = decision.force_terminal
        if pending_nudge is not None:
            state.transcript = _append(state.transcript, "user", pending_nudge)
            pending_nudge = None

        turn = model.respond(
            ModelRequest(
                plan=_plan(state, working=working, model_id=model_id, role=role_prompt),
                transcript=state.transcript,
                tools=TOOL_NAMES,
                max_tokens=max_output_tokens,
                force_terminal=forced,
            )
        )
        state.usage = state.usage.plus(turn.usage)
        state.meters = state.meters.charged(
            processed=turn.usage.processed, observed=turn.usage.observed
        )

        if turn.failed:
            state.stop_reason = "error"
            state.notes = (*state.notes, turn.detail or "the model call failed")
            state.answer = turn.text
            return state

        text, leaked = _degleak(turn, state)
        if text:
            state.transcript = _append(state.transcript, "assistant", text)
            if text is not leaks.CORRECTION:
                last_text = text

        if turn.stopped_at_limit:
            limit = gates.after_max_tokens(state.meters)
            state.meters = replace(state.meters, continuations=state.meters.continuations + 1)
            if limit.stop:
                state.answer = _salvage(state, text or last_text, limit.reason)
                state.stop_reason = "cut off"
                return state
            pending_nudge = limit.nudge.text if limit.nudge else None
            state.meters = _advance(state.meters)
            continue

        calls = turn.calls or leaked
        if not calls:
            nudge = _coverage(state)
            if nudge is not None:
                state.transcript = _append(state.transcript, "user", nudge.text)
                state.notes = (*state.notes, "coverage nudge")
                state.meters = _advance(state.meters)
                continue
            state.answer = text or _salvage(state, "", "the model returned no text")
            return state

        if forced:
            late = gates.late_tool_call()
            state.answer = _salvage(state, last_text, late.reason)
            state.stop_reason = late.reason
            return state

        novel = _dispatch(state, calls, host=host, formulator=formulator)
        state.meters = state.meters.next_round(dry=gates.is_dry(novel, max(novel, 1)))
        _elide(state)

    state.stop_reason = "rounds exhausted"
    state.answer = _salvage(state, last_text, "rounds exhausted")
    return state


def _dispatch(state, calls: Sequence[ToolCall], *, host, formulator) -> int:  # noqa: ANN001, ARG001
    """Run every call in emission order, distilling each result as it lands."""
    novel = 0
    for call in calls:
        state.calls = (*state.calls, call.tool)
        state.transcript = _append(
            state.transcript, "assistant", "", kind="tool_use",
            tool=call.tool, call_id=call.id, detail=call.detail,
            arguments_json=_json(call.arguments),
        )

        suppressed = state.memo.check_call(call.tool, call.arguments)
        if suppressed is not None:
            state.transcript = _append(
                state.transcript, "user", suppressed.text, kind="tool_result",
                tool=call.tool, call_id=call.id, detail=call.detail,
            )
            continue

        outcome = run_tool(
            call.tool, call.arguments,
            retriever=state.session.retriever, workspace=_workspace(state),
            ledger=state.ledger, host=host,
        )
        state.memo.record_call(call.tool, call.arguments, len(state.transcript.blocks))

        duplicate = state.memo.check_result(call.tool, outcome.text)
        text = duplicate.text if duplicate is not None else outcome.text
        if duplicate is None:
            state.memo.record_result(outcome.text, len(state.transcript.blocks))
            novel += gates.evidence_yield(outcome.text, state.seen)

        state.ledger, entry_id = _record(state, call, outcome, text)
        state.transcript = _append(
            state.transcript, "user", text, kind="tool_result",
            tool=call.tool, call_id=call.id, detail=call.detail, entry_id=entry_id,
        )

        if call.tool not in dedupe.REPLAYABLE:
            state.memo.clear()

    return novel


def _salvage(state, body: str, reason: str) -> str:  # noqa: ANN001
    """An answer for a turn that ended before the model wrote one.

    The loop used to assign the *final* round's text, which is empty whenever
    that round was a tool call, so a turn that read forty files and was cut off
    mid-investigation returned an empty string and every caller read that as
    "produced no output". The findings were never lost; only the sentence naming
    them was. The ledger is where they live, so it is what an interrupted turn
    answers with.
    """
    if body:
        return f"{body}\n\n[{reason}]"

    block = render_ledger(state.ledger.live(), soft_tokens=state.session.budget.ledger)
    if block.is_empty:
        return f"[{reason} - the turn produced no findings]"
    return (
        f"[{reason} before a final answer was written. "
        f"What the turn established, from the ledger:]\n\n{block.text}"
    )


def _advance(meters: gates.TurnMeters) -> gates.TurnMeters:
    """Count a round that ran no tool."""
    return replace(meters, round_index=meters.round_index + 1)


def _coverage(state):  # noqa: ANN001
    """The one gate that asks for more work rather than less."""
    if state.meters.coverage_fired:
        return None

    state.meters = replace(state.meters, coverage_fired=True)

    opened = {
        ref.path
        for entry in state.ledger.live()
        if entry.kind == "span"
        for ref in entry.refs
    }
    decision = gates.coverage_gate(
        replace(state.meters, coverage_fired=False),
        central=state.session.ranks.ordered()[:COVERAGE_CENTRAL_FILES],
        opened=opened,
        discovery_calls=len(state.calls),
    )
    return decision.nudge


def _record(state, call: ToolCall, outcome, text: str):  # noqa: ANN001
    """Put this result in the ledger, as a read or as a finding."""
    if call.tool in READ_TOOLS and outcome.ok and outcome.entry_text:
        path, _, span = outcome.entry_text.partition(":")
        start, _, end = span.partition("-")
        ref = ref_for(_workspace(state), path, int(start or 1), int(end or start or 1))
        return record_read(state.ledger, ref.path, ref.start, ref.end, ref.sha)

    return distil_into(
        state.ledger, call.tool, call.arguments, text, workspace=_workspace(state)
    )


def _elide(state) -> None:  # noqa: ANN001
    """§12, inside the turn. Bounds live results without rewriting the cache."""
    elision = hot_cold(
        state.transcript,
        max_tokens=state.session.budget.hot_results,
        max_count=state.session.budget.hot_count,
        rounds_left=state.meters.rounds_left,
    )
    state.transcript = elision.transcript
    if elision.skipped_for_amortisation:
        state.notes = (*state.notes, f"held off collapsing: {elision.reason}")


def _degleak(turn: ModelTurn, state) -> tuple[str, tuple[ToolCall, ...]]:  # noqa: ANN001
    """A tool call written as prose is not a tool call."""
    if not leaks.detect(turn.text):
        return turn.text, ()

    leak = leaks.recover(turn.text, frozenset(TOOL_NAMES))
    state.notes = (*state.notes, leak.detail or f"recovered a leaked {leak.tool} call")
    if not leak.recovered:
        return leaks.CORRECTION, ()
    return "", (ToolCall(id=f"leak-{state.meters.round_index}", tool=leak.tool,
                         arguments=dict(leak.arguments or {})),)


def _plan(state, *, working, model_id, role):  # noqa: ANN001
    """§5 — prefix, rolling marker, tail. Rebuilt every round on purpose."""
    blocks = build_prefix(
        state.session.repo_map,
        state.session.context.memory,
        role,
        model=model_id,
        budget=PromptBudget(
            map_tokens=state.session.budget.repo_map,
            memory_tokens=state.session.budget.memory,
        ),
    )
    return assemble(
        blocks=blocks,
        transcript=state.transcript.to_messages(),
        tail=_tail(state, working),
        model=model_id,
        session_id=state.session.context.root,
        previous_marker=state.session.marker,
    )


def _tail(state, working: WorkingSet) -> str:
    """§13 + §10 + §11, in that order, last in the request."""
    parts = []
    if isinstance(state.session.state, SessionState):
        rendered = state.session.state.render()
        if rendered:
            parts.append(rendered)

    block = render_ledger(state.ledger.live(), soft_tokens=state.session.budget.ledger)
    if not block.is_empty:
        parts.append(block.text)
    if working.text:
        parts.append(working.text)
    return "\n\n".join(parts)


def _close(state, utility) -> tuple[Session, SideUsage, tuple[str, ...]]:  # noqa: ANN001
    """Fold, then compact, then freeze the session for the next turn."""
    transcript = state.transcript
    session_state = state.session.state
    guard = state.session.guard or ThrashGuard()
    side = SideUsage()
    notes: tuple[str, ...] = ()

    if utility is None:
        notes = ("no utility model: this session will not fold or compact",)
    else:
        folded = fold_turn(
            transcript, session_state if isinstance(session_state, SessionState) else None,
            utility, max_tokens=state.session.budget.fold,
        )
        side = side.plus(folded.usage)
        session_state = folded.state
        notes = (*notes, *folded.notes)

    if utility is not None:
        digested = session_state.turns_folded if isinstance(session_state, SessionState) else 0
        result = compact(
            transcript,
            window_tokens=state.session.budget.window,
            folded_turns=digested,
            turn=state.session.turn,
            guard=guard,
            summariser=utility,
        )
        transcript, guard = result.transcript, result.guard
        side = side.plus(result.usage)
        notes = (*notes, *result.notices)

    return (
        state.session.advanced(
            ledger=state.ledger,
            transcript=transcript,
            state=session_state,
            guard=guard,
            turn=state.session.turn + 1,
        ),
        side,
        notes,
    )


def _workspace(state) -> Workspace:  # noqa: ANN001
    return state.session.workspace  # type: ignore[no-any-return]


def _json(arguments: Mapping[str, object]) -> str:
    """A call's arguments as JSON, for the driver that has to re-send them."""
    return json.dumps(arguments, default=repr, sort_keys=True)


def _append(transcript: Transcript, role: str, text: str, **fields: object) -> Transcript:
    return transcript.with_blocks([*transcript.blocks, Block(role=role, text=text, **fields)])  # type: ignore[arg-type]


__all__ = [
    "BUILT_IN",
    "ROLE_PROMPT",
    "FormulationCache",
    "NullHost",
    "Session",
    "TurnResult",
    "TOOL_NAMES",
    "open_session",
    "run_turn",
]
