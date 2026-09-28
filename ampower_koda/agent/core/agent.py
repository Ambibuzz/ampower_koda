"""One turn: working set, prompt, the tool loop, then the fold."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .budget.allocator import allocate
from .budget.request import MESSAGE_OVERHEAD, cleanup_target, input_limit, serialized_tokens
from .constants import (
    COMPACTION_MAX_OUTPUT_TOKENS,
    DEFAULT_ARCHITECT_MODEL,
    MAX_OUTPUT_TOKENS,
    MAX_ROUNDS,
    WORKING_SET_WEAK_COVERAGE,
)
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
from .contracts.escalation import SideUsage
from .contracts.ledger import Ledger
from .contracts.model import UtilityModel
from .contracts.prompt import PromptBudget
from .contracts.transcript import Block, Transcript
from .elide.collapse import READ_TOOLS
from .elide.compact import compact_for_request
from .elide.hotcold import hot_cold
from .fold.document import SessionState
from .fold.run import fold_turn
from .ledger.distill import distil_into
from .ledger.recall import ref_for
from .ledger.render import render_ledger
from .ledger.write import record_read
from .loop import dedupe, gates, leaks
from .prompt.cache import assemble, build_prefix
from .tools.catalogue import CATALOGUE, TOOL_NAMES
from .tokens import estimate_tokens
from .tools.run import NullHost, run_tool
from .workingset.build import working_set_for
from .workspace.ports import Workspace

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
        budget=allocate(
            context.config.context.window_tokens,
            ledger_override=context.config.context.ledger_soft_tokens,
            memory_tokens=context.config.context.memory_tokens,
            input_tokens=context.config.context.input_tokens,
        ),
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

    round_limit = max(1, int(max_rounds))
    output_limit = max(1, int(max_output_tokens))
    per_call = input_limit(session.budget.window, output_limit, session.budget.input_tokens) + output_limit
    state = _State(
        session=session,
        transcript=_append(session.transcript, "user", question),
        ledger=session.ledger,
        meters=gates.TurnMeters(
            max_rounds=round_limit,
            max_turn=session.budget.marginal_turn,
            # Cached replays of a large prompt must not exhaust a small cumulative
            # limit after two calls. The round limit still bounds all work.
            max_observed=max(session.budget.observed_turn, round_limit * per_call),
        ),
    )

    outcome = _rounds(
        state,
        working=working,
        model=model,
        model_id=model_id,
        host=host or NullHost(),
        role_prompt=role_prompt,
        utility=utility,
        max_output_tokens=output_limit,
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
        notes=(*session.notes, *working.notes, *outcome.notes, *notes),
    )


class _State:
    """The loop's working set. Mutable, and scoped to one call of ``run_turn``."""

    __slots__ = (
        "answer", "calls", "ledger", "memo", "meters", "notes", "seen",
        "session", "side", "stop_reason", "transcript", "usage", "fresh_calls", "working_added", "context_text",
        "marker", "marker_prefix",
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
        self.fresh_calls: frozenset[str] = frozenset()
        self.working_added = False
        self.context_text = ""
        self.marker = session.marker
        self.marker_prefix = (session.transcript.blocks[:session.marker.index + 1]
                              if session.marker is not None else ())


def _rounds(state, *, working, model, model_id, host, role_prompt,
            max_output_tokens, utility=None):  # noqa: ANN001, PLR0913
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

        request = _prepare_request(state, working=working, model=model, model_id=model_id,
                                   role=role_prompt, max_output_tokens=max_output_tokens,
                                   forced=forced, utility=utility)
        if request is None:
            return state
        turn = model.respond(request)
        state.fresh_calls = frozenset()
        state.usage = state.usage.plus(turn.usage)
        state.meters = state.meters.charged(
            processed=turn.usage.processed, observed=turn.usage.observed
        )

        if turn.failed:
            state.stop_reason = "error"
            state.notes = (*state.notes, turn.detail or "the model call failed")
            state.answer = turn.text
            return state

        state.marker = request.plan.marker
        state.marker_prefix = (request.transcript.blocks[:state.marker.index + 1]
                               if state.marker is not None else ())
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
            if not text:
                # Neither an answer nor a call: never "answered". What the ledger holds is still
                # reported; with no findings the answer stays empty.
                state.stop_reason = "empty"
            state.answer = text or _salvage(state, "", "the model returned no text")
            return state

        if forced:
            late = gates.late_tool_call()
            state.answer = _salvage(state, last_text, late.reason)
            state.stop_reason = late.reason
            return state

        novel = _dispatch(state, calls, host=host,
                          provider_context_json=turn.provider_context_json)
        state.meters = state.meters.next_round(dry=gates.is_dry(novel, max(novel, 1)))

    state.stop_reason = "rounds exhausted"
    state.answer = _salvage(state, last_text, "rounds exhausted")
    return state


def _dispatch(state, calls: Sequence[ToolCall], *, host, provider_context_json='') -> int:  # noqa: ANN001
    """Run every call in emission order, distilling each result as it lands."""
    novel = 0
    state.fresh_calls = frozenset(call.id for call in calls)
    for index, call in enumerate(calls):
        state.calls = (*state.calls, call.tool)
        state.transcript = _append(
            state.transcript, "assistant", "", kind="tool_use",
            tool=call.tool, call_id=call.id, detail=call.detail,
            arguments_json=_json(call.arguments),
            provider_context_json=provider_context_json if index == 0 else '',
            parallel_call_ids=tuple(c.id for c in calls) if index == 0 and provider_context_json else (),
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

    Falls back to the ledger, so an interrupted turn still reports its findings.
    Empty when there are none: a placeholder would read as an answer (stop_reason says why).
    """
    if body:
        return f"{body}\n\n[{reason}]"

    block = render_ledger(state.ledger.entries, soft_tokens=state.session.budget.ledger)
    if block.is_empty:
        return ""
    return (
        f"[{reason} before a final answer was written. "
        f"What the turn established, from the ledger:]\n\n{block.text}"
    )


def _advance(meters: gates.TurnMeters) -> gates.TurnMeters:
    """Count a round that ran no tool."""
    return replace(meters, round_index=meters.round_index + 1)


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


def _elide(state, max_tokens: int) -> None:  # noqa: ANN001
    """Batch older results into stubs while keeping unseen results available."""
    elision = hot_cold(
        state.transcript,
        max_tokens=max_tokens,
        max_count=state.transcript.live_result_count() + 1,
        target_tokens=max_tokens,
        target_count=state.transcript.live_result_count(),
        protected_call_ids=state.fresh_calls,
        evict_undistilled_reads=True,
    )
    state.transcript = elision.transcript
    if elision.changed:
        state.memo.clear()  # Removed evidence must remain re-readable.
        state.notes = (*state.notes, f"elided {elision.dropped_tokens:,} tool-result tokens")


def _prepare_request(state, *, working, model, model_id, role, max_output_tokens, forced, utility):
    """Leave history intact until full-input pressure, then free substantial room."""
    budget = state.session.budget
    limit = input_limit(budget.window, max_output_tokens, budget.input_tokens)

    if not state.working_added:
        state.working_added = True
        # Weak retrieval is omitted before the first call. Useful retrieval is
        # recorded once in history so new tool messages extend its cached prefix.
        if working.text and (working.reranked or working.coverage >= WORKING_SET_WEAK_COVERAGE):
            state.transcript = _append(state.transcript, "user", working.text, kind="context")
    context_text = _tail(state)
    if context_text and context_text != state.context_text:
        state.transcript = _append(state.transcript, "user",
                                   "Session context snapshot (newer tool evidence takes precedence):\n" + context_text,
                                   kind="context")
    state.context_text = context_text

    def build():
        return ModelRequest(plan=_plan(state, model_id=model_id, role=role),
                            transcript=state.transcript, tools=TOOL_NAMES,
                            max_tokens=max_output_tokens, force_terminal=forced,
                            input_tokens_limit=limit)

    def measure(request):
        estimator = getattr(model, "estimate_request", None)
        if estimator is not None:
            return estimator(request)
        schemas = [(s.name, s.parameters, s.description, s.caps) for s in CATALOGUE]
        return (estimate_tokens(request.system_text) +request.transcript.tokens
                + MESSAGE_OVERHEAD * (len(request.transcript.blocks) + 2)
                + serialized_tokens(schemas))

    request = build()
    size = measure(request)
    if size <= limit:
        return request
    target = cleanup_target(limit)
    original = state.transcript
    fixed = measure(replace(request, transcript=Transcript()))
    scale = max(1.0, (size - fixed) / max(1, original.tokens))
    pruned_size = size
    while pruned_size > target:
        to_drop = int((pruned_size - target) / scale) + 1
        _elide(state, max_tokens=max(0, state.transcript.live_result_tokens() - to_drop))
        request = build()
        after = measure(request)
        if after >= pruned_size:
            pruned_size = after
            break
        pruned_size = after
    if pruned_size <= target:
        return request
    pruned = state.transcript
    pinned = sum(b.tokens for b in original.blocks if b.role == "user" and b.is_prose)
    keep = max(0, int((target - fixed) / scale) - pinned - COMPACTION_MAX_OUTPUT_TOKENS)
    # Summarize the original evidence, not the stubs from the pruning attempt.
    reduced = compact_for_request(original, summariser=utility,
                                  keep_tokens=keep,
                                  protected_call_ids=state.fresh_calls,
                                  max_input_tokens=input_limit(budget.window, 1_024, budget.window))
    state.side = state.side.plus(reduced.usage)
    state.meters = state.meters.charged(processed=reduced.usage.total_tokens,
                                      observed=reduced.usage.total_tokens)
    state.notes = (*state.notes, *reduced.notices)
    if reduced.changed:
        state.transcript = reduced.transcript
        if measure(build()) > pruned_size:
            state.transcript = pruned
        state.memo.clear()
    request = build()
    size = measure(request)
    if size <= limit:
        return request
    state.stop_reason = "error"
    state.answer = (f"Context budget exceeded: estimated {size:,} input tokens, limit {limit:,}. "
                    "Narrow the task or increase context.input_tokens in .koda/config.toml. "
                    "Required requests and fresh tool results were preserved.")
    return None


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


def _plan(state, *, model_id, role):  # noqa: ANN001
    """Build the stable system prefix and markers for retained history."""
    if state.marker and state.transcript.blocks[:state.marker.index + 1] != state.marker_prefix:
        state.marker = None
        state.marker_prefix = ()
    blocks = build_prefix(
        state.session.context.memory,
        role,
        model=model_id,
        budget=PromptBudget(
            memory_tokens=state.session.budget.memory,
        ),
    )
    return assemble(
        blocks=blocks,
        transcript=state.transcript.to_messages(),
        model=model_id,
        session_id=state.session.context.root,
        previous_marker=state.marker,
    )


def _tail(state) -> str:
    """Memory and findings not already visible in live results, for a snapshot.

    The caller appends changed snapshots to history. Moving this content to a
    fresh tail every round would abandon the previous message-end cache entry.
    """
    parts = []
    if isinstance(state.session.state, SessionState):
        rendered = state.session.state.render()
        if rendered:
            parts.append(rendered)

    block = render_ledger(_ledger_entries(state), soft_tokens=state.session.budget.ledger)
    if not block.is_empty:
        parts.append(block.text)
    return "\n\n".join(parts)


def _ledger_entries(state) -> list:  # noqa: ANN001
    """Ledger entries whose result is no longer live in the transcript, plus pinned ones."""
    visible = {
        block.entry_id
        for block in state.transcript.blocks
        if block.is_result and not block.elided and block.entry_id
    }
    return [entry for entry in state.ledger.entries if entry.pinned or entry.id not in visible]


def _close(state, utility) -> tuple[Session, SideUsage, tuple[str, ...]]:  # noqa: ANN001
    """Update session memory without rewriting history between requests."""
    transcript = state.transcript
    session_state = state.session.state
    side = SideUsage()
    notes: tuple[str, ...] = ()

    if utility is None:
        notes = ("no utility model: this session will not fold or summarize",)
    else:
        folded = fold_turn(
            transcript, session_state if isinstance(session_state, SessionState) else None,
            utility, max_tokens=state.session.budget.fold,
        )
        side = side.plus(folded.usage)
        session_state = folded.state
        notes = (*notes, *folded.notes)

    return (
        state.session.advanced(
            ledger=state.ledger,
            transcript=transcript,
            state=session_state,
            marker=state.marker,
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
    "ROLE_PROMPT",
    "NullHost",
    "Session",
    "TurnResult",
    "TOOL_NAMES",
    "open_session",
    "run_turn",
]
