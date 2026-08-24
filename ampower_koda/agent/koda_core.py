# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
"""The edge between Koda's LangGraph and ``agent/core``'s retrieval pipeline.

``agent/core`` is pure: it imports nothing from ``frappe``, ``langchain`` or
``langgraph``, and it reaches no network. Everything it needs from outside
arrives through four small seams, and this module is all four of them plus the
one function the graph calls::

    understand(state) -> Understanding

**What the core does that the old explore loop did not.** The previous
understanding node was an LLM with five read tools and a history trimmer. This
one adds, in the order a turn uses them: a tree-sitter index of the whole app, a
PageRank'd repo map in the cached prefix, a per-message retrieval pass that runs
*before* the model says anything, a ranked ``search`` that fuses BM25 with graph
proximity, an append-only ledger so a finding survives its own tool result, and
hot/cold elision that turns an old result into ``[search "x" -> L14]`` instead of
dropping it. The trimmer is replaced by a fold that summarises a turn *before*
anything is deleted.

**Four seams, and why each is here rather than there.**

``ChatModel``       one provider request per round. Only this class knows what
                    ``cache_control`` is spelled like.
``UtilityModel``    the fold and compaction summariser. Optional: without it a
                    session simply never folds, and says so in ``notes``.
``ToolHost``        the tools the core cannot implement against a read-only
                    workspace. In this phase that is ``read_doctype_schema``
                    and nothing else — every writing tool is declined, so the
                    understanding phase is read-only *structurally* rather than
                    by review.
``Workspace``       already implemented by the core's ``LocalWorkspace``, over
                    the app root Frappe resolves.

**A session is expensive once and free afterwards.** Cold start indexes the app;
on this repository that is well under a second, but it is not free, and the
`Session` it produces is a frozen value that carries the index, the map, the
retriever and the ledger. It cannot go into LangGraph state — that state is
JSON-persisted — so it lives in a process-local cache keyed by request name, and
a cache miss simply pays for cold start again. Nothing is *wrong* after a miss;
it is slower.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import frappe
from ampower_koda.agent.errors import log_agent_error
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ampower_koda.agent import tools as agent_tools
from ampower_koda.agent.core import (
    ROLE_PROMPT,
    LocalWorkspace,
    ModelRequest,
    ModelTurn,
    Session,
    ToolCall,
    ToolOutcome,
    TurnUsage,
    open_session,
    run_turn,
)
from ampower_koda.agent.core.constants import DEFAULT_ARCHITECT_MODEL
from ampower_koda.agent.core.contracts.escalation import SideUsage
from ampower_koda.agent.core.contracts.model import Completion
from ampower_koda.agent.core.tools.catalogue import CATALOGUE

#: Frappe doctype the graph's request rows live in.
DOCTYPE_NAME = "Agent Request"

#: Tools this host does not implement, and what to do instead. A refusal names
#: the replacement because a model told "no" tries a synonym, and a model told
#: "use refs" uses refs.
#:
#: There is nothing here about editing. The core's catalogue has no writing tool
#: at all, so the understanding phase is read-only by *construction* rather than
#: by this dictionary remembering to say so — which is the version of that
#: guarantee that survives somebody adding a phase and forgetting to check.
DECLINED = {
    "trace_discover": "not wired — use refs to find call sites",
    "ast_search": "not wired — use search or grep",
}

#: Fields of a doctype rendered per row. The JSON also carries layout metadata —
#: column breaks, tab breaks, permissions, view settings — which is most of the
#: file's bytes and none of what a question about a doctype is asking.
DOCTYPE_FIELD_KEYS = ("fieldname", "fieldtype", "label", "options", "reqd")

#: Rows of a doctype's field table handed back before truncating.
DOCTYPE_FIELD_ROWS = 200

#: This adapter runs a bounded planning investigation, not an open-ended repo
#: chat. Independent tools may be emitted together, so sixteen rounds leave
#: ample room without exposing the core's sixty-round emergency ceiling.
UNDERSTANDING_MAX_ROUNDS = 16
UNDERSTANDING_MAX_OUTPUT_TOKENS = 4_096


# ---------------------------------------------------------------------------
# Seam 1 — the conversational model
# ---------------------------------------------------------------------------


class LangChainChatModel:
    """One provider request per round, built from a :class:`ModelRequest`.

    The request is the whole input: system blocks with their cache boundaries,
    the transcript those boundaries index into, and the frozen tool array. This
    class turns that into LangChain messages and turns the reply back into a
    :class:`ModelTurn`.

    **It never raises.** A turn arrives here with up to sixty rounds of reads
    behind it, and an exception thrown out of round forty-one discards all of
    them. Every failure becomes ``ModelTurn(failed=True)``, which the loop
    records as the answer and returns with the session intact.

    **It publishes what it sends.** The realtime feed the Agent Request form
    reads used to be written by the tool loop, which no longer exists here. The
    driver is the right replacement: it sees every request, so on each round it
    publishes the transcript blocks that appeared since the last one. That is
    strictly more truthful than publishing at call time — what shows in the UI
    is what the model was actually shown.
    """

    def __init__(self, llm, provider: str, request_name: str = "", spent: int = 0) -> None:
        self.llm = llm
        self.provider = (provider or "").strip()
        self.request_name = request_name
        self.rounds = 0
        self.total_tokens = spent
        """Seeded with what earlier phases already spent. The number is written
        straight to the request row, and a phase that started its count at zero
        would overwrite the running total with its own share of it."""

        self.failure = ""
        """Why the last call failed, for the caller that has to report it.

        Kept here rather than dug out of the turn's ``notes``, which also carry
        ordinary session remarks — "co-change memory unavailable" is the first
        of them on any tree without a git log, and reporting *that* as the
        reason a request failed sends someone to look at the wrong thing."""
        self._published = 0
        self.model_id = _model_id(llm)
        """The model, for the cache decision. Read off the client rather than
        passed in, so it is the id actually being called."""
        self._bound = llm.bind_tools(_tool_schemas()) if hasattr(llm, "bind_tools") else llm

    # -- the port -----------------------------------------------------------

    def respond(self, request: ModelRequest) -> ModelTurn:
        self.rounds += 1
        self._publish_new_blocks(request.transcript)

        try:
            messages = self._messages(request)
        except Exception as error:  # pragma: no cover - defensive; see class docstring
            return self._failed("could not build the request", error)

        model = self.llm if request.force_terminal else self._bound
        try:
            reply = model.invoke(messages, max_tokens=request.max_tokens)
        except TypeError:
            # Not every LangChain provider accepts a per-call max_tokens.
            try:
                reply = model.invoke(messages)
            except Exception as error:
                return self._failed("the model call failed", error)
        except Exception as error:
            return self._failed("the model call failed", error)

        return self._turn(reply)

    # -- request ------------------------------------------------------------

    def _messages(self, request: ModelRequest) -> list:
        """System blocks, the conversation, then the tail — in that order.

        The tail is last and is never cached: it is the session state, the
        ledger and this message's working set, and it changes every round. That
        asymmetry is the point of the whole cache plan — the expensive stable
        half sits above a boundary and is read, and the volatile half below it
        is cheap because it is small.
        """
        messages: list = [self._system(request)]
        messages.extend(_replay(request.transcript))
        if request.plan.tail:
            messages.append(HumanMessage(content=request.plan.tail))
        return messages

    def _system(self, request: ModelRequest):
        """The system blocks, with a cache breakpoint where the plan asks for one.

        Anthropic takes an explicit ``cache_control`` marker; OpenAI and DeepSeek
        cache long stable prefixes on their own; everything else gets one plain
        string. In all three cases the *text* is identical, so the plan's
        boundaries only ever change the price.

        The choice is made on the **model**, not on the provider name. An
        Anthropic model reached through OpenRouter is spelled
        ``anthropic/claude-sonnet-4`` with ``provider == "OpenRouter"``, and
        keying on the provider sent it the uncached path — so the one model
        family that *requires* an explicit marker was the one family that never
        got one, and every round paid full price for the whole prefix.
        """
        blocks = [block for block in request.plan.blocks if not block.is_empty]
        if not blocks:
            return SystemMessage(content=ROLE_PROMPT)
        if not _takes_cache_control(self.model_id):
            return SystemMessage(content="\n\n".join(block.text for block in blocks))

        content = []
        for block in blocks:
            part = {"type": "text", "text": block.text}
            if block.breakpoint:
                part["cache_control"] = {"type": "ephemeral"}
            content.append(part)
        return SystemMessage(content=content)

    # -- reply --------------------------------------------------------------

    def _turn(self, reply) -> ModelTurn:
        usage = _usage(reply)
        self.total_tokens += usage.observed
        _persist_tokens(self.request_name, self.total_tokens)

        text = _text_of(reply)
        if text:
            _publish(self.request_name, "llm_response", preview=text[:4000], round=self.rounds)
        _publish(
            self.request_name, "token_usage",
            round=self.rounds,
            tokens_this_round=usage.observed,
            tokens_total=self.total_tokens,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read,
            cache_write_tokens=usage.cache_write,
        )

        calls = tuple(
            ToolCall(
                id=str(call.get("id") or f"c{self.rounds}-{index}"),
                tool=str(call.get("name") or ""),
                arguments=dict(call.get("args") or {}),
            )
            for index, call in enumerate(getattr(reply, "tool_calls", None) or [])
        )
        return ModelTurn(
            text=text,
            calls=calls,
            usage=usage,
            stopped_at_limit=_hit_output_limit(reply),
        )

    def _failed(self, detail: str, error: Exception) -> ModelTurn:
        self.failure = f"{detail}: {error}"
        log_agent_error(
            "Koda core: chat model",
            f"request={self.request_name}\n{detail}: {error}\n{frappe.get_traceback()}",
        )
        return ModelTurn(text=f"[{detail}: {error}]", failed=True, detail=f"{detail}: {error}")

    # -- progress -----------------------------------------------------------

    def _publish_new_blocks(self, transcript) -> None:
        """Publish the transcript blocks added since the previous round."""
        blocks = transcript.blocks
        for block in blocks[self._published :]:
            if block.kind == "tool_use":
                _publish(
                    self.request_name, "tool_call",
                    tool_name=block.tool, tool_args=block.detail, round=self.rounds,
                )
            elif block.is_result:
                _publish(
                    self.request_name, "tool_result",
                    tool_name=block.tool, result_preview=block.text[:500], round=self.rounds,
                )
        self._published = len(blocks)


def _replay(transcript) -> list:
    """The transcript as LangChain messages, pairs kept together.

    A ``tool_use`` block becomes an ``AIMessage`` carrying one tool call, and its
    ``tool_result`` becomes the matching ``ToolMessage``. The arguments come from
    the block's ``arguments_json`` rather than from a driver-side memo: a memo
    works right up until the worker restarts, and a resumed session would then
    replay its tool calls with no arguments at all.

    Consecutive calls in one round each get their own ``AIMessage``. One message
    with several calls is the tidier wire format and it is not worth the risk
    here: the core emits calls in strict emission order and every provider
    accepts a one-call-per-message sequence, while grouping requires the driver
    to reconstruct round boundaries the transcript does not record.
    """
    messages: list = []
    for block in transcript.blocks:
        if block.kind == "tool_use":
            messages.append(AIMessage(content="", tool_calls=[{
                "name": block.tool,
                "args": _arguments(block),
                "id": block.call_id,
            }]))
        elif block.is_result:
            messages.append(ToolMessage(content=block.text or "(no output)",
                                        tool_call_id=block.call_id))
        elif block.text:
            messages.append(
                HumanMessage(content=block.text) if block.role == "user"
                else AIMessage(content=block.text)
            )
    return messages


def _arguments(block) -> dict:
    """A tool call's arguments, or an empty mapping.

    Empty rather than a guess. A provider re-sent a call with *invented*
    arguments would be shown a conversation that never happened, and the model
    would reason from it — worse than a call whose arguments it can see are
    missing.
    """
    if not block.arguments_json:
        return {}
    try:
        parsed = json.loads(block.arguments_json)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_schemas() -> list[dict]:
    """The frozen catalogue as provider tool schemas.

    Built from ``CATALOGUE`` and nowhere else. Every parameter is a string here
    because the core's specs describe parameters positionally — ``"glob?"``,
    ``"target = anchor|symbol|span"`` — and inventing a JSON-Schema type per
    parameter would be this module asserting something the catalogue never said.
    The description carries the real contract, including the caps.
    """
    schemas = []
    for spec in CATALOGUE:
        caps = f" Limits: {', '.join(spec.caps)}." if spec.caps else ""
        schemas.append({
            "name": spec.name,
            "description": (spec.description + caps).strip(),
            "parameters": {
                "type": "object",
                "properties": {
                    parameter.rstrip("?"): {"type": _parameter_type(parameter.rstrip("?"))}
                    for parameter in spec.parameters
                },
                "required": [
                    parameter for parameter in spec.parameters if not parameter.endswith("?")
                ],
            },
        })
    return schemas


#: Parameters a provider should send as numbers. Everything else is a string —
#: the tools coerce, and a schema that guessed richer types than the catalogue
#: states would be this module asserting something the catalogue never said.
NUMERIC_PARAMETERS = frozenset({"start", "end"})


def _parameter_type(name: str) -> str:
    return "integer" if name in NUMERIC_PARAMETERS else "string"


def _text_of(reply) -> str:
    """Reply text as one string. The Responses API returns a list of blocks."""
    content = getattr(reply, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ]
        return "".join(part for part in parts if part)
    return str(content or "")


def _usage(reply) -> TurnUsage:
    """What the round cost, split the way the core's two budgets need it.

    Cache reads are pulled out of the input count rather than left in it. The
    loop re-sends its whole prefix every round, so a meter that charged reads at
    full price grew with the square of the round count — an 80k ceiling closed
    on round four of a nominal sixty.
    """
    metadata = getattr(reply, "usage_metadata", None) or {}
    details = metadata.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or 0)
    cache_write = int(details.get("cache_creation") or 0)
    total_input = int(metadata.get("input_tokens") or 0)
    return TurnUsage(
        input_tokens=max(0, total_input - cache_read - cache_write),
        cache_write=cache_write,
        cache_read=cache_read,
        output_tokens=int(metadata.get("output_tokens") or 0),
    )


def _hit_output_limit(reply) -> bool:
    """Whether the reply stopped because it ran out of output budget.

    Worth detecting rather than ignoring: the loop offers two continuations and
    then says plainly that it was cut off, and a truncated answer read as a
    complete one is the failure that makes a cut-off worse than an error.
    """
    metadata = getattr(reply, "response_metadata", None) or {}
    reason = str(
        metadata.get("finish_reason") or metadata.get("stop_reason") or ""
    ).lower()
    return reason in ("length", "max_tokens", "model_length")


# ---------------------------------------------------------------------------
# Seam 2 — the utility model
# ---------------------------------------------------------------------------


class LangChainUtility:
    """One call, one string back. Powers the fold and compaction.

    Failure is a value here for the same reason as everywhere else in the core:
    every caller is doing something *optional*. A fold that fails means the
    session did not fold, which costs some context later; a fold that raises
    means a turn that had already answered returns an error instead.
    """

    def __init__(self, llm, request_name: str = "") -> None:
        self.llm = llm
        self.request_name = request_name

    def complete(self, system: str, user: str, *, max_tokens: int) -> Completion:
        try:
            reply = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
        except Exception as error:
            log_agent_error(
                "Koda core: utility model",
                f"request={self.request_name}\n{error}\n{frappe.get_traceback()}",
            )
            return Completion(failed=True, detail=str(error))

        usage = _usage(reply)
        return Completion(
            text=_text_of(reply)[: max_tokens * 8],
            usage=SideUsage(
                calls=1,
                input_tokens=usage.input_tokens + usage.cache_read + usage.cache_write,
                output_tokens=usage.output_tokens,
            ),
        )


# ---------------------------------------------------------------------------
# Seam 3 — the tool host
# ---------------------------------------------------------------------------


@dataclass
class UnderstandingHost:
    """The tools the core cannot implement against a read-only workspace.

    Exactly one of the three is supplied — ``read_doctype_schema`` — and the
    other two are declined by name rather than dropped from the array. The array
    is frozen at session start and is part of the cached prefix, so removing a
    row would invalidate the prefix *and* make the core's ``TOOL_NAMES``
    disagree with what the model was handed, which is the list leak recovery
    checks a leaked call against.

    A refusal names what to do instead: a model told "no" tries a synonym.
    """

    app_name: str
    request_name: str = ""
    refused: list = field(default_factory=list)

    def call(self, name: str, arguments) -> ToolOutcome:
        if name == "read_doctype_schema":
            return self._doctype(str(arguments.get("doctype") or ""))

        reason = DECLINED.get(name, f"{name} is not available in the understanding phase")
        self.refused.append(name)
        return ToolOutcome(text=f"[declined: {reason}]", ok=False)

    def _doctype(self, doctype: str) -> ToolOutcome:
        """The doctype's fields, not its layout.

        The raw JSON is mostly column breaks, tab breaks, permission rows and
        view settings. Handing all of it over spends the round's context on
        metadata nobody asked about, and buries the five fields that answer the
        question.
        """
        if not doctype:
            return ToolOutcome(text="[error: read_doctype_schema needs a doctype]", ok=False)

        raw = agent_tools.read_doctype_schema(self.app_name, doctype)
        if raw.startswith("Error:") or raw.startswith("DocType schema not found"):
            return ToolOutcome(text=f"[error: {raw}]", ok=False)
        try:
            schema = json.loads(raw)
        except ValueError as error:
            return ToolOutcome(text=f"[error: {doctype} schema is not valid JSON — {error}]",
                               ok=False)

        fields = schema.get("fields") or []
        rows = [
            " ".join(
                f"{key}={schema_field[key]}"
                for key in DOCTYPE_FIELD_KEYS
                if schema_field.get(key) not in (None, "", 0)
            )
            for schema_field in fields[:DOCTYPE_FIELD_ROWS]
        ]
        header = [
            f"doctype: {schema.get('name', doctype)}",
            f"module: {schema.get('module', '?')}"
            f"  is_submittable: {schema.get('is_submittable', 0)}",
            f"fields: {len(fields)}",
        ]
        dropped = max(0, len(fields) - DOCTYPE_FIELD_ROWS)
        if dropped:
            rows.append(f"… +{dropped} more fields (truncated)")
        return ToolOutcome(
            text="\n".join([*header, *rows]),
            truncated=bool(dropped),
            dropped=dropped,
        )


# ---------------------------------------------------------------------------
# The session cache
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, Session] = {}
_SESSIONS_LOCK = threading.Lock()

#: Sessions held in this worker before the oldest is dropped. Small: a session
#: holds an index of a whole app, and a worker serving eleven requests at once
#: is not the shape this runs in.
MAX_CACHED_SESSIONS = 10


def _cached(request_name: str) -> Session | None:
    with _SESSIONS_LOCK:
        return _SESSIONS.get(request_name)


def _remember(request_name: str, session: Session) -> None:
    """Hold the session for the next turn of the same request.

    Process-local on purpose. A ``Session`` carries the index, the repo map and
    the retriever, none of which are JSON, so it cannot ride in LangGraph state —
    and a cache that spanned workers would have to serialise all three. A miss
    costs one cold start and loses nothing: the ledger and transcript are
    rebuilt from the request row, and cold start on this tree is sub-second.
    """
    if not request_name:
        return
    with _SESSIONS_LOCK:
        _SESSIONS[request_name] = session
        while len(_SESSIONS) > MAX_CACHED_SESSIONS:
            _SESSIONS.pop(next(iter(_SESSIONS)))


def forget_session(request_name: str) -> None:
    """Drop a request's cached session. Call when a request finishes."""
    with _SESSIONS_LOCK:
        _SESSIONS.pop(request_name, None)


# ---------------------------------------------------------------------------
# The one function the graph calls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Understanding:
    """What one understanding pass produced."""

    summary: str
    explored_paths: tuple[str, ...] = ()
    tools_called: tuple[str, ...] = ()
    rounds: int = 0
    tokens: int = 0
    notes: tuple[str, ...] = ()
    error: str = ""
    stop_reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.summary.strip())

    @property
    def why(self) -> str:
        """Why this pass produced nothing, in the caller's words rather than none.

        Only ``stop_reason == "error"`` used to reach the graph, so a turn that
        ran out of rounds or was stopped by a late tool call arrived with an
        empty ``error`` and was reported as "produced no output" - the one
        message that says nothing about the cause.
        """
        if self.error:
            return self.error
        if self.stop_reason and self.stop_reason != "answered":
            return f"Understanding phase stopped: {self.stop_reason}"
        return "Understanding phase produced no output"


def understand(
    *,
    question: str,
    app_name: str,
    llm,
    provider: str,
    request_name: str = "",
    system_prompt: str = "",
    retrieval_query: str = "",
    utility_llm=None,
    spent: int = 0,
) -> Understanding:
    """Run one full core turn and return the summary the plan phase needs.

    Everything §1–§17 does happens inside :func:`run_turn`: cold start, the
    working set, prompt assembly with its cache plan, the round loop, tools, the
    ledger, elision, the fold and compaction. What this function adds is the
    four seams and the translation back to the flat ``understanding_summary``
    string the rest of the graph already knows how to read.

    Never raises. The graph's nodes short-circuit on ``state["error"]``, so a
    failure has to arrive as a value or the whole request dies on a traceback.
    """
    chat = None
    host = UnderstandingHost(app_name=app_name, request_name=request_name)

    try:
        chat = LangChainChatModel(
            llm, provider=provider, request_name=request_name, spent=spent
        )
        session = _cached(request_name)
        if session is None:
            session = open_session(
                LocalWorkspace(root_path=Path(_app_root(app_name))),
                model=chat.model_id,
                overrides=_overrides(chat.model_id),
            )
        result = run_turn(
            question,
            session=session,
            model=chat,
            host=host,
            utility=LangChainUtility(utility_llm or llm, request_name) if utility_llm else None,
            role_prompt=_role_prompt(system_prompt),
            retrieval_query=retrieval_query,
            max_rounds=UNDERSTANDING_MAX_ROUNDS,
            max_output_tokens=UNDERSTANDING_MAX_OUTPUT_TOKENS,
        )
    except Exception as error:
        log_agent_error(
            "Koda core: understand",
            f"request={request_name}\napp={app_name}\n{error}\n{frappe.get_traceback()}",
        )
        return Understanding(
            summary="",
            error=str(error),
            tokens=chat.total_tokens if chat is not None else spent,
            stop_reason="error",
        )

    _remember(request_name, result.session)
    notes = tuple(result.notes)
    if host.refused:
        notes = (*notes, f"declined: {', '.join(sorted(set(host.refused)))}")

    return Understanding(
        summary=result.answer,
        explored_paths=_opened(result.session),
        tools_called=tuple(result.calls),
        rounds=result.rounds,
        tokens=chat.total_tokens,
        notes=notes,
        error=chat.failure if result.stop_reason == "error" else "",
        stop_reason=result.stop_reason,
    )


def _app_root(app_name: str) -> str:
    """The app's own directory — the whole of what the core is allowed to see.

    ``LocalWorkspace`` refuses any path that resolves outside its root, so this
    one value is the sandbox. Frappe resolves it; nothing here builds a path by
    hand.
    """
    if not app_name:
        raise ValueError("target app name is required to open a session")
    return frappe.get_app_path(app_name)


def _model_id(llm) -> str:
    """The model id, for the cache-limit table the prompt assembler consults.

    Only the *family* in the string matters — the core makes no model calls, it
    only needs to know how wide a block has to be before caching it pays, and a
    version suffix does not change that. Falls back to the core's default, whose
    table is the conservative one.
    """
    found = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "").strip()
    return found or DEFAULT_ARCHITECT_MODEL


CACHE_CONTROL_FAMILIES = ("claude", "anthropic/")


def _takes_cache_control(model_id: str) -> bool:
    identifier = (model_id or "").lower()
    return any(family in identifier for family in CACHE_CONTROL_FAMILIES)


MODEL_WINDOWS = {
    "deepseek/deepseek-v4-flash": 128_000,
    "deepseek/deepseek-chat": 64_000,
    "qwen/qwen-2.5-coder": 32_000,
    "gpt-4o-mini": 128_000,
    "gpt-5": 400_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "claude-3-5": 200_000,
    "claude-sonnet-4": 200_000,
}


def _window_for(model_id: str) -> int:
    """The model's context window, or 0 to let the core decide.

    Longest match first: ``deepseek/deepseek-v4-flash-0731`` must not be scored
    against a shorter key that happens to be a prefix of a different model.
    """
    identifier = (model_id or "").lower()
    for name in sorted(MODEL_WINDOWS, key=len, reverse=True):
        if name in identifier:
            return MODEL_WINDOWS[name]
    return 0


def _overrides(model_id: str) -> dict | None:
    """Cold-start config for this model, or ``None`` for the core's defaults.

    Only the window is set, because every other budget is derived from it — and
    the derivation is the core's business, not this module's. Handing over one
    measured number is a different act from second-guessing the allocator.
    """
    window = _window_for(model_id)
    return {"context": {"window_tokens": window}} if window else None


def _role_prompt(system_prompt: str) -> str:
    """Koda's conventions first, then the core's reading discipline.

    Both, and in that order. The house rules are what make an answer usable
    here; the discipline — prefer outline over read, cite path:line, say when
    you have not read something — is what keeps the turn cheap enough to finish.
    Dropping either has been tried and shows up as a different failure.
    """
    house = (system_prompt or "").strip()
    return f"{house}\n\n{ROLE_PROMPT}" if house else ROLE_PROMPT


def _opened(session: Session) -> tuple[str, ...]:
    """Paths the turn actually read, from the ledger's span entries.

    The ledger rather than a regex over the answer. A path scraped out of prose
    is a path the model *mentioned*, which is a different and much weaker claim
    than one it opened — and the span entries are the same set that would gate
    editing.
    """
    paths: list[str] = []
    for entry in session.ledger.live():
        if entry.kind != "span":
            continue
        for ref in entry.refs:
            if ref.path not in paths:
                paths.append(ref.path)
    return tuple(paths)


def _publish(request_name: str, log_type: str, **payload) -> None:
    """One realtime event. Silent on failure — a log is not worth a turn."""
    if not request_name:
        return
    try:
        user = frappe.db.get_value(DOCTYPE_NAME, request_name, "owner") or "Administrator"
        frappe.publish_realtime(
            "agent_log",
            {"request_name": request_name, "type": log_type, **payload},
            user=user,
        )
    except Exception:
        log_agent_error(
            "Koda core: publish agent_log",
            f"request={request_name}\ntype={log_type}\n{frappe.get_traceback()}",
        )


def _persist_tokens(request_name: str, total: int) -> None:
    if not request_name:
        return
    try:
        frappe.db.set_value(DOCTYPE_NAME, request_name, "tokens_used", int(total))
        frappe.db.commit()
    except Exception:
        log_agent_error(
            "Koda core: persist token usage",
            f"request={request_name}\n{frappe.get_traceback()}",
        )
