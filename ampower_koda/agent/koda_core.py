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

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import frappe
from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.cache_usage import persist_usage, provider_cost
from ampower_koda.agent import recovery
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ampower_koda.agent import tools as agent_tools
from ampower_koda.agent.prompt_caching import mark_message, native_cache_messages, openai_breakpoints, terminal_model
from ampower_koda.agent.run_control import check_active, MODEL_TIME_RESERVE
from ampower_koda.agent.reranking import OpenRouterReranker
from ampower_koda.agent.core.budget.calibrator import TokenCalibrator
from ampower_koda.agent.core.budget.request import estimate_messages
from ampower_koda.agent.core.config.merge import merge_config
from ampower_koda.agent.core.context.bootstrap import _resolve_config
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
    search,
)
from ampower_koda.agent.core.constants import DEFAULT_ARCHITECT_MODEL
from ampower_koda.agent.git_ops import get_repo_root, run_git, worktree_signature
from ampower_koda.agent.core.contracts.escalation import SideUsage
from ampower_koda.agent.core.contracts.model import Completion
from ampower_koda.agent.core.retrieval.excerpts import excerpt
from ampower_koda.agent.core.retrieval.query import plain_query
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

    It never raises: every failure becomes ``ModelTurn(failed=True)``, so the
    loop keeps the session. Each round it publishes the transcript's new tool
    blocks to the form's realtime feed.
    """

    def __init__(self, llm, provider: str, request_name: str = "", spent: int = 0) -> None:
        self.llm = llm
        self.provider = (provider or "").strip()
        self.request_name = request_name
        self.rounds = 0
        # Seeded with earlier spend: it is written to the row as the running total.
        self.total_tokens = spent
        # Why the last call failed; session ``notes`` also carry ordinary remarks.
        self.failure = ""
        self._published: set[tuple[str, str]] = set()
        # Read off the client, so the cache decision uses the id actually called.
        self.model_id = _model_id(llm)
        self._schemas = _tool_schemas()
        self._bound = llm.bind_tools(self._schemas) if hasattr(llm, "bind_tools") else llm
        self._calibrator = TokenCalibrator()
        self._request_estimate = 0
        self._request_limit = 0
        self._cache_request_kind = "initial"

    # the port

    def estimate_request(self, request: ModelRequest) -> int:
        raw = estimate_messages(self._messages(request), self._schemas)
        return max(raw, self._calibrator.estimate(raw))

    def respond(self, request: ModelRequest) -> ModelTurn:
        check_active(reserve=MODEL_TIME_RESERVE)
        self.rounds += 1
        self._cache_request_kind = ("forced_final" if request.force_terminal else
                                    "initial" if self.rounds == 1 else "continuation")
        self._publish_new_blocks(request.transcript)

        try:
            messages = self._messages(request)
        except Exception as error:  # pragma: no cover - defensive; see class docstring
            return self._failed("could not build the request", error)

        self._request_estimate = estimate_messages(messages, self._schemas)
        self._request_limit = request.input_tokens_limit
        measured = max(self._request_estimate, self._calibrator.estimate(self._request_estimate))
        if request.input_tokens_limit and measured > request.input_tokens_limit:
            return self._failed("context budget exceeded", ValueError(
                f"estimated {measured:,} input tokens, limit {request.input_tokens_limit:,}"))

        model = (terminal_model(self.llm, self._schemas, self._bound, provider=self.provider)
                 if request.force_terminal else self._bound)
        options = {}
        if self.provider == "OpenRouter":
            extra = dict(getattr(self.llm, "extra_body", None) or {})
            session_id = extra.get("session_id") or self.request_name or request.plan.session_id
            if session_id:
                options["extra_body"] = {**extra, "session_id": session_id}
        check_active(reserve=MODEL_TIME_RESERVE)
        try:
            reply = model.invoke(messages, max_tokens=request.max_tokens, **options)
        except TypeError:
            # Not every LangChain provider accepts a per-call max_tokens.
            try:
                reply = model.invoke(messages, **options)
            except Exception as error:
                return self._failed("the model call failed", error)
        except Exception as error:
            return self._failed("the model call failed", error)

        check_active()
        return self._turn(reply)

    # request

    def _messages(self, request: ModelRequest) -> list:
        """System blocks, then the conversation — in that order.

        Retrieval and memory snapshots are recorded in history rather than
        sent as a trailing message, so message-end cache entries stay reusable.
        """
        messages: list = [self._system(request)]
        markers = {}
        if _takes_cache_control(self.model_id):
            for marker in (request.plan.previous_marker, request.plan.marker):
                if marker is not None:
                    markers[marker.index] = marker.ttl
        messages.extend(_replay(request.transcript, markers))
        return native_cache_messages(messages, self.provider)

    def _system(self, request: ModelRequest):
        """The system blocks, with a cache breakpoint where the plan asks for one.

        Anthropic uses ``cache_control``; GPT-5.6+ uses explicit system
        boundaries alongside implicit conversation caching. Older OpenAI and
        DeepSeek models cache stable prefixes automatically. The choice is made
        on the model id, so Anthropic models reached through OpenRouter still
        get ``cache_control``.
        """
        blocks = [block for block in request.plan.blocks if not block.is_empty]
        if not blocks:
            return SystemMessage(content=ROLE_PROMPT)
        openai = openai_breakpoints(self.provider, self.model_id)
        if not _takes_cache_control(self.model_id) and not openai:
            return SystemMessage(content="\n\n".join(block.text for block in blocks))

        content = []
        for block in blocks:
            part = {"type": "text", "text": block.text}
            if block.breakpoint and block.ttl != "none":
                if openai:
                    # Keep implicit caching for the rolling conversation; at
                    # most two explicit system boundaries plus its latest one.
                    part["prompt_cache_breakpoint"] = {"mode": "explicit"}
                else:
                    part["cache_control"] = {"type": "ephemeral", "ttl": block.ttl}
            content.append(part)
        return SystemMessage(content=content)

    # reply

    def _turn(self, reply) -> ModelTurn:
        usage = _usage(reply)
        cost = provider_cost(reply)
        actual_input = usage.input_tokens + usage.cache_read + usage.cache_write
        self._calibrator = self._calibrator.observe(self._request_estimate, actual_input)
        self.total_tokens += usage.observed
        _persist_tokens(
            self.request_name,
            self.total_tokens,
            input_tokens=actual_input,
            cache_read_tokens=usage.cache_read,
            cache_write_tokens=usage.cache_write,
            cost_delta=cost,
        )

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
            cache_phase="understanding",
            cache_request_kind=self._cache_request_kind,
            cache_read_ratio=round(usage.cache_read / max(1, actual_input), 4),
            cache_new_input_tokens=max(0, actual_input - usage.cache_read),
            provider_cost=cost,
            context_input_tokens=actual_input,
            context_estimated_tokens=self._request_estimate,
            input_budget_tokens=self._request_limit,
            upstream_provider=(getattr(reply, "response_metadata", None) or {}).get("upstream_provider"),
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
            stopped_at_limit=recovery.hit_output_limit(reply),
            provider_context_json=(json.dumps({key: reply.additional_kwargs[key]
                for key in ('reasoning_details', 'reasoning') if key in getattr(reply, 'additional_kwargs', {})})
                if any(key in getattr(reply, 'additional_kwargs', {}) for key in ('reasoning_details', 'reasoning')) else ''),
        )

    def _failed(self, detail: str, error: Exception) -> ModelTurn:
        self.failure = f"{detail}: {error}"
        log_agent_error(
            "Koda core: chat model",
            f"request={self.request_name}\n{detail}: {error}\n{frappe.get_traceback()}",
        )
        return ModelTurn(text=f"[{detail}: {error}]", failed=True, detail=f"{detail}: {error}")

    # progress

    def _publish_new_blocks(self, transcript) -> None:
        """Stable call IDs keep progress correct after history shrinks."""
        for block in transcript.blocks:
            if block.kind not in ("tool_use", "tool_result"):
                continue
            key = (block.kind, block.call_id)
            if key in self._published:
                continue
            self._published.add(key)
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


def _replay(transcript, markers=None) -> list:
    """The transcript as LangChain messages, pairs kept together.

    A ``tool_use`` block becomes an ``AIMessage`` carrying one tool call, and its
    ``tool_result`` becomes the matching ``ToolMessage``. The arguments come from
    the block's ``arguments_json`` rather than from a driver-side memo: a memo
    works right up until the worker restarts, and a resumed session would then
    replay its tool calls with no arguments at all.

    Rounds carrying provider reasoning retain their original parallel call group.
    Legacy transcripts without that metadata keep their one-call message shape.
    """
    messages: list = []
    grouped = set()
    calls_by_id = {block.call_id: block for block in transcript.blocks if block.kind == 'tool_use'}
    for index, block in enumerate(transcript.blocks):
        before = len(messages)
        if block.kind == "tool_use":
            if block.call_id in grouped:
                continue
            call_blocks = [block]
            provider_context = {}
            if block.provider_context_json and block.parallel_call_ids and all(i in calls_by_id for i in block.parallel_call_ids):
                call_blocks = [calls_by_id[i] for i in block.parallel_call_ids]
                grouped.update(block.parallel_call_ids)
                provider_context = json.loads(block.provider_context_json)
            messages.append(AIMessage(content='', tool_calls=[{
                'name': call.tool, 'args': _arguments(call), 'id': call.call_id,
            } for call in call_blocks], additional_kwargs=provider_context))
        elif block.is_result:
            messages.append(ToolMessage(content=block.text or "(no output)",
                                        tool_call_id=block.call_id))
        elif block.text:
            messages.append(
                HumanMessage(content=block.text) if block.role == "user"
                else AIMessage(content=block.text)
            )
        if markers and index in markers and len(messages) > before:
            messages[-1] = mark_message(messages[-1], markers[index])
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

    Built from ``CATALOGUE`` and nowhere else. Parameters are strings except the
    ``NUMERIC_PARAMETERS``; the description carries the real contract, including the caps.
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
NUMERIC_PARAMETERS = frozenset({"start", "end", "offset"})


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

    Cache reads and writes are pulled out of the input count, so re-sending the
    cached prefix each round is not charged at full price.
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


# Seam 2 — the utility model


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
        check_active(reserve=MODEL_TIME_RESERVE)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        try:
            reply = self.llm.invoke(messages, max_tokens=max_tokens)
        except Exception as error:
            log_agent_error(
                "Koda core: utility model",
                f"request={self.request_name}\n{error}\n{frappe.get_traceback()}",
            )
            return Completion(failed=True, detail=str(error))

        check_active()
        usage = _usage(reply)
        return Completion(
            text=_text_of(reply)[: max_tokens * 8],
            usage=SideUsage(
                calls=1,
                input_tokens=usage.input_tokens + usage.cache_read + usage.cache_write,
                output_tokens=usage.output_tokens,
            ),
        )


# Seam 3 — the tool host


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


# Shared retrieval for a task the user is writing by hand


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
# Model-free retrieval for a task the user is writing by hand
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _rerank_client(site: str, api_key: str, model: str, timeout: float) -> OpenRouterReranker:
    """Keep result caches separate by site, credentials, model and deadline."""
    return OpenRouterReranker(api_key=api_key, model=model, timeout_seconds=timeout)


def _with_reranker(session: Session) -> Session:
    """Inject the service at the host boundary; the core imports no HTTP client."""
    config = session.context.config.rerank
    client = None
    if config.enabled:
        site = str(getattr(getattr(frappe, "local", None), "site", "") or "")
        try:
            settings = frappe.get_single("Agent Settings")
            api_key = settings.get_password("openrouter_api_key") or ""
        except Exception:
            # Outside a Frappe site, the standard environment key supports
            # scripts. A site never borrows another site's process-global key.
            api_key = "" if site else os.environ.get("OPENROUTER_API_KEY", "")
        if api_key.strip():
            client = _rerank_client(site, api_key.strip(), config.model, config.timeout_seconds)
    return replace(session, retriever=replace(session.retriever, reranker=client))


def _without_reranker(session: Session) -> Session:
    """The session with local ranking only: no paid reranking request."""
    if getattr(session.retriever, "reranker", None) is None:
        return session
    return replace(session, retriever=replace(session.retriever, reranker=None))


#: One read-only session per app root, reused while the checkout is unchanged; held without
#: the reranker, which is attached per call.
_APP_SESSIONS: dict[str, tuple[str, Session]] = {}


def _tree_key(app_name: str) -> str:
    """HEAD plus a content hash of every uncommitted change, tracked or not.

    ``git status`` would not do: a file that is already modified keeps the same
    status line as it changes again, and the session would go stale.
    """
    repo_root = get_repo_root(app_name)
    ok, head = run_git(["rev-parse", "HEAD"], cwd=repo_root)
    return f"{head.strip() if ok else ''}:{worktree_signature(repo_root)}"


def remember_app_session(app_name: str, session: Session) -> None:
    """Hold a session for searches while the checkout is unchanged."""
    try:
        entry = (_tree_key(app_name), _without_reranker(session))
    except Exception:
        return  # not a git checkout, or git unavailable: fall back to cold start
    with _SESSIONS_LOCK:
        _APP_SESSIONS[_app_root(app_name)] = entry


def _app_session(app_name: str, *, rerank: bool = True) -> Session:
    root = _app_root(app_name)
    key = _tree_key(app_name)
    with _SESSIONS_LOCK:
        held = _APP_SESSIONS.get(root)
    if held is not None and held[0] == key:
        return _with_reranker(held[1]) if rerank else _without_reranker(held[1])
    session = open_session(LocalWorkspace(root_path=Path(root)))
    with _SESSIONS_LOCK:
        _APP_SESSIONS[root] = (key, session)
    return _with_reranker(session) if rerank else session


def suggest_context(app_name: str, query: str, *, limit: int = 12, rerank: bool = True) -> list[dict]:
    """Rank code spans for a task description with the retriever planning used.

    No chat call. Cold start on a large app takes tens of seconds, which is why
    the API runs this in a job and streams the result back over realtime.
    ``rerank=False`` skips the paid reranker, whose spend the cost ledger cannot see.
    """
    return suggestions_for(_app_session(app_name, rerank=rerank), query, limit=limit)


def suggestions_for(session: Session, query: str, *, limit: int = 12) -> list[dict]:
    """Translate search hits into the shape a ``context_refs`` entry needs.

    This is the working set's retrieved tier — the same ``search`` call and the
    same location dedup ``working_set_for`` does before the model's first round —
    kept as structured hits rather than its rendered ``path:start-end`` lines.
    """
    query = plain_query(query)
    index = session.retriever.index
    seen: set[tuple[str, int, int]] = set()
    out: list[dict] = []
    limit = max(1, min(limit, 50))
    for hit in search(session.retriever, query, limit=min(50, limit * 3)).hits:
        chunk = hit.chunk
        definition = _enclosing_definition(index, chunk.path, chunk.span.start, chunk.span.end, chunk.identity)
        start, end = (definition.extent.start, definition.extent.end) if definition else (chunk.span.start, chunk.span.end)
        if (chunk.path, start, end) in seen:
            continue
        seen.add((chunk.path, start, end))
        out.append({
            "path": chunk.path, "start": start, "end": end,
            "symbol": definition.qualified_name if definition else (chunk.identity or ""),
            "snippet": excerpt(chunk.body, query), "score": round(hit.score, 3), "note": hit.note,
        })
        if len(out) >= limit:
            break
    return out


def _enclosing_definition(index, path: str, start: int, end: int, identity: str = ""):
    """The definition a chunk belongs to, so the ref names the function.

    A symbol chunk knows its own name; use it, because the chunk often starts a
    few lines above the definition (its comment) and so is not *inside* it —
    containment alone would climb to the enclosing class and cite 900 lines.
    Plain line windows fall back to the smallest definition containing them.
    """
    analysis = index.files.get(path)
    if analysis is None:
        return None
    if identity:
        named = next((d for d in analysis.definitions if d.qualified_name == identity), None)
        if named is not None:
            return named
    best = None
    for definition in analysis.definitions:
        extent = definition.extent
        if extent.start <= start and end <= extent.end:
            if best is None or (extent.end - extent.start) < (best.extent.end - best.extent.start):
                best = definition
    if best is not None and best.extent.end - best.extent.start > MAX_WIDEN_LINES:
        return None  # A window inside a 900-line class is better cited as the window.
    return best


MAX_WIDEN_LINES = 200


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
    # The index this turn just built is exactly what a task-suggestion search
    # needs; keeping it under the app key saves the next click a cold start.
    remember_app_session(app_name, result.session)
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

    Only the *family* in the string matters here: this table determines how
    wide a block has to be before caching it pays, and a
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
    "gpt-6": 1_050_000,  # luna, sol, astra (OpenRouter model list, 2026-09)
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
    """Investigations use question-specific retrieval instead of a global map."""
    window = _window_for(model_id)
    context = {"map_tokens": 0}
    if window:
        context["window_tokens"] = window
    return {"context": context}


def request_limits(app_name: str, model_id: str) -> tuple[int, int]:
    """Resolve the same window and input ceiling for planning and execution."""
    if app_name:
        workspace = LocalWorkspace(root_path=Path(_app_root(app_name)))
        config = _resolve_config(workspace, _overrides(model_id), [])
    else:
        config = merge_config(_overrides(model_id))
    return config.context.window_tokens, config.context.input_tokens


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
    for entry in session.ledger.entries:
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


_persist_tokens = persist_usage  # module-level seam that tests replace
