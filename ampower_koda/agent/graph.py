# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# LangGraph workflows: Planning (Understand -> Plan) and Execution (Implement -> Review).
# Completion checks gate dependency progress; final integration reviews the full plan
# before bench and deploy run in executor.py.

import json
import os
import re as _re
from copy import deepcopy
from datetime import datetime

import frappe
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from ampower_koda.agent.prompt_caching import mark_message, native_cache_messages, openai_breakpoints, rolling_messages, terminal_model
from langchain_openai import ChatOpenAI

from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.state import AgentState
from ampower_koda.agent.cache_usage import persist_usage, provider_cost
from ampower_koda.agent import koda_core
from ampower_koda.agent import checkpoint
from ampower_koda.agent import recovery
from ampower_koda.agent import verification
from ampower_koda.agent import python_diagnostics
from ampower_koda.agent.advisor import directive as advisor_directive
from ampower_koda.agent.run_control import (
    check_active, set_request_value, MODEL_TIMEOUT_SECONDS, MODEL_MAX_RETRIES, MODEL_TIME_RESERVE,
)
from ampower_koda.agent.core.budget.calibrator import TokenCalibrator
from ampower_koda.agent.core.budget.request import cleanup_target, estimate_messages, input_limit, serialized_tokens
from ampower_koda.agent.core.constants import DEFAULT_WINDOW_TOKENS
from langchain_core.utils.function_calling import convert_to_openai_tool
from ampower_koda.agent import tools as agent_tools
from ampower_koda.agent.plan_contract import (
    MAX_PLAN_TASKS,
    PLAN_JSON_SCHEMA,
    PlanValidationError,
    plan_to_markdown,
    validate_plan,
)
from ampower_koda.agent.checks import run_health_checks, run_query_schema_checks, run_task_checks, CheckResult
from ampower_koda.agent.execution_contract import (
    load_plan, read_snapshot, change_evidence, review_verdict, review_decision, completion_report, revision,
)
from ampower_koda.agent.execution_evidence import source_context, SourceMemory
from ampower_koda.agent.prompts import (
    get_system_prompt,
    get_understand_system_prompt,
    get_understand_prompt,
    get_plan_prompt,
    get_implement_prompt,
    get_follow_up_implement_prompt,
    get_review_prompt,
)


from ampower_koda.agent.history_prune import describe_call, prune_price, prune_rounds
MAX_TOOL_ROUNDS_EXECUTION = 18
MAX_TOOL_ROUNDS_REPAIR = 8        # a retry continues from current source with a remaining-work list
MAX_TOOL_ROUNDS_REVIEW = 6
MAX_TOOL_ROUNDS_REVIEW_RECOVERY = 4  # continues the first pass's history, so these are new reads only
MAX_REVIEW_ATTEMPTS = 2           # per task and for final integration
BASE_EXECUTION_CALL_BUDGET = 18
# A task must be able to afford one full implementation turn and one repair
# turn, each with its forced final call. The old value of 10 left a one-task
# plan 11 rounds for the first attempt and none for the retry.
PER_TASK_CALL_BUDGET = (MAX_TOOL_ROUNDS_EXECUTION + 1) + (MAX_TOOL_ROUNDS_REPAIR + 1)
FINAL_REVIEW_RESERVE = 16  # review (7), evidence recovery (5), repair/re-review minimum (4)
REPAIR_REVIEW_RESERVE = 3
# Direct spend fences complement call-count limits. The request ledger includes
# understanding/planning too, so execution cannot ignore cost already incurred
# before it started. A new explicitly scoped follow-up gets a fresh budget.
BASE_REQUEST_NEW_INPUT_BUDGET = 100_000
PER_TASK_REQUEST_NEW_INPUT_BUDGET = 50_000
MAX_REQUEST_NEW_INPUT_BUDGET = 250_000
MAX_REQUEST_PROVIDER_COST_USD = 0.15
WRITE_TOOLS = {"edit_file", "write_file", "copy_file", "rename_file", "delete_file"}
REPLAYABLE_TOOLS = {
    "find_files", "list_directory", "read_file", "search_code", "find_code",
    "read_doctype_schema", "get_file_outline", "validate_code",
}
TOOL_FAILURE_PREFIXES = (
    "Error:", "Tool error:", "Unknown tool:", "Not a file:", "Not a directory:", "Not a file or directory:",
    "DocType schema not found", "VALIDATION_ERROR", "VALIDATION_FAILED", "READ_FAILED",
    "WRITE_FAILED", "EDIT_FAILED", "COPY_FAILED", "RENAME_FAILED", "DELETE_FAILED",
    "TESTS_FAILED", "SYNTAX_ERROR",
    "VALIDATION_UNAVAILABLE", "CALL_FAILED", "RUNTIME_UNAVAILABLE",
    "FIND_FAILED", "SEARCH_FAILED", "SUBMIT_FAILED", "EXPLORE_FAILED", "EXPLORE_UNAVAILABLE",
)
#: Tools whose failures say what to fix in the arguments, not a cause in the code.
CAUSE_GUARD_EXEMPT = {"submit_plan", "explore"}

# Per-phase output stored in conversation_log. High so full phase text is retained
# (phase outputs are LLM summaries and are naturally well under this in practice).
MAX_PHASE_OUTPUT_CHARS = 60000

# Full-input pressure is the only trigger for rewriting retained tool history.
# A "round" is one assistant tool-call message plus all of its tool results.
MIN_KEEP_ROUNDS = 1           # latest call/result pair must survive into the next request
COMPACT_RESULT_PREVIEW = 140  # chars of each tool result kept in the compact summary
MAX_COMPACTED_HISTORY_CHARS = 12000
MAX_TASK_PROMPT_CHARS = 60000
MAX_TOOL_RESULT_CHARS = 8000
# A whole source file is read once (about 25k tokens at most) rather than in slices.
MAX_READ_RESULT_CHARS = 80000
# Searches and outlines bound themselves and say what they left out; the generic
# cap would cut them without saying so.
SELF_BOUNDED_TOOLS = {"read_file", "search_code", "get_file_outline"}
FIND_CODE_HITS = 8
FIND_CODE_EXCERPT_CHARS = 160
# A result fetched with a purpose is read by a bare model call and the conversation
# gets only its answer, so the full result is not re-sent with every later request.
READER_TOOLS = frozenset({"read_file", "search_code", "get_file_outline", "call_method"})
READER_MIN_CHARS = 1500  # shorter results cost less than the call that would read them
READER_INPUT_CHARS = MAX_READ_RESULT_CHARS  # what the helper reads of one result, a whole read at most
READER_OUTPUT_TOKENS = 4000  # the answer plus low-effort reasoning
READER_SYSTEM = (
    "You read one tool result for a coding agent that will not see it, and give it what it needs. "
    "Answer the agent's purpose from this text alone. Cite each finding as path:line or path:start-end, "
    "and copy identifiers exactly: function and method names, parameters, response keys, field names, "
    "CSS classes, routes and messages. Quote code only where a line or two is itself the answer. Say what "
    "the text covers of the purpose and what it does not. If nothing in it bears on the purpose, reply in "
    "one line: \"UNRELATED: <what this text is>\". Never guess beyond the text. At most 25 lines.")
# Mid-turn, a stale message is retired only where the rewrite is cheap: the
# rewrite uncaches everything after it, so only messages with at most ~8k
# tokens after them qualify, once ~1k tokens are free; further back, only a ~20k bulk saving.
PRUNE_TAIL_CHARS = 8000 * 3.3
PRUNE_TAIL_MIN_CHARS = 1000 * 3.3
PRUNE_BULK_CHARS = 20000 * 3.3
# Prompt-cache prices as multiples of uncached input. A rewrite re-caches what
# follows it at the write price, so below context pressure a prune rarely pays.
CACHE_READ_PRICE = {"openai/": 0.1, "anthropic/": 0.1, "deepseek/": 0.1, "google/": 0.25}
DEFAULT_CACHE_READ_PRICE = 0.3
CACHE_WRITE_PRICE = 1.25
MAX_UNDERSTANDING_CONTEXT_CHARS = 8000
# A copy at least this long changes by edits: rewriting it whole drops what the reference does.
COPY_REWRITE_MIN_LINES = 150
# Room to write a whole file in one call, with the reasoning that precedes it.
# Output is billed as generated, so a high ceiling costs nothing unused.
MODEL_ROUND_OUTPUT_TOKENS = 32000
MODEL_FINAL_OUTPUT_TOKENS = 8192
# Reasoning budget for ALWAYS_REASONING models: half the smallest per-call cap, so every call keeps room.
MODEL_REASONING_BUDGET_TOKENS = MODEL_FINAL_OUTPUT_TOKENS // 2
# OpenRouter families that reason on every call (GLM 4.5+, DeepSeek R1, QwQ, "thinking" variants).
ALWAYS_REASONING = _re.compile(r"glm-(?:4\.[5-9]|[5-9])|deepseek-r1|qwq|thinking|reasoner", _re.I)
# Sized for the largest valid plan plus its overview and scope.
PLAN_OUTPUT_TOKENS_PER_TASK = 450
MODEL_PLAN_OUTPUT_TOKENS = 1000 + MAX_PLAN_TASKS * PLAN_OUTPUT_TOKENS_PER_TASK
# OpenRouter counts hidden reasoning inside ``max_tokens``; reserve room for it
# on top of the visible plan so the answer still fits.
OPENROUTER_REASONING_ALLOWANCE_TOKENS = 16000


# Provider-native prompt caching for the stable system prefix (safe no-op when unsupported).
ENABLE_PROMPT_CACHE = True

DOCTYPE_NAME = "Agent Request"

# Appended to the understanding prompt. That prompt is configurable per request
# ("Understand Prompt"), so a site's customized copy still names the old explore
# tools — and a model told to call find_files() calls it and gets nothing back.
# Stating the mapping is cheaper than migrating every customized prompt, and it
# is correct for the ones nobody customized too.
CORE_TOOL_NOTE = """

## TOOLS AVAILABLE IN THIS PHASE
`search` (ranked — start here) · `grep` (literal) · `glob` (paths) · `outline`
(signatures, no bodies — PREFER over read) · `symbols` · `refs` · `definition`
(pass `path` when a name is shadowed) · `read` (a `path`, or a `symbol` to resolve;
optional `start`/`end`) · `explore` · `recall` (a ledger id such as L7) ·
`read_doctype_schema`.

If the instructions above name a different tool, use these instead:
find_files / list_directory -> glob, search_code -> search, read_file -> read,
get_file_outline -> outline.

There is no shell and no editing in this phase. Describe the change; do not make it.
Results you have already seen may be replaced by a pointer like [search "x" -> L14];
that is not a loss — call `recall` with the id to get the full text back.
"""


def _message_content_to_str(content) -> str:
    """Normalize AIMessage content to plain text (Responses API returns list blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif block.get("type") == "refusal" and block.get("refusal"):
                    parts.append(str(block["refusal"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _llm_response_text(response) -> str:
    """Extract plain text from a LangChain chat model response."""
    return _message_content_to_str(getattr(response, "content", ""))


def _parse_review_verdict(text: str, criteria: list[str] | None = None) -> tuple[bool, str]:
    return review_verdict(_extract_review_json(text or ""), criteria or [])


def _extract_review_json(text: str) -> dict | None:
    """Pull a {"review_passed": ...} object out of the model's response.

    Tries, in order: the whole response as JSON, a ```json fenced block,
    then the last {...} object found anywhere in the text — models
    sometimes add a sentence of prose before or after the JSON despite
    being asked not to.
    """
    candidates = []

    stripped = text.strip()
    candidates.append(stripped)

    fence_match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    brace_match = _re.search(r"\{[^{}]*\"review_passed\"[^{}]*\}", text, _re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and "review_passed" in parsed:
            return parsed
    return None
    

# ---------------------------------------------------------------------------
# LLM factory — supports OpenAI, OpenRouter, Gemini and Claude
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class _OpenRouterChat(ChatOpenAI):
    """ChatOpenAI that keeps OpenRouter's extensions.

    It replays reasoning blocks and tool-result cache markers that ChatOpenAI's
    serializer drops, and records the serving upstream (``provider``) in the
    message metadata, ``llm_output`` and the LangSmith run, since the prompt
    cache lives with that upstream. Routing is left to ``session_id`` stickiness.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # ChatOpenAI's OpenAI serializer drops OpenRouter's provider reasoning
        # fields. Replay the complete opaque blocks unchanged with their turn.
        for message, original in zip(payload.get('messages', []), self._convert_input(input_).to_messages()):
            if isinstance(original, AIMessage):
                for key in ('reasoning_details', 'reasoning'):
                    if key in original.additional_kwargs:
                        message[key] = deepcopy(original.additional_kwargs[key])
        # ChatOpenAI sanitizes tool-result content to OpenAI's schema, which
        # removes Anthropic cache_control. OpenRouter accepts that extension.
        if _uses_explicit_prompt_cache("OpenRouter", self.model_name):
            marked = {
                message.tool_call_id: message.content
                for message in self._convert_input(input_).to_messages()
                if isinstance(message, ToolMessage) and isinstance(message.content, list)
            }
            for message in payload.get("messages", []):
                original = marked.get(message.get("tool_call_id"))
                if message.get("role") != "tool" or not original or not isinstance(message.get("content"), list):
                    continue
                texts = iter(p for p in original if isinstance(p, dict) and p.get("type") == "text")
                restored = []
                for part in message["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        source = next(texts, {})
                        if source.get("text") == part.get("text") and "cache_control" in source:
                            part = dict(part, cache_control=source["cache_control"])
                    restored.append(part)
                message["content"] = restored
        return payload

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        try:
            raw = response if isinstance(response, dict) else response.model_dump(warnings=False)
            for generation, choice in zip(result.generations, raw.get('choices', [])):
                message = choice.get('message') or {}
                if message.get('reasoning_details'):
                    generation.message.additional_kwargs['reasoning_details'] = deepcopy(message['reasoning_details'])
                elif message.get('reasoning') or message.get('reasoning_content'):
                    generation.message.additional_kwargs['reasoning'] = message.get('reasoning') or message['reasoning_content']
            provider_name = raw.get("provider")
            if provider_name:
                result.llm_output = dict(result.llm_output or {}, upstream_provider=provider_name)
                for generation in result.generations:
                    generation.message.response_metadata["upstream_provider"] = provider_name
        except Exception:
            log_agent_error("Agent LLM: openrouter provider", frappe.get_traceback())
        return result

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        _tag_llm_run(run_manager, (result.llm_output or {}).get("upstream_provider"))
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        _tag_llm_run(run_manager, (result.llm_output or {}).get("upstream_provider"))
        return result


def _tag_llm_run(run_manager, provider_name) -> None:
    """Put ``upstream_provider`` on the LangSmith run for this very llm call.

    Found via the run manager (``get_current_run_tree()`` is the parent chain);
    the run is still open here, so the metadata goes out with its final update.
    """
    if not provider_name or run_manager is None:
        return
    try:
        from langchain_core.tracers.langchain import LangChainTracer
        for handler in getattr(run_manager, "handlers", []):
            if isinstance(handler, LangChainTracer):
                run = handler.run_map.get(str(run_manager.run_id))
                if run is not None:
                    run.extra.setdefault("metadata", {})["upstream_provider"] = provider_name
    except Exception:
        pass


def _configured_reasoning_effort(provider: str, model: str) -> str | None:
    # These OpenAI reasoning families support high effort through either route.
    # Other models retain their provider defaults; do not send unsupported fields.
    family = model.rsplit('/', 1)[-1].lower()
    if provider not in {'OpenAI', 'OpenRouter'} or not _re.match(r'^(gpt-[56](?:[.\-]|$)|o[134](?:[\-]|$))', family):
        return None
    try:
        setting = frappe.db.get_single_value('Agent Settings', 'reasoning_effort')
    except Exception:
        setting = None  # installations not yet migrated still use the default
    setting = str(setting or '')
    if setting.lower() == 'provider default':
        return None
    return str(setting) if setting in {'low', 'medium', 'high'} else 'high'


def _get_llm(provider: str = "OpenAI", model: str = "gpt-4o-mini", session_id: str = "",
             reasoning_effort: str | None = None):
    """Build the chat model for the given provider and model.

    ``session_id`` (the request name) pins an OpenRouter conversation to one upstream,
    where its prompt cache lives; ``provider.order`` is not set because it disables
    that sticky routing. Other providers ignore it.
    """
    provider = (provider or "OpenAI").strip()
    model = (model or "gpt-4o-mini").strip()
    effort = _configured_reasoning_effort(provider, model)
    if effort and reasoning_effort:
        effort = reasoning_effort  # only where the model takes an effort at all
    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Google's retry setting counts attempts, including the initial call.
        return ChatGoogleGenerativeAI(model=model, temperature=0, timeout=MODEL_TIMEOUT_SECONDS,
                                      max_retries=MODEL_MAX_RETRIES + 1)
    if provider == "Claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, timeout=MODEL_TIMEOUT_SECONDS, max_retries=MODEL_MAX_RETRIES)
    if provider == "OpenRouter":
        # OpenRouter speaks the OpenAI wire format, so ChatOpenAI drives it — only
        # the base URL and key differ. `usage.include` asks OpenRouter to return the
        # real billed cost of each call in the usage block; without it there is no
        # cost field at all. Never enable the Responses API here: OpenRouter only
        # implements /v1/chat/completions.
        extra_body = {"usage": {"include": True}}
        if openai_breakpoints(provider, model):
            # A conversation continues minutes later (a review after a repair pass);
            # under the default in-memory retention its cache would have expired.
            extra_body["prompt_cache_options"] = {"mode": "implicit", "ttl": "30m"}
        if effort:
            extra_body['reasoning'] = {'effort': effort}
            if openai_breakpoints(provider, model):
                extra_body['reasoning']['context'] = 'all_turns'
        elif ALWAYS_REASONING.search(model):
            # Always-reasoning models need a limit or can spend the whole cap thinking.
            # Others get no reasoning field: it would switch a hybrid model's thinking on.
            extra_body['reasoning'] = ({'effort': reasoning_effort} if reasoning_effort
                                       else {'max_tokens': MODEL_REASONING_BUDGET_TOKENS})
        if session_id:
            extra_body["session_id"] = str(session_id)[:256]
        # Namespaced IDs bypass ChatOpenAI's temperature guard. Let the routed
        # model use its default: forcing temperature=0 excludes reasoning models
        # such as Luna when planning requires support for every parameter.
        return _OpenRouterChat(
            name="ChatOpenAI",
            model=model,
            temperature=None,
            timeout=MODEL_TIMEOUT_SECONDS, max_retries=MODEL_MAX_RETRIES,
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY") or "",
            extra_body=extra_body,
        )
    if provider not in ("OpenAI", "Gemini", "Claude", "OpenRouter"):
        log_agent_error(
            "Agent LLM Warning",
            f"Unknown AI provider '{provider}', defaulting to OpenAI",
        )
    # Direct OpenAI always goes through /v1/responses; it serves every model
    # chat-completions serves, so there is nothing to discriminate on and no
    # per-model allowlist to keep current. OpenRouter returns above precisely
    # because it is the one route that cannot do this.
    direct_options = {}
    if effort:
        direct_options['reasoning'] = {'effort': effort}
        if openai_breakpoints('OpenAI', model):
            direct_options['reasoning']['context'] = 'all_turns'
    if session_id:
        # Earlier OpenAI models use this stable key to improve cache routing;
        # GPT-5.6+ uses it only for per-request cache accounting.
        direct_options["extra_body"] = {"prompt_cache_key": str(session_id)[:256]}
    if openai_breakpoints("OpenAI", model):
        # Keep the rolling message-end boundary alongside our explicit stable
        # system/shared-context boundaries, with the documented minimum TTL.
        direct_options["prompt_cache_options"] = {"mode": "implicit", "ttl": "30m"}
    return ChatOpenAI(model=model, temperature=None if effort else 0, use_responses_api=True,
                      timeout=MODEL_TIMEOUT_SECONDS, max_retries=MODEL_MAX_RETRIES,
                      **direct_options)


#: Open-keyed object parameters travel as JSON text, since strict tool schemas cannot express them.
#: An object sent anyway is serialized before the tool runs; null becomes empty text.
JSON_TEXT_ARGUMENTS = frozenset({"arguments", "replacements"})


def _json_object_argument(value, name: str) -> tuple[dict | None, str]:
    """Parse a JSON-object tool argument; return (object, "") or (None, why not)."""
    if not str(value or "").strip():
        return {}, ""
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as error:
        return None, f"{name} must be a JSON object string, e.g. '{{\"key\": \"value\"}}' ({error})."
    if not isinstance(parsed, dict):
        return None, f"{name} must be a JSON object, not {type(parsed).__name__}."
    return parsed, ""


#: Failures of one tool on one target with one cause, since the last write, before
#: another call is refused; catches arguments varied around an unfixable cause.
FAILURE_CAUSE_LIMIT = 2

#: Consecutive one-edit responses before the model is told what they cost.
SINGLE_EDIT_STREAK = 3
MAX_BATCH_NUDGES = 2
BATCH_EDITS_TEXT = (
    "Your last {rounds} responses each made one edit. Every response re-sends this whole conversation"
    "{context}, so one edit per response pays that once per edit. Send every edit you already know, to "
    "this file and any other, together in your next response; read first only what those edits need."
)


def _failure_target(name: str, arguments: dict) -> str:
    """What a call acts on: its method or path, so varied arguments count together."""
    return f"{name}:{arguments.get('method') or arguments.get('path') or ''}"


#: Parts of an error line that vary between two runs of the same failure.
_VOLATILE = _re.compile(r"\bline \d+|0x[0-9a-fA-F]+|\d{4}-\d\d-\d\d[ T][\d:.]+|\b\d+(?:\.\d+)?\s*(?:ms|s|seconds)\b")


def _failure_cause(result: str) -> str:
    """A failure's cause: the exception line of a traceback, else its first line.

    Line numbers, addresses, timestamps and timings are masked; other numbers stay.
    """
    lines = [line.strip() for line in str(result or "").splitlines() if line.strip()]
    errors = [line for line in lines if _re.match(r"^[A-Za-z_.]*(Error|Exception)\b.*:", line)]
    line = (errors[-1] if errors else lines[0] if lines else "")[:200]
    return _VOLATILE.sub("N", line)


def _cache_read_price(provider: str, model: str) -> float:
    """A cached input token's price as a fraction of an uncached one."""
    if provider in ("OpenAI", "Claude"):
        return 0.1
    if provider == "Gemini":
        return 0.25
    name = (model or "").lower()
    return next((price for prefix, price in CACHE_READ_PRICE.items() if name.startswith(prefix)),
                DEFAULT_CACHE_READ_PRICE)


def _prune_pays(freed_chars: float, rewritten_chars: float, calls_left: int, read_price: float) -> bool:
    """Whether retiring ``freed_chars`` saves more than re-caching what follows the edit."""
    return freed_chars * read_price * calls_left >= rewritten_chars * (CACHE_WRITE_PRICE - read_price)


def _uses_explicit_prompt_cache(provider: str, model: str = "") -> bool:
    """Whether this route needs Anthropic-style cache breakpoints.

    Model identity matters as well as the direct provider: Claude reached via
    OpenRouter still needs the same marker.
    """
    provider_name = (provider or "").strip().lower()
    model_name = (model or "").strip().lower()
    return (
        provider_name == "claude"
        or "claude" in model_name
        or "anthropic/" in model_name
    )


def _cacheable_content(text: str) -> list[dict]:
    return [{
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }]


def _build_system_message(provider: str, system_prompt: str, model: str = "") -> SystemMessage:
    """Build the system message, adding provider-native prompt caching where supported.

    Anthropic supports an explicit `cache_control` breakpoint on the system block,
    which caches the large, stable instruction prefix (big cost saver on repeated
    tool rounds). GPT-5.6+ also gets an explicit system boundary. Older OpenAI
    models and Gemini/others fall back to a plain system message.
    Caching only reduces cost; it never changes model output.
    """
    if ENABLE_PROMPT_CACHE and openai_breakpoints(provider, model):
        return mark_message(SystemMessage(content=system_prompt), openai=True)
    if ENABLE_PROMPT_CACHE and _uses_explicit_prompt_cache(provider, model):
        try:
            return SystemMessage(content=_cacheable_content(system_prompt))
        except Exception:
            log_agent_error(
                "Agent Graph: anthropic cache system message",
                frappe.get_traceback(),
            )
    return SystemMessage(content=system_prompt)


def _build_task_message(provider: str, task_prompt: str, model: str = "") -> HumanMessage:
    """Cache the stable task body too; it is often much larger than system text."""
    if ENABLE_PROMPT_CACHE and openai_breakpoints(provider, model):
        return mark_message(HumanMessage(content=task_prompt), openai=True)
    if ENABLE_PROMPT_CACHE and _uses_explicit_prompt_cache(provider, model):
        try:
            return HumanMessage(content=_cacheable_content(task_prompt))
        except Exception:
            log_agent_error(
                "Agent Graph: anthropic cache task message",
                frappe.get_traceback(),
            )
    return HumanMessage(content=task_prompt)


def _bounded_text(text: str, limit: int) -> str:
    """Keep the actionable beginning and instructions at the end of long prompts."""
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    marker = f"\n\n... ({len(text) - limit:,} middle characters omitted for context budget) ...\n\n"
    available = max(0, limit - len(marker))
    head = int(available * 0.65)
    tail = available - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def _tool_call_key(name: str, arguments: dict) -> str:
    """Stable identity for a replayable call, normalizing omitted defaults."""
    normalized = {
        key: value for key, value in (arguments or {}).items()
        if value not in (None, "", [], {})
    }
    if name == "read_file":
        for key in ("start_line", "end_line"):
            if normalized.get(key) == 0:
                normalized.pop(key)
    return f"{name}\0{json.dumps(normalized, sort_keys=True, separators=(',', ':'), default=repr)}"


def _tool_result_succeeded(result: str) -> bool:
    """Classify every public tool failure prefix consistently."""
    return not result.startswith(TOOL_FAILURE_PREFIXES)


def _invoke_limited(model, messages: list, max_tokens: int):
    """Apply an output ceiling where the provider supports a per-call limit."""
    check_active(reserve=MODEL_TIME_RESERVE)
    try:
        result = model.invoke(messages, max_tokens=max_tokens)
    except TypeError:
        check_active(reserve=MODEL_TIME_RESERVE)
        result = model.invoke(messages)
    check_active()
    return result


def _extract_file_paths(text: str) -> list[str]:
    """Extract app-relative file paths from text produced by understand/plan phases."""
    patterns = [
        r'(?:[a-zA-Z_][a-zA-Z0-9_]*/[a-zA-Z0-9_/]+\.(?:py|js|json|html|css))',
        r'(?:hooks\.py|setup\.py|__init__\.py)',
        r'(?:patches/[a-zA-Z0-9_/]+\.py)',
        r'(?:public/[a-zA-Z0-9_/]+\.(?:js|css))',
    ]
    paths = set()
    for pat in patterns:
        for m in _re.finditer(pat, text):
            paths.add(m.group(0))
    return sorted(paths)


def tool_reader(provider: str, model: str, request_name: str = ""):
    """The bare call that reads a purpose read for the model: low effort, no tools, no conversation."""
    llm = _get_llm(provider=provider, model=model, session_id=request_name, reasoning_effort="low")
    return lambda messages: _invoke_limited(llm, messages, READER_OUTPUT_TOKENS)


#: Tokens that calls outside a running tool loop (explore, the plan check) charged to a
#: request. The loop adds them to its own total, so its next write does not undercount.
_SIDE_TOKENS: dict[str, int] = {}


def add_side_tokens(request_name: str, tokens: int) -> None:
    if request_name and tokens > 0:
        _SIDE_TOKENS[request_name] = _SIDE_TOKENS.get(request_name, 0) + int(tokens)


def _output_share(window: int, wanted: int) -> int:
    """An output cap that leaves the window room for a working prompt: at most a quarter of it."""
    return max(recovery.MIN_OUTPUT_TOKENS, min(int(wanted), int(window) // 4))


def _reader_replaces(purpose: str, reader, result: str) -> bool:
    """Whether a tool result goes to the reader and the model gets only its answer."""
    return (bool(str(purpose or "").strip()) and reader is not None and _tool_result_succeeded(result)
            and len(result) >= READER_MIN_CHARS)


# Helpers

def _app_file_exists(app_name: str, rel_path: str) -> bool:
    """True if rel_path resolves to a real file inside the app (best-effort)."""
    try:
        return os.path.isfile(agent_tools._resolve_path(app_name, rel_path))
    except Exception:
        return False


def _extract_change_summary(text: str) -> str:
    """Pull the plain-English 'SUMMARY OF CHANGES' the model writes at the end.

    Falls back to the trailing prose of the final message if the explicit heading
    is missing, so the request always shows a readable summary.
    """
    text = (text or "").strip()
    if not text:
        return ""
    m = _re.search(r"(?is)summary of changes\s*:?\s*(.+)$", text)
    summary = (m.group(1) if m else text).strip()
    # Drop leftover markdown fences / list bullets noise but keep readable text.
    summary = summary.strip("`").strip()
    return summary[:4000]


def _file_manifest(paths: list[str], max_files: int = 25) -> str:
    """Name likely files without injecting their bodies into every model round.

    The old preloader inserted up to 120k characters and the implementation
    prompt then told the model to call ``read_file`` for those same files. A
    manifest preserves routing while one targeted tool read supplies the bytes.
    """
    unique = list(dict.fromkeys(path for path in paths if path))
    shown = unique[:max_files]
    lines = [f"- {path}" for path in shown]
    if len(unique) > len(shown):
        lines.append(f"- ... ({len(unique) - len(shown)} additional paths omitted)")
    return "\n".join(lines) if lines else "(no target files identified yet)"


def _log_stage(state: dict, stage: str, status: str, summary: str) -> list:
    """Append a stage log entry and persist to DB + realtime."""
    logs = list(state.get("stage_log") or [])
    entry = {
        "stage": stage,
        "status": status,
        "summary": summary[:500],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    logs.append(entry)

    request_name = state.get("request_name")
    if request_name:
        try:
            stage_text = "\n".join(
                f"[{l['timestamp']}] {l['stage']} - {l['status']}: {l['summary']}"
                for l in logs
            )
            set_request_value(request_name, {
                "status": stage if status == "started" else state.get("current_stage", stage),
                "stage_log": stage_text[:50000],
            })
            frappe.db.commit()

            user = frappe.db.get_value(DOCTYPE_NAME, request_name, "owner") or "Administrator"
            check_active()
            frappe.publish_realtime("agent_progress", {
                "request_name": request_name,
                "status": stage,
                "stage": stage,
                "stage_status": status,
                "message": summary[:200],
            }, user=user)
        except Exception:
            log_agent_error(
                "Agent Graph: stage log persist",
                f"request={request_name}\nstage={stage}\n{frappe.get_traceback()}",
            )

    return logs


_persist_token_usage = persist_usage  # module-level seam that tests replace


def _publish_agent_log(request_name: str, log_type: str, **kwargs):
    """Publish a detailed agent_log realtime event."""
    if not request_name:
        return
    check_active()
    try:
        user = frappe.db.get_value(DOCTYPE_NAME, request_name, "owner") or "Administrator"
        payload = {
            "request_name": request_name,
            "type": log_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs,
        }
        frappe.publish_realtime("agent_log", payload, user=user)
    except Exception:
        log_agent_error(
            "Agent Graph: publish agent_log",
            f"request={request_name}\ntype={log_type}\n{frappe.get_traceback()}",
        )


def _request_spend_budget(state: dict) -> tuple[int, float]:
    """Fresh-input and provider-cost ceilings for one complete request."""
    tasks = max(1, len(state.get("execution_tasks") or []))
    fresh = min(
        MAX_REQUEST_NEW_INPUT_BUDGET,
        BASE_REQUEST_NEW_INPUT_BUDGET + PER_TASK_REQUEST_NEW_INPUT_BUDGET * tasks,
    )
    return fresh, MAX_REQUEST_PROVIDER_COST_USD


def _request_spend_pressure(state: dict, request_name: str) -> str:
    """Describe overspend without terminating a recoverable workflow.

    This is intentionally based on uncached input and the provider's reported
    bill, not total context tokens. Cache reads can make total input look huge
    without costing the same; conversely, a cold oversized prompt must count.
    The caller uses this as a telemetry/compaction signal, never as an error.
    """
    if not request_name:
        return ""
    try:
        row = frappe.db.get_value(
            DOCTYPE_NAME,
            request_name,
            ["cache_input_tokens", "cache_read_tokens", "cost_estimate"],
            as_dict=True,
        ) or {}
    except (AttributeError, TypeError):
        # Minimal/offline Frappe stubs and pre-migration sites have no readable
        # ledger. Existing call-count valves still apply there.
        return ""
    input_used = int((row.get("cache_input_tokens", 0) if isinstance(row, dict)
                      else getattr(row, "cache_input_tokens", 0)) or 0)
    cache_read = int((row.get("cache_read_tokens", 0) if isinstance(row, dict)
                      else getattr(row, "cache_read_tokens", 0)) or 0)
    fresh_used = max(0, input_used - cache_read)
    cost_used = float((row.get("cost_estimate", 0) if isinstance(row, dict)
                       else getattr(row, "cost_estimate", 0)) or 0)
    fresh_limit, cost_limit = _request_spend_budget(state)
    reasons = []
    if fresh_used >= fresh_limit:
        reasons.append(f"{fresh_used:,} new input tokens reached the {fresh_limit:,} limit")
    if cost_used >= cost_limit:
        reasons.append(f"provider cost ${cost_used:.6f} reached the ${cost_limit:.2f} limit")
    return "; ".join(reasons)


def _make_tools(app_name: str, read_only: bool = False, *, before=None, file_moves=None,
                delete_paths=(), verification_contract=None, verification_observer=None, copied_files=None,
                session=None, reader=None):
    """Build LangChain tools bound to a specific app_name.

    The catalogue is the same in every phase (tools precede messages in cache keys),
    so read-only safety is enforced here, not by hiding schemas.

    Writes are bounded by the app root only. A task's ``files`` list tells the
    implementer where to start; it is not a permission list, so a change the
    task genuinely needs in a neighbouring file does not have to round-trip
    through a plan amendment. Deletion stays gated on ``delete_paths`` because
    it is the one irreversible operation here.
    """

    observed = {}
    copied = copied_files if copied_files is not None else {}  # destination -> reference, across passes

    def capture_write(path, *, fresh_read=False):
        check_active(reserve=5)
        full = agent_tools._resolve_path(app_name, path)
        # _resolve_path resolves symlinks; measure against the resolved root too,
        # or a junctioned app root turns every written path into "../../...".
        canonical = os.path.relpath(full, os.path.realpath(agent_tools._app_root(app_name))).replace("\\", "/")
        if verification_contract is not None and (
            canonical == verification.CONFIG_PATH or canonical in verification_contract.get("existing_tests", {})
            or canonical in (verification_contract.get("frozen_tests") or {})
        ):
            raise ValueError(f"{canonical} belongs to the frozen verification contract. "
                             "Repair the implementation or add a new regression test instead of weakening this check.")
        current = read_snapshot(full)
        if fresh_read and current is not None and (full not in observed or observed[full] != current):
            raise ValueError(f"Read {path} again before editing: line anchors or file content may be stale.")
        if before is not None and canonical not in before:
            before[canonical] = current
        active_journal = checkpoint.journal()
        if active_journal:
            baseline = dict(active_journal.state.get("execution_baseline") or {})
            for key, content in (before or {}).items():
                baseline.setdefault(key, content)
            active_journal.update(execution_baseline=baseline)
        return full, canonical, current

    @tool
    def find_files(pattern: str = "", max_depth: int = 6) -> str:
        """List the app's files. Without a pattern: an indented tree (max_depth defaults to 6).
        With a glob ('*.py', 'doctype/**/*.json', '*.{py,js}'): the matching paths, one per line.
        Use only when the supplied task context does not identify the files you need."""
        return agent_tools.find_files(app_name, pattern, max_depth)

    @tool
    def list_directory(path: str) -> str:
        """List files and directories at path (relative to app root).
        Directories are prefixed with [DIR] in the output."""
        return agent_tools.list_directory(app_name, path)

    @tool
    def read_file(path: str, start_line: int = 0, end_line: int = 0, ranges: str = "", purpose: str = "") -> str:
        """Read a file (path relative to app root) with line numbers.
        With purpose ("what get_totals returns and where it stops"), a helper reads the text and you get
        only its answer, with line numbers: use it to understand. Without purpose you get the exact text,
        which stays in the conversation: use it for lines you will copy, edit or quote.
        A code file of 200+ lines comes back as a summary (signatures kept, long bodies elided);
        then fetch every span you need in ONE call with ranges="40-80,120-160" (ranges="1-N" reads it all).
        Edit_file receipts show the edited region."""
        full = agent_tools._resolve_path(app_name, path)
        snapshot = read_snapshot(full)
        result = agent_tools.read_file(app_name, path, start_line, end_line, ranges)
        # A summary shows signatures, not the text an overwrite or edit replaces, and
        # a result the helper reads reaches the model only as the helper's answer.
        if (snapshot is not None and " summary of " not in result.split("\n", 1)[0]
                and not _reader_replaces(purpose, reader, result)):
            observed[full] = snapshot
        return result

    @tool
    def search_code(pattern: str, path: str = "", glob: str = "", output_mode: str = "", context: int = 2,
                    head_limit: int = 0, purpose: str = "") -> str:
        """Search source for a regular expression (case-insensitive; | alternation works).
        output_mode "files" (the default for a directory or the whole app) lists matching files with their
        match counts, most first; "content" (the default when path is one file) shows each matching line with
        `context` lines around it. path narrows to a file or directory; glob to names like "*.js" or
        "page/**/*.py". head_limit caps what is shown (50 files or 40 lines by default, 200 at most), and
        the result says how many more there are. purpose: a helper reads the matches and returns only
        what answers it."""
        return agent_tools.search_code(app_name, pattern, path, glob=glob, output_mode=output_mode,
                                       context=context, head_limit=head_limit)

    @tool
    def find_code(query: str) -> str:
        """Find where something is implemented from a plain description ("where an order's items are
        priced", "report filter rendering") when you do not know the names to search_code for. Returns the
        best-matching functions and blocks, ranked: path:start-end, the enclosing name and a short excerpt."""
        try:
            hits = koda_core.suggest_context(app_name, query, limit=FIND_CODE_HITS, rerank=False)
        except Exception:
            log_agent_error("Agent Graph: find_code", frappe.get_traceback())
            return "FIND_FAILED: the code index is unavailable; use search_code."
        if not hits:
            return f"No code matched: {query}. Try search_code with names or phrases from the UI."
        rows = []
        for hit in hits:
            excerpt = " ".join(str(hit.get("snippet") or "").split())
            if len(excerpt) > FIND_CODE_EXCERPT_CHARS:
                excerpt = excerpt[:FIND_CODE_EXCERPT_CHARS] + "…"
            symbol = f" {hit['symbol']}" if hit.get("symbol") else ""
            rows.append(f"{hit['path']}:{hit['start']}-{hit['end']}{symbol} — {excerpt}")
        return "\n".join(rows)

    @tool
    def read_doctype_schema(doctype_name: str) -> str:
        """Read a DocType's source schema or installed dependency metadata and database columns.
        Use this to verify actual ERPNext/Frappe fields before writing queries; doctype_name e.g. 'Sales Order'."""
        return agent_tools.read_doctype_schema(app_name, doctype_name)

    @tool
    def get_file_outline(path: str, purpose: str = "") -> str:
        """Get lightweight outline of a Python/JS/TS file — class/function signatures with line numbers.
        Much cheaper than reading the whole file. Use to understand structure before reading specific sections.
        purpose: a helper reads the outline and returns only the signatures that answer it."""
        return agent_tools.get_file_outline(app_name, path)

    @tool
    def validate_code(path: str) -> str:
        """Check syntax, undefined JavaScript names, Python type-checker errors and Query Builder fields against installed schema.
        Source edits already return these diagnostics. Call separately only when the current source has no receipt.
        Restore missing local declarations/imports; do not silence them as globals. Runtime tests are still needed."""
        outcome = agent_tools.validate_code(app_name, path)
        if outcome.startswith('VALID:') and path.endswith('.py'):
            schema = run_query_schema_checks(app_name, [path])
            if not schema.passed:
                return 'VALIDATION_FAILED: Query schema does not match the installed site.\n' + schema.summary()
            if schema.results:
                outcome += '\n' + schema.summary()
            outcome += type_check(path)
        return outcome

    @tool
    def run_tests() -> str:
        """Execute configured integration tests and .koda/tests behavioral tests.
        Python test*.py uses unittest against the live site; *.test.cjs/js/mjs uses node --test.
        Nothing a Python test does persists: database writes roll back, file writes, background jobs and
        email are discarded, and schema changes or commits are refused.
        Import the actual changed code, never patch the module under test, and repair
        failures before completion. A test that passes is frozen for the rest of the run.
        This tool never accepts shell commands."""
        contract = verification_contract if verification_contract is not None else verification.prepare_contract(app_name)
        active_journal = checkpoint.journal()
        changed = set(before or ()) | set((active_journal.state.get("execution_baseline") or {})
                                          if active_journal else ())
        report, receipts = verification.run_verification(app_name, contract, env=_get_bench_env(),
                                                         required=verification.needs_tests(changed))
        if verification_observer:
            verification_observer(report, receipts)
        return ("TESTS_PASSED\n" if report.passed else "TESTS_FAILED\n") + report.summary()

    # The parameter must not be called "kwargs": the tool schema renames that
    # field to "v__kwargs" and every call then fails before reaching the site.
    @tool
    def call_method(method: str, arguments: str = "", purpose: str = "") -> str:
        """Run a function of this app against the live site and return its result or traceback.
        method is the full dotted path (e.g. app.module.page.name.name.get_data); arguments is a JSON object
        string of its keyword arguments, e.g. '{"customer": "CUST-0001", "limit": 20}'. Runs as Administrator
        with real records and schema; nothing it does persists: database writes are rolled back, and file
        writes, background jobs and email are discarded (the result says which). Use it to see what a query
        or endpoint really returns before and after changing it. purpose ("does every row have a
        due date?"): a helper reads the whole result and returns only the answer."""
        kwargs, problem = _json_object_argument(arguments, "arguments")
        if problem:
            return "CALL_FAILED: " + problem
        return verification.call_method(app_name, method, kwargs, env=_get_bench_env(),
                                        limit=READER_INPUT_CHARS if purpose.strip() else verification.MAX_CALL_OUTPUT)

    def read_only_failure(prefix: str) -> str:
        if session is not None and session.get("plan_sink") is not None:
            return (f"{prefix}: Writes start after the user approves your plan. Keep investigating, "
                    "then call submit_plan.")
        return f"{prefix}: Tool unavailable during read-only review; inspect source and report findings instead."

    def remember(full, result):
        # The model knows what it just wrote: its own write or edit is a current
        # view of the file, so overwriting it later needs no re-read first.
        # An edit refreshes a view the model already had; it does not create one.
        if result.startswith("WRITE_OK:") or (result.startswith("EDIT_OK:") and full in observed):
            snapshot = read_snapshot(full)
            if snapshot is not None:
                observed[full] = snapshot
        return result

    def type_check(path):
        try:
            return python_diagnostics.receipt(agent_tools._app_root(app_name),
                                              agent_tools._resolve_path(app_name, path), app_name,
                                              env=agent_tools.command_environment())
        except Exception:
            log_agent_error("Agent Graph: pyright diagnostics", frappe.get_traceback())
            return ""

    def source_feedback(result, path):
        if result.startswith(('WRITE_OK:', 'EDIT_OK:', 'COPY_OK:')) and path.endswith(('.py', '.js')):
            # Report diagnostics in the mutation receipt, before another turn
            # can build on a broken edit. This does not claim runtime success.
            checked = agent_tools.validate_code(app_name, path)
            if checked.startswith('VALID:') and path.endswith('.py'):
                schema = run_query_schema_checks(app_name, [path])
                if schema.results:
                    checked += '\n' + schema.summary()
                checked += type_check(path)
            result += '\nAutomatic source checks (integration still required):\n' + checked
        return result

    @tool
    def write_file(path: str, content: str) -> str:
        """Write or overwrite a file at the given relative path. Parent directories are created if needed."""
        if read_only:
            return read_only_failure("WRITE_FAILED")
        full, canonical, current = capture_write(path, fresh_read=True)
        reference = copied.get(canonical)
        if reference and current is not None and current.count("\n") >= COPY_REWRITE_MIN_LINES:
            return (f"WRITE_FAILED: {path} was copied from {reference} so that it keeps the reference's "
                    "behavior. Change it with edit_file on the parts the task changes (several edits in one "
                    "response are fine); rewriting it whole drops what the reference does.")
        return source_feedback(remember(full, agent_tools.write_file(app_name, path, content)), path)

    @tool
    def copy_file(source_path: str, destination_path: str, replacements: str = "") -> str:
        """Copy a reference into a new file without regenerating its complete source.
        Optional replacements, a JSON object string of literal old->new text such as
        '{"sales_report": "purchase_report"}', rename identity strings such as page names,
        class names and CSS prefixes. The source is preserved and unrelated existing destinations are
        never overwritten. Prefer this when adapting a large existing feature; then edit only the
        requested behavior."""
        if read_only:
            return read_only_failure('COPY_FAILED')
        replacements, problem = _json_object_argument(replacements, "replacements")
        if problem:
            return "COPY_FAILED: " + problem
        source = agent_tools._resolve_path(app_name, source_path)
        known = observed.get(source)
        destination, canonical, _ = capture_write(destination_path)
        result = agent_tools.copy_file(app_name, source_path, destination_path,
                                      replacements=replacements, expected_sha256=revision(known) if known is not None else '')
        if result.startswith('COPY_OK:'):
            copied[canonical] = source_path
            if known is not None:
                observed[destination] = read_snapshot(destination)
                if not any("\n" in old + new for old, new in replacements.items()):
                    # Single-line replacements keep the copy line for line with its source.
                    result += (f" It matches {source_path} line for line, so what you read of the source is "
                               "this file with the replacements applied: edit it from that without reading it again.")
        return source_feedback(result, destination_path)

    @tool
    def edit_file(path: str, old_string: str, new_string: str, expected_occurrences: int = 1) -> str:
        """Replace EXACT text, including whitespace. Read current source first.
        Default requires a unique match. For an intentional global rename, set expected_occurrences
        to the exact current count; no write occurs if the count differs. Returns source diagnostics.
        Send every edit you already know in one response, several to one file included."""
        if read_only:
            return read_only_failure("EDIT_FAILED")
        full, _, _ = capture_write(path)
        return source_feedback(remember(full, agent_tools.edit_file(
            app_name, path, old_string, new_string, expected_occurrences)), path)

    @tool
    def rename_file(source_path: str, destination_path: str) -> str:
        """Rename a file without changing its bytes or overwriting an existing destination.
        Read the source first.
        Use this for filename fixes; writing an empty source does not remove it."""
        if read_only:
            return read_only_failure("RENAME_FAILED")
        full, source, current = capture_write(source_path, fresh_read=True)
        _, destination, destination_content = capture_write(destination_path)
        known = current if current is not None else observed.get(full)
        if known is None and before is not None and before.get(destination) is None:
            known = before.get(source)
        expected = revision(known) if known is not None else ""
        if known is not None:
            checkpoint.write_intent({source: current, destination: destination_content},
                {source: None, destination: known}, move={"source": source, "destination": destination, "sha256": expected})
        result = agent_tools.rename_file(app_name, source_path, destination_path, expected_sha256=expected)
        if result.startswith("RENAME_OK:") and file_moves is not None:
            move = {"source": source, "destination": destination, "sha256": expected}
            if move not in file_moves:
                file_moves.append(move)
            active_journal = checkpoint.journal()
            if active_journal:
                all_moves = list(active_journal.state.get("file_moves") or [])
                if move not in all_moves:
                    all_moves.append(move)
                active_journal.update(file_moves=all_moves)
        return result

    @tool
    def delete_file(path: str) -> str:
        """Remove an explicitly approved DELETE-task file. Read current source first."""
        if read_only:
            return read_only_failure("DELETE_FAILED")
        _, canonical, current = capture_write(path, fresh_read=True)
        if canonical not in delete_paths:
            raise ValueError(f"Deletion is not approved for {path}.")
        return agent_tools.delete_file(app_name, path, expected_sha256=revision(current))

    catalogue = [find_files, list_directory, read_file, search_code, find_code, read_doctype_schema,
                 get_file_outline, validate_code, run_tests, call_method, write_file, copy_file,
                 edit_file, rename_file, delete_file]
    return catalogue


# Tool-calling loop with detailed realtime logging

def _bounded_tool_result(name: str, arguments: dict, result: str) -> str:
    limit = MAX_READ_RESULT_CHARS if name in SELF_BOUNDED_TOOLS else MAX_TOOL_RESULT_CHARS
    if len(result) <= limit:
        return result
    # Several spans cannot be relabelled as one contiguous range.
    if name == 'read_file' and not _re.match(r'\[[^\]\n]*\] lines \d+-\d+,', result):
        complete = result[:limit].rsplit('\n', 1)[0]
        numbered = _re.findall(r'^\s*(\d+) \| .*$', complete, _re.M)
        total = _re.search(r'(?:of |\] )(\d+)(?: lines)?$', result.split('\n', 1)[0])
        if numbered and total:
            start, end = int(numbered[0]), int(numbered[-1])
            return (f'[{arguments.get("path", "")}] lines {start}-{end} of {total[1]}\n'
                    + complete.split('\n', 1)[1]
                    + f'\n[Output limit reached. Continue with read_file start_line={end + 1}; use a focused end_line.]')
    return result[:limit] + '\n[Output limit reached; request a narrower path or source range.]'


def _compact_round_summary(round_entry: dict) -> list[str]:
    """One short line per tool call in a round, for the compacted-history block."""
    return round_entry.get("summary", [])


def _run_tool_calling_loop(llm, tools, system_prompt: str, task_prompt: str,
                           request_name: str = "", max_rounds: int = 20,
                           state: dict = None, provider: str = "OpenAI",
                           require_writes: bool = False, progress: dict | None = None,
                           history: dict | None = None,
                           validate_final=None, shared_context: str = "",
                           cache_phase: str = "", advisor=None,
                           stop_when=None, reader=None, after_round=None) -> tuple[str, list[str], int, int, bool]:
    """
    Run a tool-calling loop, publishing every tool call and LLM response via realtime.

    ``stop_when()``, checked after each tool round, ends the turn without another model call.
    ``reader(messages)`` answers a call's ``purpose`` in place of its result; without it, results come whole.
    ``after_round()`` runs after each complete tool round.
    Returns (text, edited_paths, tokens, model_calls, exhausted).

    ``history`` is an optional mutable dict that carries the retained rounds,
    compacted summaries, duplicate-call bookkeeping and source memory from one
    loop into the next. A reviewer recovery pass continues from what the first
    pass already read instead of re-fetching it; round numbers keep counting.

    Context control:
      - The stable system prompt is a separate, cache-friendly message.
      - Stale results are pruned only when the saving repays the cache rewrite.
        At the full-input limit, older rounds are summarized in one batch.

    A "round" is one assistant tool-call message plus all of its tool results;
    rounds are trimmed atomically so no tool_call_id is ever left dangling.

    Implementation gets one focus reminder per turn after a long read-only streak.
    No-op completion claims are left to the independent reviewer to assess.

    If max_rounds is reached without the LLM stopping, one final llm.invoke() is
    fired with the same tool definitions and tool_choice="none".

    Output caps grow on their own: a reply cut off by ``max_tokens`` is resent
    with more room until it fits or the context window has no more to give
    (``recovery.invoke_growing``). ``validate_final(text)`` returns why the
    turn's last reply is not the report the phase needs, or ""; when it is
    given, an unusable report is re-asked in place with that exact reason
    instead of being handed to a whole new implementation attempt.
    """
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    # Final calls and re-asks keep the tool schemas (part of the cached prefix) and
    # forbid calling them instead of unbinding them.
    final_tools_bound = bool(tools)
    llm_final = terminal_model(llm, tools, llm_with_tools, provider=provider)
    schemas = [convert_to_openai_tool(tool) for tool in tools]
    context_window, input_ceiling = koda_core.request_limits(
        (state or {}).get("target_app_name", ""), koda_core._model_id(llm))
    calibrator = history.get("calibrator", TokenCalibrator()) if history else TokenCalibrator()
    evidence_chars = history.get("evidence_chars", 12_000) if history else 12_000

    model_id = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "")
    # The task prompt is not capped here; the serialized input budget governs it.
    task_prompt = task_prompt or ""
    system_msg = _build_system_message(provider, system_prompt, model_id)
    # Keep two fixed boundaries, leaving room for previous/current rolling ones.
    shared_msg = _build_task_message(provider, shared_context, model_id) if shared_context else None
    # OpenAI has no rolling boundaries, so the task (identical across repair
    # passes) gets its own.
    task_msg = (HumanMessage(content=task_prompt)
                if shared_msg is not None and not openai_breakpoints(provider, model_id)
                else _build_task_message(provider, task_prompt, model_id))

    if history is None:
        history = {}
    rounds: list[dict] = history.setdefault("rounds", [])      # each: {"number", "ai", "tools", "summary"}
    compacted: list[str] = history.setdefault("compacted", [])  # summary lines for dropped (older) rounds
    followups: list[dict] = history.setdefault("followups", [])  # see _queue_followup
    edited_paths: list[str] = []
    if state and state.get("execution_tasks"):
        source_memory = history.get("source_memory") or SourceMemory(lambda path: _read_current(state, path))
        history["source_memory"] = source_memory
    else:
        source_memory = None
    seen_calls: dict[str, int] = history.setdefault("seen_calls", {})
    seen_results: dict[str, int] = history.setdefault("seen_results", {})
    failed_calls: dict[str, dict] = history.setdefault("failed_calls", {})
    failure_causes: dict[str, dict] = history.setdefault("failure_causes", {})  # see _failure_cause
    progress_generation = int(history.get("tool_progress_generation", 0))
    round_base = int(history.get("rounds_done", 0))  # rounds already numbered by earlier loops
    total_tokens = (state or {}).get("tokens_used", 0)  # carry forward from prior phases
    usage_missing_logged = False  # warn once per loop, not once per round

    read_only_streak = 0
    write_nudges = 0
    READ_STREAK_LIMIT = 5
    # One edit per response re-sends the whole conversation per edit. Twice per turn at most.
    single_edit_streak = 0
    batch_nudges = 0
    NUDGE_TEXT = (
        "Work toward the plan's acceptance criteria. Read any missing context "
        "needed for a correct edit, then apply and validate a focused change. "
        "If the task is blocked or already satisfied, explain the evidence instead "
        "of making an unnecessary edit. Do not guess anchors or field names."
    )
    overthought = 0
    OVERTHINK_NUDGES = 2
    OVERTHINK_TEXT = (
        "Your last reply spent its whole output budget on reasoning and produced no tool call. "
        "Stop deliberating. Make the next concrete tool call now: "
        + ("write or edit the first file the task needs, in pieces if it is large. Settle an open design "
           "question with the most reasonable choice and record it in your final report's unverified list "
           "instead of resolving it in thought." if require_writes else
           "read or run the one thing your next conclusion needs, or give your answer with what you "
           "already know.")
    )

    def build_messages():
        msgs = [system_msg, *([shared_msg] if shared_msg is not None else []), task_msg]
        if compacted:
            msgs.append(HumanMessage(content=(
                "## Earlier tool activity (older rounds, summarized to save context)\n"
                "These tools already ran; re-read a file only if you need details not captured here.\n"
                + "\n".join(compacted)
            )))
        if source_memory:
            evidence = source_memory.render({r["number"] for r in rounds}, max_chars=evidence_chars)
            if evidence:
                msgs.append(HumanMessage(content="## Retained source evidence (current revision)\n" + evidence))
        present = {r["number"] for r in rounds}
        # A follow-up whose round was trimmed away still has to precede every
        # later round, so it goes right after the summaries instead.
        for f in followups:
            if f["after"] not in present:
                msgs.extend(f["messages"])
        for r in rounds:
            msgs.append(r["ai"])
            msgs.extend(r["tools"])
            for f in followups:
                if f["after"] == r["number"]:
                    msgs.extend(f["messages"])
        return msgs

    def message_chars(message) -> int:
        size = len(str(getattr(message, "content", "")))
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            size += len(json.dumps(calls, sort_keys=True, default=repr))
        return size

    def estimate_chars() -> int:
        return sum(message_chars(m) for r in rounds for m in [r["ai"], *r["tools"]])

    def cap_compacted() -> None:
        while compacted and sum(len(line) + 1 for line in compacted) > MAX_COMPACTED_HISTORY_CHARS:
            compacted.pop(0)

    def prune(reason: str, *, force: bool, calls_made: int = 0) -> None:
        """Retire stale results, applied write bodies and old reasoning in one rewrite.

        Any change to a retained message costs one uncached pass over what
        follows it, so mid-turn this rewrites only near the end of the history
        (``PRUNE_TAIL_CHARS``) unless the whole history holds a bulk saving
        (``PRUNE_BULK_CHARS``), and only when the expected saving repays re-caching
        at this model's prices. A full context forces it.
        Old reasoning waits for a forced prune: it is cheap in tokens and
        stripping it mid-turn rewrote the cache for almost nothing.
        """
        if not rounds:
            return
        scope = None
        if not force:
            calls_left = min(calls_made, max_rounds - calls_made)
            if calls_left <= 0:
                return  # no call left to repay a rewrite
            read_price = _cache_read_price(provider, model_id)
            bulk, bulk_rewrite = prune_price(rounds, WRITE_TOOLS, reasoning=False, followups=followups)
            if bulk >= PRUNE_BULK_CHARS and _prune_pays(bulk, bulk_rewrite, calls_left, read_price):
                scope = "all"
            else:
                tail, tail_rewrite = prune_price(rounds, WRITE_TOOLS, reasoning=False,
                                                 tail_chars=PRUNE_TAIL_CHARS, followups=followups)
                if not (tail >= PRUNE_TAIL_MIN_CHARS and _prune_pays(tail, tail_rewrite, calls_left, read_price)):
                    return
                scope = "tail"
        freed = prune_rounds(rounds, WRITE_TOOLS, reasoning=force,
                             tail_chars=PRUNE_TAIL_CHARS if scope == "tail" else None)
        if freed:
            _publish_agent_log(request_name, "history_pruned", reason=reason,
                               freed_chars=freed, kept_rounds=len(rounds))

    def charge(reply) -> int:
        """Charge a helper call's reply (reader, advisor) to the request: tokens, cache and cost."""
        nonlocal total_tokens
        usage = getattr(reply, "usage_metadata", None) or {}
        if not usage:
            return 0
        tokens = int(usage.get("total_tokens") or 0)
        total_tokens += tokens + _SIDE_TOKENS.pop(request_name, 0)
        details = usage.get("input_token_details") or {}
        _persist_token_usage(request_name, total_tokens, input_tokens=int(usage.get("input_tokens") or 0),
                             cache_read_tokens=int(details.get("cache_read") or 0),
                             cache_write_tokens=int(details.get("cache_creation") or 0),
                             cost_delta=provider_cost(reply))
        if progress is not None:
            progress["tokens"] = total_tokens
        return tokens

    def read_for(purpose: str, name: str, arguments: dict, raw: str, label) -> str:
        """The bare call's answer to ``purpose`` in place of ``raw``; ``raw`` itself if it cannot answer."""
        what = f"{name}({describe_call(name, arguments)})"
        try:
            reply = reader([SystemMessage(content=READER_SYSTEM),
                            HumanMessage(content=f"Purpose: {purpose}\n\nResult of {what}:\n{raw}")])
        except Exception:
            log_agent_error("Agent Graph: tool reader", f"request={request_name}\n{frappe.get_traceback()}")
            return raw
        tokens = charge(reply)
        answer = _llm_response_text(reply).strip()
        _publish_agent_log(request_name, "tool_reader", round=label, tool_name=name, purpose=purpose[:200],
                           result_chars=len(raw), answer_chars=len(answer), tokens=tokens)
        if not answer:
            return raw
        return (f"[{what}, read for: {purpose[:200]}]\n{answer}\n"
                f"[A helper read {len(raw):,} characters for this answer; call without purpose for the exact text.]")

    def maybe_trim(extra=(), with_tools=True, output_tokens=None):
        nonlocal evidence_chars
        limit = input_limit(context_window, output_tokens or round_cap.value, input_ceiling)

        def current():
            return [*build_messages(), *extra]

        def size():
            raw = estimate_messages(current(), schemas if with_tools else ())
            return max(raw, calibrator.estimate(raw))

        before = size()
        if before <= limit:
            return current()

        # Stale results go first; they are useless anyway. Only if the live
        # history alone is still too large are whole rounds summarized, and
        # then deeply, so the next compaction (a near-total cache miss) is rare.
        prune("context_limit", force=True)
        if size() <= cleanup_target(limit):
            return current()
        target = min(cleanup_target(limit), limit // 2)
        cap_compacted()
        while len(rounds) > MIN_KEEP_ROUNDS and size() > target:
            compacted.extend(_compact_round_summary(rounds.pop(0)))
            cap_compacted()
        while compacted and size() > target:
            compacted.pop(0)
        if size() > target and evidence_chars:
            evidence_chars = max(0, evidence_chars - int((size() - target) * 3.6) - 256)
            history["evidence_chars"] = evidence_chars
        after = size()
        # A result removed from the prompt must be readable again.
        seen_calls.clear()
        seen_results.clear()
        if after > limit:
            raise ValueError(f"Context budget exceeded: estimated {after:,} input tokens, limit {limit:,}. "
                             "The task instructions and latest tool exchange cannot be reduced safely.")
        _publish_agent_log(request_name, "history_trim",
            kept_rounds=len(rounds),
            compacted_lines=len(compacted),
            retained_chars=estimate_chars(),
            input_before=before,
            input_after=after,
            input_limit=limit,
            input_target=target,
        )
        return current()

    def account_tokens(response, round_label, context_chars: int = 0, estimated_tokens: int = 0):
        nonlocal total_tokens, usage_missing_logged, calibrator
        total_tokens += _SIDE_TOKENS.pop(request_name, 0)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            context_input = int(usage.get("input_tokens") or 0)
            calibrator = calibrator.observe(estimated_tokens, context_input)
            history["calibrator"] = calibrator
            total_tokens += int(usage.get("total_tokens") or 0)
            details = usage.get("input_token_details") or {}
            cache_read = int(details.get("cache_read") or 0)
            cache_write = int(details.get("cache_creation") or 0)
            history["previous_context_input_tokens"] = context_input
            cost = provider_cost(response)
            uncached_input = max(
                0, context_input - cache_read - cache_write
            )
            _publish_agent_log(request_name, "token_usage",
                round=round_label,
                tokens_this_round=usage.get("total_tokens", 0),
                tokens_total=total_tokens,
                input_tokens=uncached_input,
                output_tokens=usage.get("output_tokens", 0),
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                cache_phase=cache_phase or ("implement" if require_writes else "review"),
                cache_request_kind=cache_request_kind,
                cache_read_ratio=round(cache_read / max(1, context_input), 4),
                cache_new_input_tokens=max(0, context_input - cache_read),
                provider_cost=cost,
                context_chars=context_chars,
                context_input_tokens=context_input,
                context_estimated_tokens=estimated_tokens,
                input_budget_tokens=input_limit(context_window, round_cap.value, input_ceiling),
                upstream_provider=(getattr(response, "response_metadata", None) or {}).get("upstream_provider"),
            )
            _persist_token_usage(
                request_name,
                total_tokens,
                input_tokens=context_input,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                cost_delta=cost,
            )
        elif not usage_missing_logged:
            # A provider that omits usage_metadata omits it every round, so this
            # is a property of the provider, not of this round. Report it once;
            # every later round would repeat the same fact.
            usage_missing_logged = True
            _publish_agent_log(request_name, "token_usage_missing",
                round=round_label,
                tokens_total=total_tokens,
                provider=provider,
            )
        if progress is not None:
            progress["tokens"] = total_tokens

    # Grown caps are carried as plain ints so a retained history stays a
    # plain dict; the ceiling is the room the window leaves after the prompt.
    output_room = recovery.output_ceiling(context_window, input_ceiling)
    round_tokens = _output_share(context_window, MODEL_ROUND_OUTPUT_TOKENS)
    final_tokens = _output_share(context_window, MODEL_FINAL_OUTPUT_TOKENS)
    round_cap = recovery.OutputCap(history.get("round_cap_tokens", round_tokens), max(round_tokens, output_room))
    final_cap = recovery.OutputCap(history.get("final_cap_tokens", final_tokens), max(final_tokens, output_room))
    last_truncated = False
    cache_request_kind = "initial"
    # A new turn on retained history: the previous turn's edits made many of
    # its reads stale, and every one of them would ride along all turn long.
    # Retiring them rewrites the whole retained history, though, so it runs only
    # when the saving over a typical turn repays that; context pressure still forces it.
    prune("turn_start", force=False, calls_made=max_rounds // 2)

    def invoke_growing(model, messages, cap, label, estimated, context_chars):
        """One model call that resends itself with more output room when cut off."""
        nonlocal last_truncated, cache_request_kind
        cache_request_kind = ("forced_final" if model is llm_final else
                              "continuation" if history.get("cache_calls", 0) else "initial")
        messages = rolling_messages(messages, history,
            enabled=ENABLE_PROMPT_CACHE and _uses_explicit_prompt_cache(provider, model_id))
        messages = native_cache_messages(messages, provider)

        def note(previous, grown, reasoning):
            _publish_agent_log(request_name, "output_cap_grown", round=label,
                               previous_cap=previous, cap=grown, reasoning_tokens=reasoning)

        def call_once(max_tokens):
            # Every wire request, retries and re-asks included, checks the
            # deadline and the request's spend first.
            nonlocal evidence_chars
            check_active(reserve=MODEL_TIME_RESERVE)
            pressure = _request_spend_pressure(state or {}, request_name)
            if pressure and not history.get("request_spend_pressure_logged"):
                history["request_spend_pressure_logged"] = True
                evidence_chars = min(evidence_chars, 4_000)
                history["evidence_chars"] = evidence_chars
                _publish_agent_log(
                    request_name,
                    "request_spend_pressure",
                    detail=pressure,
                    action="continue with compact retained evidence; do not terminate the workflow",
                )
            history["cache_calls"] = history.get("cache_calls", 0) + 1
            return _invoke_limited(model, messages, max_tokens)

        outcome = recovery.invoke_growing(call_once, cap, on_retry=note)
        for wasted in outcome.wasted:
            account_tokens(wasted, label, context_chars, estimated)
        account_tokens(outcome.reply, label, context_chars, estimated)
        history["round_cap_tokens"] = round_cap.value
        history["final_cap_tokens"] = final_cap.value
        last_truncated = outcome.truncated
        if outcome.truncated:
            _publish_agent_log(request_name, "output_cap_exhausted", round=label, cap=cap.value,
                               reasoning_tokens=outcome.reasoning_tokens)
        return outcome.reply

    def finish(text, extra, label):
        """Re-ask in place while the final reply is not the report the phase needs.

        The rejected reply goes back as the assistant turn it was, followed by
        the exact rule it broke, on the same conversation: the retained rounds
        and the prompt cache are kept. A reply cut off at the model's own
        limit is not re-asked; more asking cannot make more room.
        """
        if validate_final is None or last_truncated:
            return text

        def resend(previous, problem):
            reask = HumanMessage(content=(
                "REPORT REJECTED: " + problem + " Tools are not available. "
                "Reply with only the required JSON report and nothing else."))
            messages = maybe_trim(extra=(*extra, AIMessage(content=previous or "(no output)"), reask),
                                  with_tools=final_tools_bound, output_tokens=final_cap.value)
            reply = invoke_growing(llm_final, messages, final_cap, label,
                                   estimate_messages(messages, schemas if final_tools_bound else ()),
                                   sum(message_chars(m) for m in messages))
            return _llm_response_text(reply)

        fixed, problem, reasks = recovery.repair_output(text, validate_final, resend)
        if reasks:
            _publish_agent_log(request_name, "report_repaired" if not problem else "report_unrepaired",
                               round=label, reasks=reasks, problem=problem[:300])
        return fixed

    for round_num in range(max_rounds):
        check_active(reserve=MODEL_TIME_RESERVE)
        label = round_base + round_num + 1
        messages = maybe_trim(output_tokens=round_cap.value)
        context_chars = sum(message_chars(m) for m in messages)
        if progress is not None:
            progress["calls"] = round_num + 1
        # One save per round, before the call, so a run resumed after a crash still counts it.
        checkpoint.update(tool_rounds_used=(state or {}).get("tool_rounds_used", 0) + round_num + 1,
                          tokens_used=total_tokens)
        response = invoke_growing(llm_with_tools, messages, round_cap, label,
                                  estimate_messages(messages, schemas), context_chars)

        response_text = _llm_response_text(response)
        if response_text:
            _publish_agent_log(request_name, "llm_response",
                preview=response_text[:4000],
                round=label,
            )

        invalid_calls = getattr(response, "invalid_tool_calls", None) or []
        if invalid_calls and not getattr(response, "tool_calls", None):
            # Arguments that did not parse. Truncation is already handled by the
            # growing cap, so this is a malformed call: say so and let the model
            # send it again, instead of reading "no tool calls" as "finished".
            names = ", ".join(str(c.get("name") or "?") for c in invalid_calls)
            _publish_agent_log(request_name, "invalid_tool_call", round=label, tools=names)
            _queue_followup(
                history, f"Invalid tool call emitted for: {names}",
                f"Your last tool call ({names}) could not be parsed and was not executed. Send it again as a "
                "valid tool call, with every string argument properly escaped JSON.",
            )
            continue
        if (not getattr(response, "tool_calls", None) and last_truncated and not response_text.strip()
                and round_num + 1 < max_rounds and overthought < OVERTHINK_NUDGES):
            # The whole output went to reasoning: nothing to execute, nothing to report.
            # Asking for the next concrete step keeps the pass and its cache.
            overthought += 1
            _publish_agent_log(request_name, "reasoning_exhausted", round=label, attempt=overthought)
            _queue_directive(history, OVERTHINK_TEXT)
            continue
        if not getattr(response, "tool_calls", None):
            # No-op claims are assessed by review; never force an unnecessary edit.
            problem = validate_final(response_text) if (validate_final and not last_truncated) else ""
            if problem and round_num + 1 < max_rounds:
                # Prose instead of the report while calls remain: the turn is not
                # over. Say what is missing and keep the tools available, rather
                # than ending the turn and re-asking with the tools gone.
                _publish_agent_log(request_name, "report_rejected", round=label, problem=problem[:300])
                _queue_followup(
                    history, response_text,
                    "REPORT REJECTED: " + problem + " If the task is unfinished, continue with tools; "
                    "otherwise reply with only the JSON report.",
                )
                continue
            history["rounds_done"] = label
            return finish(response_text, (), label), edited_paths, total_tokens, round_num + 1, False

        round_entry = {"number": label, "ai": response, "tools": [], "summary": []}
        round_had_write = False
        for tc in response.tool_calls:
            check_active(reserve=5)
            _publish_agent_log(request_name, "tool_call",
                tool_name=tc["name"],
                tool_args={k: (str(v)[:200] if len(str(v)) > 200 else v) for k, v in tc.get("args", {}).items()},
                round=label,
            )

            name = tc["name"]
            arguments = {key: (value if key not in JSON_TEXT_ARGUMENTS or isinstance(value, str)
                               else "" if value is None else json.dumps(value))
                         for key, value in (tc.get("args") or {}).items()}
            retained_numbers = {entry["number"] for entry in rounds}
            call_key = _tool_call_key(name, arguments)
            prior_round = seen_calls.get(call_key) if name in REPLAYABLE_TOOLS else None
            duplicate_call = prior_round is not None and prior_round in retained_numbers
            # A call that failed is not re-run unchanged until something else succeeds.
            prior_failure = failed_calls.get(call_key)
            unchanged_failed_call = (
                isinstance(prior_failure, dict)
                and int(prior_failure.get("generation", -1)) == progress_generation
            )

            fn = tool_map.get(name)
            tool_ok = False
            tool_invoked = False
            tool_raised = False
            read_by_helper = False  # the model gets the reader's answer, not the source text
            target = _failure_target(name, arguments)
            cause_guarded = name not in WRITE_TOOLS and name not in CAUSE_GUARD_EXEMPT
            known_cause = failure_causes.get(target) if cause_guarded else None
            repeated_cause = (isinstance(known_cause, dict) and known_cause.get("count", 0) >= FAILURE_CAUSE_LIMIT
                              and known_cause.get("writes") == history.get("write_generation", 0))
            if repeated_cause:
                result = (
                    f"[{name} on {target.split(':', 1)[1] or 'this target'} failed {known_cause['count']} times with "
                    f"the same cause since the last change: {known_cause['cause']} It was not run again: other "
                    "arguments to the same call fail the same way. Read the code at the failure site and change "
                    "it, or report the blocker.]"
                )
                _publish_agent_log(request_name, "repeated_failure_blocked", tool_name=name, round=label,
                                   cause=known_cause["cause"][:200])
            elif unchanged_failed_call:
                result = (
                    f"[same failed call was already attempted in round {prior_failure.get('round')}; "
                    "it was not executed again because no successful intervening tool action changed the evidence. "
                    f"Previous result: {prior_failure.get('result', '')} Change the arguments, inspect relevant "
                    "state, or report the blocker.]"
                )
                _publish_agent_log(
                    request_name, "blind_retry_blocked",
                    tool_name=name, original_round=prior_failure.get("round"), round=label,
                )
            elif duplicate_call:
                result = (
                    f"[{name} with these exact arguments already returned in round "
                    f"{prior_round}; use that result. Nothing has changed.]"
                )
                _publish_agent_log(
                    request_name, "duplicate_tool_call",
                    tool_name=name, original_round=prior_round, round=label,
                )
            elif fn:
                try:
                    tool_invoked = True
                    result = str(fn.invoke(arguments))
                    tool_ok = _tool_result_succeeded(result)
                    if tool_ok and name in REPLAYABLE_TOOLS and len(result) >= 400:
                        result_round = seen_results.get(result)
                        if result_round is not None and result_round in retained_numbers:
                            result = (
                                f"[{name} returned bytes identical to round {result_round}; "
                                "use the earlier result and move on.]"
                            )
                        else:
                            seen_results[result] = label
                    raw = result
                    result = _bounded_tool_result(name, arguments, result)
                    purpose = str(arguments.get("purpose") or "").strip() if name in READER_TOOLS else ""
                    if _reader_replaces(purpose, reader, result):
                        # The helper reads the unbounded text; the conversation keeps only its answer.
                        text = raw[:READER_INPUT_CHARS]
                        answer = read_for(purpose, name, arguments, text, label)
                        read_by_helper = answer is not text
                        if read_by_helper:
                            result = answer
                            if seen_results.get(raw) == label:
                                del seen_results[raw]  # the model never saw these bytes; an exact read must run
                except Exception as e:
                    log_agent_error(
                        f"Agent Graph: tool {tc['name']}",
                        f"request={request_name}\n{e}\n{frappe.get_traceback()}",
                    )
                    result = f"Tool error: {e}"
                    tool_raised = True
            else:
                result = f"Unknown tool: {name}"

            if name in REPLAYABLE_TOOLS and not duplicate_call and tool_ok:
                seen_calls[call_key] = label

            if tool_ok and tool_invoked:
                progress_generation += 1
                history["tool_progress_generation"] = progress_generation
            if tool_invoked:
                if tool_ok:
                    failed_calls.pop(call_key, None)
                elif name in WRITE_TOOLS or not tool_raised:  # an exception may be transient
                    failed_calls[call_key] = {
                        "round": label,
                        "generation": progress_generation,
                        "result": result[:1000],
                    }
            if tool_invoked and cause_guarded:
                if tool_ok:
                    failure_causes.pop(target, None)
                elif not tool_raised:
                    cause, writes = _failure_cause(result), history.get("write_generation", 0)
                    entry = failure_causes.get(target)
                    same = isinstance(entry, dict) and entry.get("cause") == cause and entry.get("writes") == writes
                    count = entry["count"] + 1 if same else 1
                    failure_causes[target] = {"cause": cause, "count": count, "writes": writes}
                    if count == FAILURE_CAUSE_LIMIT:
                        result += ("\n[Second failure with this same cause since the last change; varying the "
                                   "arguments will not fix it. Read the code at the failure site, or report the blocker.]")

            if source_memory and name == "read_file" and not duplicate_call and not read_by_helper:
                source_memory.record(arguments, result, label)
            if name in WRITE_TOOLS:
                if source_memory:
                    source_memory.invalidate()
                # A failed write can still have partially changed the file.
                seen_calls.clear()
                seen_results.clear()
                if result.startswith(("WRITE_OK:", "EDIT_OK:", "COPY_OK:", "RENAME_OK:", "DELETE_OK:")):
                    round_had_write = True
                    history["write_generation"] = history.get("write_generation", 0) + 1
                    changed_paths = ([arguments.get("source_path"), arguments.get("destination_path")]
                                     if name == "rename_file" else
                                     [arguments.get("destination_path")] if name == "copy_file" else [arguments.get("path")])
                    for path_arg in changed_paths:
                        if path_arg and path_arg not in edited_paths:
                            edited_paths.append(path_arg)
                active_journal = checkpoint.journal()
                if active_journal and tool_invoked:
                    active_journal.finish_write()

            _publish_agent_log(request_name, "tool_result",
                tool_name=name,
                result_preview=result[:500],
                round=label,
            )
            round_entry["tools"].append(ToolMessage(content=result, tool_call_id=tc["id"]))

            arg_preview = ", ".join(
                f"{k}={str(v)[:60]}" for k, v in list(arguments.items())[:3]
            )
            round_entry["summary"].append(
                f"[r{label}] {name}({arg_preview}) -> {result[:COMPACT_RESULT_PREVIEW]}"
            )

        # A written body stays: it is the model's view of that file, and edits by
        # text anchor build on it. prune_rounds retires it once a newer view exists.
        rounds.append(round_entry)
        prune("batch", force=False, calls_made=round_num + 1)
        if after_round is not None:
            after_round()

        if advisor is not None and round_had_write:
            try:
                notes, reply = advisor.advise(round_entry)
            except Exception:
                log_agent_error("Agent Graph: advisor", frappe.get_traceback())
                notes, reply = [], None
            _publish_agent_log(request_name, "advisor", round=label, tokens=charge(reply), notes=notes)
            if notes:
                # After this round's results, like any directive: nothing already sent changes.
                _queue_directive(history, advisor_directive(notes))

        if stop_when is not None and stop_when():
            history["rounds_done"] = label
            total_tokens += _SIDE_TOKENS.pop(request_name, 0)  # e.g. the check of the plan just submitted
            return response_text, edited_paths, total_tokens, round_num + 1, False

        # A long read-only streak in implementation, even after an edit, gets one
        # reminder per turn to start editing.
        if round_had_write:
            read_only_streak = 0
        else:
            read_only_streak += 1
        if require_writes and read_only_streak >= READ_STREAK_LIMIT and not write_nudges:
            write_nudges = 1
            read_only_streak = 0
            # Anchored after this round, so the cached prefix stays unchanged.
            _queue_directive(history, NUDGE_TEXT)
            _publish_agent_log(request_name, "write_nudge",
                round=label, attempt=write_nudges, trigger="read_streak")
        single_edit = round_had_write and len(response.tool_calls) == 1
        single_edit_streak = single_edit_streak + 1 if single_edit else 0
        if single_edit_streak >= SINGLE_EDIT_STREAK and batch_nudges < MAX_BATCH_NUDGES:
            batch_nudges += 1
            single_edit_streak = 0
            context = int(history.get("previous_context_input_tokens") or 0)
            _queue_directive(history, BATCH_EDITS_TEXT.format(
                rounds=SINGLE_EDIT_STREAK, context=f" (about {context // 1000}k tokens)" if context else ""))
            _publish_agent_log(request_name, "batch_nudge", round=label, attempt=batch_nudges, context_tokens=context)

    if stop_when is not None:
        # A turn that ends on a tool call (investigation's submit_plan) has no report to ask
        # for: its caller decides what the time-up message says.
        history["rounds_done"] = round_base + max_rounds
        total_tokens += _SIDE_TOKENS.pop(request_name, 0)
        return "", edited_paths, total_tokens, max_rounds, True

    # Tool calls are forbidden on this call. Without saying so, a model that still
    # wants to work writes its next tool calls as prose instead of a report.
    final_notice = HumanMessage(content=(
        "STOP: the call limit for this turn is reached and tools are no longer available. "
        "Do not write any further tool calls. Return the required final report now. "
        'If work remains, use status "blocked" and list exactly what is unfinished and '
        "which edits were already applied, so the next attempt can continue from current source."
    ))
    final_messages = maybe_trim(extra=(final_notice,), with_tools=final_tools_bound, output_tokens=final_cap.value)
    final_context_chars = sum(message_chars(m) for m in final_messages)
    if progress is not None:
        progress["calls"] = max_rounds + 1
    checkpoint.update(tool_rounds_used=(state or {}).get("tool_rounds_used", 0) + max_rounds + 1)
    final_label = round_base + max_rounds + 1
    final = invoke_growing(llm_final, final_messages, final_cap, final_label,
                           estimate_messages(final_messages, schemas if final_tools_bound else ()), final_context_chars)
    history["rounds_done"] = final_label
    text = finish(_llm_response_text(final), (final_notice,), final_label)
    return text, edited_paths, total_tokens, max_rounds + 1, True


# Agent turn helper

def _queue_followup(history: dict, previous_output: str, feedback: str, *, kind: str = "") -> None:
    """Append the previous answer and new feedback after the retained rounds.

    The opening prompt never changes, so the cached prefix stays byte-identical.
    """
    rounds = history.get("rounds") or []
    history.setdefault("followups", []).append({
        "after": rounds[-1]["number"] if rounds else 0,
        "kind": kind,
        "messages": [
            AIMessage(content=previous_output or "(no output)"),
            HumanMessage(content=feedback),
        ],
    })


def _queue_directive(history: dict, text: str) -> None:
    """Anchor new context after retained rounds without inventing an assistant turn."""
    rounds = history.get("rounds") or []
    history.setdefault("followups", []).append({
        "after": rounds[-1]["number"] if rounds else 0,
        "messages": [HumanMessage(content=text)],
    })


def _run_agent_turn(state: dict, phase: str, prompt: str, read_only_tools: bool, max_rounds: int = 20,
                    history: dict | None = None) -> dict:
    """Run one agent turn for the given phase. Returns state updates.

    ``history`` lets a follow-on turn (reviewer recovery) continue from the
    retained tool rounds of the previous one instead of starting cold.
    """
    before = {}
    progress = {"tokens": state.get("tokens_used", 0), "calls": 0}
    try:
        if len(prompt) > MAX_TASK_PROMPT_CHARS:
            return {"error": "Execution context exceeds the prompt limit; shorten the task contract or custom prompt. No criteria were silently discarded."}
        app_name = state.get("target_app_name", "")
        provider = state.get("ai_provider", "OpenAI")
        model = state.get("ai_model", "gpt-4o-mini")
        request_name = state.get("request_name", "")

        remaining = state.get("tool_rounds_limit", 1000) - state.get("tool_rounds_used", 0)
        if not read_only_tools and state.get("execution_tasks"):
            if state.get("integration_mode"):
                remaining -= REPAIR_REVIEW_RESERVE
            else:
                pending = max(1, len(state["execution_tasks"]) - state.get("task_index", 0))
                remaining = (remaining - FINAL_REVIEW_RESERVE) // pending
        elif read_only_tools and state.get("execution_tasks") and not state.get("integration_mode") and not state.get("is_follow_up"):
            # A no-op task still needs independent verification, but cannot use
            # final integration's allocation or the minimum for future tasks.
            future_tasks = max(0, len(state["execution_tasks"]) - state.get("task_index", 0) - 1)
            remaining -= FINAL_REVIEW_RESERVE + 2 * future_tasks
        if remaining < 2:
            return {"error": "Execution call budget exhausted before verification completed."}
        max_rounds = min(max_rounds, remaining - 1)
        tools = _make_tools(
            app_name, read_only=read_only_tools,
            before=before,
        )
        llm = _get_llm(provider=provider, model=model)
        system_prompt = get_system_prompt(app_name or "target_app", request_name=request_name)
        content, tool_edited_paths, total_tokens, rounds_used, exhausted = _run_tool_calling_loop(
            llm, tools, system_prompt, prompt,
            request_name=request_name,
            max_rounds=max_rounds,
            state=state,
            provider=provider,
            require_writes=not read_only_tools,
            progress=progress,
            history=history,
        )
        content = _message_content_to_str(content)
        max_output = MAX_PHASE_OUTPUT_CHARS
        steps = list(state.get("intermediate_steps") or []) + [
            {"phase": phase, "output": (content[:max_output] if content else "")}
        ]
        result = {
            "current_stage": phase,
            "intermediate_steps": steps,
            "tokens_used": total_tokens,
            "tool_rounds_used": state.get("tool_rounds_used", 0) + rounds_used,
            "turn_exhausted": exhausted,
            "_write_baseline": before,
        }
        if tool_edited_paths:
            result["_tool_edited_paths"] = tool_edited_paths
        return result
    except Exception as e:
        log_agent_error(
            f"Agent Graph: {phase}",
            f"request={state.get('request_name')}\n{e}\n{frappe.get_traceback()}",
        )
        steps = list(state.get("intermediate_steps") or []) + [
            {"phase": phase, "output": f"Error: {e}"}
        ]
        return {
            "current_stage": phase,
            "intermediate_steps": steps,
            "_write_baseline": before,
            "tokens_used": progress["tokens"],
            "tool_rounds_used": state.get("tool_rounds_used", 0) + progress["calls"],
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Graph nodes — Planning phase
# ---------------------------------------------------------------------------

def understand_node(state: dict) -> dict:
    """Explore the codebase with the retrieval core and summarize it for planning.

    This is the one node backed by ``agent/core`` rather than by the tool-calling
    loop the other phases use, and the difference is worth naming because it is
    the difference between *searching* and *retrieving*.

    The old loop handed the model five read tools and trimmed its history when it
    got long. This one opens a session — a tree-sitter index of the app, a
    PageRank'd repo map in the cached prefix, a fused BM25 + graph retriever —
    then runs one turn against it. Before the model says anything, a retrieval
    pass has already put the files this request is about in front of it. Every
    finding goes to an append-only ledger, so an old tool result becomes
    ``[search "reminders" -> L14]`` instead of being cut; and a turn is
    summarized into SESSION STATE *before* anything is dropped rather than after.

    Read-only structurally, not by review: the writing tools are in the frozen
    array (removing them would invalidate the cached prefix) and the host
    declines every one of them by name.
    """
    if state.get("error"):
        return {"error": state["error"]}

    logs = _log_stage(state, "Understanding", "started", "Indexing the app and exploring")
    request_name = state.get("request_name", "")
    try:
        app_name = state.get("target_app_name", "")
        provider = state.get("ai_provider", "OpenAI")
        spent = int(state.get("tokens_used") or 0)

        llm = _get_llm(provider=provider, model=state.get("ai_model", "gpt-4o-mini"))
        # A planning run is explicitly from scratch. The process-local core cache is
        # useful for multi-turn chat, but this graph has one understanding turn; if
        # the same request is restarted, reusing that entry imports the old run's
        # transcript and tool results into a supposedly fresh request.
        koda_core.forget_session(request_name)
        try:
            result = koda_core.understand(
                question=(
                    get_understand_prompt(
                        state.get("user_message", ""),
                        state.get("request_type", "Improvement"),
                        request_name=request_name,
                    )
                    + CORE_TOOL_NOTE
                ),
                app_name=app_name,
                llm=llm,
                provider=provider,
                request_name=request_name,
                system_prompt=get_understand_system_prompt(
                    app_name or "target_app", request_name=request_name
                ),
                retrieval_query=state.get("user_message", ""),
                utility_llm=llm,
                spent=spent,
            )
        finally:
            koda_core.forget_session(request_name)

        steps = list(state.get("intermediate_steps") or []) + [
            {"phase": "Understanding", "output": result.summary[:MAX_PHASE_OUTPUT_CHARS]}
        ]
        updates = {
            "current_stage": "Understanding",
            "intermediate_steps": steps,
            "tokens_used": result.tokens,
            "understanding_summary": result.summary,
            "explored_paths": list(result.explored_paths),
        }

        if not result.ok:
            # Fail here rather than letting the plan node discover an empty summary,
            # and name the stop reason while doing it: `why` reports the rounds
            # ceiling, a late tool call or the provider error, where this used to
            # report only "produced no output" and send someone to look at nothing.
            #
            # And the summary is cleared. `result.summary` on a failed turn is the
            # error rendered as prose - "[the model call failed: 429]" - and leaving
            # that in the field the plan phase reads is how an outage becomes a plan.
            updates["understanding_summary"] = ""
            updates["error"] = result.why
            logs = _log_stage({**state, "stage_log": logs}, "Understanding", "failed",
                              updates["error"][:200])
            updates["stage_log"] = logs
            return updates

        summary = (
            f"{result.summary}\n\n"
            f"[explored {len(result.explored_paths)} file(s) over {result.rounds} round(s)]"
        )
        updates["understanding_summary"] = summary
        logs = _log_stage({**state, "stage_log": logs}, "Understanding", "completed",
                          result.summary[:200])
        updates["stage_log"] = logs
        return updates
    except Exception as e:
        log_agent_error(
            "Agent Graph: Understanding",
            f"request={request_name}\n{e}\n{frappe.get_traceback()}",
        )
        logs = _log_stage({**state, "stage_log": logs}, "Understanding", "failed",
                          str(e)[:200])
        return {
            "current_stage": "Understanding",
            "intermediate_steps": list(state.get("intermediate_steps") or [])
            + [{"phase": "Understanding", "output": f"Error: {e}"}],
            "error": str(e),
            "stage_log": logs,
        }

def _structured_plan_model(llm, provider: str):
    """Bind the plan schema using the provider's structured-output API."""
    # Native structured-output path per wrapper: OpenAI/OpenRouter enforce json_schema (strict),
    # Claude only has json_schema on newer models so use tool input, Gemini rejects json_schema.
    method = {
        "Claude": "function_calling",
        "Gemini": "json_mode",
    }.get(provider, "json_schema")
    options = {"method": method, "include_raw": True}
    if provider in ("OpenAI", "OpenRouter"):
        options["strict"] = True
    structured = llm.with_structured_output(PLAN_JSON_SCHEMA, **options)
    if provider == "OpenRouter":
        structured = structured.bind(extra_body={
            "usage": {"include": True},
            "provider": {"require_parameters": True},
        })
    return structured


def _plan_run_config(run_name: str, *, round_number: int, mode: str) -> dict:
    """Trace name, tags and round metadata for one planner call."""
    return {
        "run_name": run_name,
        "tags": ["plan", f"plan:{mode}"],
        "metadata": {"plan_round": round_number, "plan_mode": mode},
    }


def _charge_plan_call(raw_response, request_name: str, total_tokens: int, *, round_label: int) -> int:
    usage = getattr(raw_response, "usage_metadata", None)
    if not usage:
        return total_tokens
    total_tokens += int(usage.get("total_tokens") or 0)
    details = usage.get("input_token_details") or {}
    input_tokens = int(usage.get('input_tokens') or 0)
    cache_read = int(details.get('cache_read') or 0)
    cache_write = int(details.get('cache_creation') or 0)
    cost = provider_cost(raw_response)
    _publish_agent_log(request_name, "token_usage",
        round=round_label,
        tokens_this_round=usage.get("total_tokens", 0),
        tokens_total=total_tokens,
        input_tokens=max(0, input_tokens - cache_read - cache_write),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        context_input_tokens=input_tokens,
        cache_phase='planning',
        provider_cost=cost,
        upstream_provider=(getattr(raw_response, "response_metadata", None) or {}).get("upstream_provider"),
    )
    _persist_token_usage(request_name, total_tokens, input_tokens=input_tokens,
                         cache_read_tokens=cache_read, cache_write_tokens=cache_write, cost_delta=cost)
    return total_tokens


def _plan_output_tokens(provider: str) -> int:
    """Output cap for one planner call, including reasoning room on OpenRouter."""
    if provider == "OpenRouter":
        return MODEL_PLAN_OUTPUT_TOKENS + OPENROUTER_REASONING_ALLOWANCE_TOKENS
    return MODEL_PLAN_OUTPUT_TOKENS


def _plan_budget(provider: str, state: dict | None = None) -> dict:
    """Mutable output cap shared by every planner call in one stage.

    Once a call had to grow the cap, later repair and patch calls start from
    the grown value instead of rediscovering the same truncation. ``window``
    is the model's context window, from which each call derives the real
    room it has; ``ceiling`` records a smaller limit the provider rejected
    above, once known.
    """
    state = state or {}
    try:
        window, _ = koda_core.request_limits(state.get("target_app_name", ""), state.get("ai_model", "gpt-4o-mini"))
    except Exception:  # noqa: BLE001 - an unknown window only limits growth, never the call
        window = 0
    return {"max_tokens": _plan_output_tokens(provider), "window": int(window or 0), "ceiling": 0}


def _invoke_structured(llm, provider: str, schema: dict, messages: list, config: dict, *,
                       budget: dict, request_name: str, total_tokens: int, round_label: int) -> tuple[dict, int]:
    """One planner call that grows its output cap when the reply is truncated.

    Every attempt is charged. Truncation at the context window's ceiling raises
    ``PlanValidationError``. Returns the structured response and the token total.
    """
    estimate = estimate_messages(messages) + serialized_tokens(schema)
    window = int(budget.get("window") or 0) or DEFAULT_WINDOW_TOKENS
    ceiling = max(int(budget["max_tokens"]), recovery.output_ceiling(window, estimate))
    if budget.get("ceiling"):
        ceiling = min(ceiling, int(budget["ceiling"]))
    cap = recovery.OutputCap(budget["max_tokens"], ceiling)

    def call(max_tokens: int):
        check_active(reserve=MODEL_TIME_RESERVE)
        return _structured_model(llm, provider, schema, max_tokens=max_tokens).invoke(messages, config=config)

    def note(previous: int, grown: int, reasoning: int) -> None:
        _publish_agent_log(request_name, "llm_response", round=round_label,
            preview=(f"Planner ran out of output room at {previous:,} tokens"
                     + (f" ({reasoning:,} spent reasoning)" if reasoning else "")
                     + f"; retrying with a {grown:,}-token cap"))

    outcome = recovery.invoke_growing(
        call, cap,
        raw_of=lambda reply: reply.get("raw") if isinstance(reply, dict) else None,
        on_retry=note,
    )
    # Wasted attempts are still billed; keep the running total where the
    # caller can read it back when this ends in PlanValidationError.
    for chargeable in outcome.wasted:
        total_tokens = _charge_plan_call(chargeable, request_name, total_tokens, round_label=round_label)
    raw = outcome.reply.get("raw") if isinstance(outcome.reply, dict) else None
    total_tokens = _charge_plan_call(raw, request_name, total_tokens, round_label=round_label)
    budget["max_tokens"] = cap.value
    if cap.ceiling < ceiling:
        budget["ceiling"] = cap.ceiling
    budget["total_tokens"] = total_tokens
    if outcome.truncated:
        raise PlanValidationError([
            f"Planner output reached the model output limit ({cap.value:,} tokens is all this model has left"
            + (f"; {outcome.reasoning_tokens:,} spent reasoning" if outcome.reasoning_tokens else "") + ")"])
    return outcome.reply, total_tokens


def _structured_model(llm, provider: str, schema: dict, *, max_tokens: int | None = None):
    """Bind ``schema`` using the provider's structured-output API.

    The output cap and extra_body go on a copy of the model before wrapping:
    kwargs passed to the wrapped sequence never reach the request.
    """
    update: dict = {}
    if max_tokens:
        fields = getattr(type(llm), "model_fields", {})
        for name in ("max_tokens", "max_output_tokens"):
            if name in fields:
                update[name] = int(max_tokens)
                break
    if provider == "OpenRouter":
        # Keep the usage flag and sticky-routing session_id already in extra_body.
        body = dict(getattr(llm, "extra_body", None) or {})
        body["provider"] = {**dict(body.get("provider") or {}), "require_parameters": True}
        update["extra_body"] = body
    if update:
        llm = llm.model_copy(update=update)
    # Native structured-output path per wrapper: OpenAI/OpenRouter enforce json_schema (strict),
    # Claude only has json_schema on newer models so use tool input, Gemini rejects json_schema.
    method = {
        "Claude": "function_calling",
        "Gemini": "json_mode",
    }.get(provider, "json_schema")
    options = {"method": method, "include_raw": True}
    if provider in ("OpenAI", "OpenRouter"):
        options["strict"] = True
    return llm.with_structured_output(schema, **options)


def _unpack_structured_plan(result) -> tuple[object, dict]:
    if not isinstance(result, dict) or "parsed" not in result:
        raise PlanValidationError(["Provider returned no structured plan result"])

    raw = result.get("raw")
    metadata = getattr(raw, "response_metadata", None) or {}
    reason = str(metadata.get("finish_reason") or metadata.get("stop_reason") or "").lower()
    status = str(metadata.get("status") or "").lower()
    if reason in {"length", "max_tokens", "max_output_tokens"} or status == "incomplete":
        raise PlanValidationError(["Planner output reached the model output limit"])
    if result.get("parsing_error"):
        raise PlanValidationError([f"Provider rejected the structured plan: {result['parsing_error']}"])

    parsed = result.get("parsed")
    if hasattr(parsed, "model_dump"):
        parsed = parsed.model_dump()
    if not isinstance(parsed, dict):
        raise PlanValidationError(["Provider returned no structured plan object"])
    return raw, parsed


def plan_node(state: dict) -> dict:
    """Turn the understanding summary into a step-by-step implementation plan."""
    if state.get("error"):
        return {"error": state["error"]}

    logs = _log_stage(state, "Planning", "started", "Creating todo-based implementation plan")

    try:
        provider = state.get("ai_provider", "OpenAI")
        model = state.get("ai_model", "gpt-4o-mini")
        app_name = state.get("target_app_name", "target_app")
        request_name = state.get("request_name", "")
        understanding = _message_content_to_str(state.get("understanding_summary", ""))
        user_message = state.get("user_message", "")

        if not understanding.strip():
            logs = _log_stage({**state, "stage_log": logs}, "Planning", "failed",
                "No understanding summary available — explore phase produced no output")
            return {"error": "Understanding phase produced no output", "stage_log": logs}

        llm = _get_llm(provider=provider, model=model)
        system_prompt = get_system_prompt(app_name, request_name=request_name)
        plan_prompt = get_plan_prompt(understanding, user_message, request_name=request_name)

        _publish_agent_log(request_name, "llm_response",
            preview="Generating todo-based plan from codebase analysis...", round=1)

        structured_llm = _structured_plan_model(llm, provider)
        response = structured_llm.invoke([
            _build_system_message(provider, system_prompt, model),
            HumanMessage(content=plan_prompt),
        ], max_tokens=MODEL_PLAN_OUTPUT_TOKENS)
        raw_response, plan_object = _unpack_structured_plan(response)

        total_tokens = state.get("tokens_used", 0)
        usage = getattr(raw_response, "usage_metadata", None)
        if usage:
            total_tokens += int(usage.get("total_tokens") or 0)
            _publish_agent_log(request_name, "token_usage",
                round=1,
                tokens_this_round=usage.get("total_tokens", 0),
                tokens_total=total_tokens,
            )
            _persist_token_usage(request_name, total_tokens)
        else:
            logs = _log_stage(
                {**state, "stage_log": logs}, "Planning", "progress",
                f"{provider} reported no token usage for the plan call - "
                f"recorded total stays at {total_tokens} and undercounts this request",
            )

        plan_object = validate_plan(plan_object)

        tasks = plan_object["tasks"]
        _publish_agent_log(
            request_name,
            "llm_response",
            preview=f"Structured plan returned {len(tasks)} validated task(s)",
            round=2,
        )
        logs = _log_stage(
            {**state, "stage_log": logs},
            "Planning",
            "progress",
            f"{len(tasks)} task(s) validated; paths and dependency graph verified",
        )

        # Markdown is a display projection; execution consumes plan_object.
        plan = plan_to_markdown(plan_object)

        steps = list(state.get("intermediate_steps") or []) + [
            {"phase": "Planning", "output": plan[:MAX_PHASE_OUTPUT_CHARS]}
        ]

        logs = _log_stage({**state, "stage_log": logs}, "Planning", "completed",
            f"Plan generated ({len(plan)} chars, {len(tasks)} task(s))")

        return {
            "current_stage": "Planning",
            "plan": plan,
            "plan_object": plan_object,
            "intermediate_steps": steps,
            "stage_log": logs,
            "tokens_used": total_tokens,
        }
    except Exception as e:
        log_agent_error(
            "Agent Graph: Planning",
            f"request={state.get('request_name')}\n{e}\n{frappe.get_traceback()}",
        )
        logs = _log_stage({**state, "stage_log": logs}, "Planning", "failed", str(e)[:200])
        return {
            "current_stage": "Planning",
            "error": str(e),
            "stage_log": logs,
        }


# ---------------------------------------------------------------------------
# Graph nodes — Execution phase
# ---------------------------------------------------------------------------

def _check_plan_input(state: dict, messages: list, schema: dict, budget: dict) -> None:
    estimate = estimate_messages(messages) + serialized_tokens(schema)
    window, ceiling = koda_core.request_limits(state.get("target_app_name", ""), state.get("ai_model", "gpt-4o-mini"))
    limit = input_limit(window, budget["max_tokens"], ceiling)
    if estimate > limit:
        raise ValueError(f"Planning context exceeds the input budget ({estimate:,} > {limit:,} tokens).")


# Graph nodes — Execution phase

def prepare_execution_node(state: dict) -> dict:
    """Freeze the validated contract once, before any task writes."""
    try:
        if state.get("is_follow_up"):
            plan = load_plan(state.get("plan_object"))
            tasks = [{
                "id": "FOLLOW_UP", "title": "Follow-up patch",
                "goal": state.get("follow_up_message", ""),
                "description": "Patch the reported issue without rebuilding the original plan.",
                "files": state.get("prior_changed_paths") or [], "context_refs": [],
                "acceptance_criteria": [state.get("follow_up_message") or "Fix the reported issue"],
                "depends_on": [],
            }]
        else:
            plan = load_plan(state.get("plan_object"))
            tasks = plan["tasks"]
        return {
            "plan_object": plan, "execution_tasks": tasks, "task_index": 0,
            "task_results": [], "task_baseline": {}, "execution_baseline": {},
            "task_completion": {}, "turn_exhausted": False,
            "review_attempts": 0, "review_notes": "", "review_passed": False,
            "integration_mode": False, "tool_rounds_used": 0,
            "tool_rounds_limit": BASE_EXECUTION_CALL_BUDGET + PER_TASK_CALL_BUDGET * len(tasks),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _execution_context(state: dict) -> tuple[dict, list[str], list[str] | None]:
    tasks = state["execution_tasks"]
    if state.get("integration_mode"):
        active = {"id": "INTEGRATION", "tasks": tasks}
        criteria = [f"{t['id']}: {c}" for t in tasks for c in t["acceptance_criteria"]]
        paths = list(dict.fromkeys(p for t in tasks for p in t["files"]))
    else:
        active = tasks[state["task_index"]]
        criteria = active["acceptance_criteria"]
        paths = active["files"]
    return active, criteria, None if state.get("is_follow_up") else paths


def _read_current(state: dict, path: str):
    return read_snapshot(agent_tools._resolve_path(state["target_app_name"], path))


def _persist_task_results(state: dict):
    """Persist audit results, not a claim that filesystem writes are replayable."""
    if state.get("request_name"):
        frappe.db.set_value(DOCTYPE_NAME, state["request_name"], "execution_results", json.dumps({
            "tasks": state.get("task_results", []),
            "final_review_passed": bool((state.get("integration_mode") or state.get("is_follow_up")) and state.get("review_passed")),
            "review_notes": state.get("review_notes", ""),
            "model_calls": state.get("tool_rounds_used", 0),
            "error": state.get("error", ""),
        }, ensure_ascii=True))
        frappe.db.commit()


def implement_node(state: dict) -> dict:
    """Implement one ready task, or repair concrete integration findings."""
    if state.get("error"):
        return {"error": state["error"]}
    active, criteria, allowed = _execution_context(state)
    dependencies = {d.casefold() for d in active.get("depends_on", [])}
    completed = [r for r in state.get("task_results", []) if r["task_id"].casefold() in dependencies]
    ready = {r["task_id"].casefold() for r in completed if r.get("status") in {"implemented", "passed"}}
    if dependencies - ready:
        return {"error": "Incomplete dependency: " + ", ".join(sorted(dependencies - ready))}
    if state.get("integration_mode"):
        completed = [r for r in state.get("task_results", []) if r["task_id"] != "INTEGRATION"]
    completed = [{**r, "changed_since_completion": any(
        revision(_read_current(state, c["path"])) != c["after"] for c in r.get("changes", [])
    )} for r in completed]
    logs = _log_stage(state, "Implementing", "started", f"{active['id']}: attempt {state.get('review_attempts', 0) + 1}")
    task_before = dict(state.get("task_baseline") or {})
    for path in allowed or []:
        if path not in task_before:
            task_before[path] = _read_current(state, path)
    plan = state["plan_object"]
    if state.get("is_follow_up"):
        prompt = get_follow_up_implement_prompt(
            state.get("follow_up_message", ""), json.dumps(plan, ensure_ascii=True),
            # The snapshot accumulates across follow-ups up to 50k; unbounded it
            # trips MAX_TASK_PROMPT_CHARS and fails every later follow-up.
            _bounded_text(state.get("implementation_memory", ""), MAX_UNDERSTANDING_CONTEXT_CHARS),
            "\n".join(state.get("prior_changed_paths") or []),
            _file_manifest(state.get("prior_changed_paths") or []),
            request_name=state.get("request_name"),
        )
    else:
        # Custom implementation instructions still apply, but receive only the active task.
        prompt = get_implement_prompt(
            json.dumps(active, ensure_ascii=True),
            _bounded_text(state.get("understanding_summary", ""), MAX_UNDERSTANDING_CONTEXT_CHARS),
            state.get("user_message", ""), _file_manifest(allowed or []),
            request_name=state.get("request_name"),
        )
    prompt += "\n\n## EXECUTION CONTRACT (authoritative)\n" + json.dumps({
        "overview": plan.get("overview", ""), "scope": plan.get("scope", {}),
        "active_task": active, "dependency_results": completed,
        "acceptance_criteria": criteria,
    }, ensure_ascii=True)
    prompt += (
        "\nImplement only the active task. Read its context_refs from the CURRENT files; "
        "line numbers may have shifted. Dependency reports are implementation claims, not proof. "
        "Do not execute other tasks. If approved scope is insufficient, report the blocker. "
        "Preserve the dependency behavior across files and languages; verify shared input/output examples. "
        "Do not claim runtime tests ran when only static checks are available."
    )
    dependency_ids = {r["task_id"].casefold() for r in completed}
    dependency_paths = list(dict.fromkeys(p for task in state["execution_tasks"]
        if task["id"].casefold() in dependency_ids for p in task["files"]))
    if dependency_paths:
        prompt += "\n\n## CURRENT DEPENDENCY SOURCE\n" + source_context(
            dependency_paths, lambda p: _read_current(state, p), state.get("execution_baseline"), limit=8000)
    prompt += (
        '\nReturn ONLY a JSON completion report: {"status":"complete","summary":"Actual changes",'
        '"behavior":["path:symbol, rule and concrete input/output example for dependents"],'
        '"verification":["Checks actually performed, with outcomes"],"unverified":["Remaining uncertainty"]}. '
        'Use status "blocked" if scope or missing information prevents completion. '
        'List all unresolved work; do not claim complete after a forced call-limit summary. '
        'Verification and unverified may be empty arrays; behavior must describe the completed contract. '
        'This JSON format overrides any summary-format instructions above.'
    )
    if state.get("integration_mode"):
        prompt += "\nThis is an integration repair: fix only the findings below; do not rebuild completed tasks."
    if state.get("review_notes"):
        prompt += "\n\n## FINDINGS TO REPAIR\n" + state["review_notes"]
    updates = _run_agent_turn(
        {**state, "allowed_write_paths": allowed}, "Implementing", prompt,
        read_only_tools=False,
        max_rounds=MAX_TOOL_ROUNDS_REPAIR if state.get("review_attempts") else MAX_TOOL_ROUNDS_EXECUTION,
    )
    before = updates.pop("_write_baseline", {})
    updates.pop("_tool_edited_paths", None)
    global_before = dict(state.get("execution_baseline") or {})
    for path, content in before.items():
        task_before.setdefault(path, content)
        global_before.setdefault(path, content)
    # Include approved files even for no-op tasks; the reviewer must verify them.
    for path, content in task_before.items():
        global_before.setdefault(path, content)
    edits, _ = change_evidence(global_before, lambda p: _read_current(state, p))
    steps = updates.get("intermediate_steps") or []
    output = steps[-1].get("output", "") if steps else ""
    completion = completion_report(output)
    updates.update({
        "task_baseline": task_before, "execution_baseline": global_before,
        "edits_made": edits,
        "task_completion": completion,
        "task_summary": completion.get("summary", _bounded_text(output, 4000)),
        "stage_log": _log_stage({**state, "stage_log": logs}, "Implementing", "completed" if not updates.get("error") else "failed",
                                f"{active['id']}: {len(edits)} total changed file(s); review pending"),
    })
    return updates


def review_node(state: dict) -> dict:
    """Gate completed tasks; independently review no-ops and final integration."""
    if state.get("error"):
        _persist_task_results(state)
        return {"error": state["error"]}
    active, criteria, allowed = _execution_context(state)
    integration = bool(state.get("integration_mode"))
    logs = _log_stage(state, "Reviewing", "started", f"Reviewing {active['id']}")
    baseline = state.get("execution_baseline") if integration else state.get("task_baseline")
    changes, diff = change_evidence(baseline or {}, lambda p: _read_current(state, p))
    paths = list(dict.fromkeys((allowed or []) + [e["path"] for e in changes]))
    reviewed_content = {p: _read_current(state, p) for p in paths}
    # Task-local checks must not reject temporarily incomplete cross-task wiring.
    if integration or state.get("is_follow_up"):
        health = run_health_checks(state["target_app_name"], [{"path": p} for p in paths])
    else:
        health = run_task_checks(state["target_app_name"], paths)
    health.results.extend(
        CheckResult(f"exists:{p}", False, "Approved task file is missing")
        for p in paths if not _app_file_exists(state["target_app_name"], p)
    )
    attempt = state.get("review_attempts", 0) + 1
    updates = {}
    completion = completion_report(json.dumps(state.get("task_completion") or {}))
    gate_only = False
    if not health.passed:
        passed, notes = False, health.summary()
    elif state.get("turn_exhausted"):
        passed, notes = False, "Implementation reached its call limit without finishing. Inspect current changes and complete the active task."
        # A blocked report from the forced final call says what is left; hand
        # it to the retry instead of making it rediscover the state.
        remaining = [completion.get("summary", "")] + list(completion.get("unverified") or [])
        remaining = [item for item in remaining if isinstance(item, str) and item.strip()]
        if completion.get("status") == "blocked" and remaining:
            notes += " Reported remaining work: " + " | ".join(remaining)[:1500]
    elif (not integration or state.get("review_attempts", 0) > 0) and completion.get("status") != "complete":
        passed, notes = False, "Implementation did not return a valid complete JSON report. Finish the task and report its behavior and verification."
        if completion.get("status") == "blocked":
            notes = "Implementation blocked: " + completion["summary"]
            updates["error"] = notes
    elif not integration and not state.get("is_follow_up") and changes:
        # Completion + mechanical checks gate dependency progress. Semantic
        # correctness is deliberately deferred, never recorded as reviewed.
        gate_only = True
        passed, notes = True, "Task completion and static checks passed; semantic review pending final integration."
    else:
        prompt = get_review_prompt(
            [{"path": p} for p in paths], state.get("follow_up_message") or state.get("user_message", ""),
            request_name=state.get("request_name"),
        )
        prompt += "\n\n## REVIEW CONTRACT (authoritative)\n" + json.dumps({
            "phase": "final integration" if integration else "task review (including no-op verification)",
            "active_task": active,
            "criteria": [{"criterion": i, "requirement": c} for i, c in enumerate(criteria, 1)],
            "implementation_claims": state.get("task_results", []) if integration else [completion],
            "remaining_tasks": [] if integration else [
                {key: task[key] for key in ("id", "goal", "files", "acceptance_criteria", "depends_on")}
                for task in state["execution_tasks"][state["task_index"] + 1:]
            ],
        }, ensure_ascii=True)
        prompt += "\n\n## ACTUAL CHANGES\n" + diff
        prompt += "\n\n## CURRENT SOURCE EVIDENCE\n" + source_context(paths, reviewed_content.get, baseline)
        prompt += "\n\n## STATIC CHECKS\n" + health.summary()
        prompt += (
            "\nInspect current source and relevant dependencies to assess every criterion. "
            "A task review covers this task's obligations; wiring assigned to a remaining task "
            "is checked at integration. Final integration must inspect interactions and recheck "
            "earlier criteria against final source. Trace connected behavior through callers, "
            "document lifecycle/persistence, queries and consumers where relevant; isolated function "
            "checks do not establish the connected result. Compare shared algorithms and concrete "
            "input/output examples across languages. Do not invent requirements outside the approved scope. "
            "Static checks do not prove runtime behavior. Evidence must cite current paths/symbols "
            "and distinguish source inspection from executed tests. Implementation claims are not proof. "
            "Truncated excerpts or missing context are requests to use your read-only tools, not code defects. "
            "You cannot run a browser, a server or a test suite here. Criteria about runtime, UI or "
            "interaction behavior are judged by tracing the source path that produces that behavior: "
            'mark them "satisfied" or "unmet" from that trace and say in the evidence text that the runtime '
            "was not exercised. Never fail a criterion only because no live test or screenshot was supplied. "
            'Return ONLY JSON: {"review_passed":true,"issues":[],"evidence":'
            '[{"criterion":1,"status":"satisfied","evidence":"path:symbol and concrete evidence"}]}. '
            'Include exactly one entry per numbered criterion. Use status "unmet" with actionable issues '
            'for an observed defect. Use "unverified" only for current source your read-only tools can '
            "still fetch and you have not read yet; it is never the answer for behavior that cannot be "
            "executed in this environment. Either requires review_passed false. First fetch missing "
            "source with tools. Do not ask implementation to change code merely because you lack evidence."
        )
        decision, notes = "invalid", ""
        prior_unmet = set()
        # The recovery pass continues the first pass's retained tool rounds, so
        # its smaller round budget is spent on reads that have not happened yet.
        history: dict = {}
        for recovery in range(2):
            updates = _run_agent_turn(
                {**state, **updates}, "Reviewing", prompt, read_only_tools=True,
                max_rounds=MAX_TOOL_ROUNDS_REVIEW if recovery == 0 else MAX_TOOL_ROUNDS_REVIEW_RECOVERY,
                history=history,
            )
            updates.pop("_write_baseline", None)
            updates.pop("_tool_edited_paths", None)
            steps = updates.get("intermediate_steps") or []
            output = steps[-1].get("output", "") if steps else ""
            payload = _extract_review_json(output)
            decision, notes = review_decision(payload, criteria)
            if prior_unmet and decision != "invalid":
                now_satisfied = {e["criterion"] for e in payload["evidence"] if e["status"] == "satisfied"}
                resolved = payload.get("resolved_findings", [])
                valid_resolutions = isinstance(resolved, list) and all(
                    isinstance(item, dict) and type(item.get("criterion")) is int
                    and isinstance(item.get("explanation"), str) and item["explanation"].strip()
                    for item in resolved
                )
                resolved_ids = {item["criterion"] for item in resolved} if valid_resolutions else set()
                if (prior_unmet & now_satisfied) - resolved_ids:
                    decision, notes = "invalid", "Invalid review result: earlier concrete findings need explicit resolution evidence."
            if updates.get("turn_exhausted") and not updates.get("error") and decision in {"pass", "repair"}:
                # The forced no-tools final call still produced a complete,
                # well-formed verdict. Reaching the round cap is not a reason
                # to discard it; only an incomplete or invalid verdict is.
                updates["turn_exhausted"] = False
                _publish_agent_log(state.get("request_name", ""), "review_verdict_after_cap",
                                   decision=decision, recovery=recovery)
            if updates.get("error") or updates.get("turn_exhausted"):
                notes = updates.get("error") or "Reviewer exhausted its call limit; verification is incomplete."
                updates["error"] = notes
                decision = "needs_evidence"
                break
            if decision not in {"invalid", "needs_evidence"}:
                break
            if recovery == 0:
                entries = payload.get("evidence", []) if isinstance(payload, dict) else []
                # A malformed envelope may still contain a valid concrete finding.
                prior_unmet = {e["criterion"] for e in entries if isinstance(e, dict)
                    and type(e.get("criterion")) is int and 1 <= e["criterion"] <= len(criteria)
                    and e.get("status") == "unmet"} if isinstance(entries, list) else set()
                prompt += (
                    "\n\n## REVIEWER RECOVERY\n"
                    "Your previous verdict needs evidence or format correction. The tool results from "
                    "your first pass are retained in this conversation; do not re-read them. Use read-only "
                    "tools only for current source you have not fetched yet, preserve all concrete findings, "
                    "then return the complete verdict. Runtime, UI or interaction criteria are judged from "
                    "the source trace, never left unverified for lack of a live test. "
                    "Do not send missing evidence to implementation. "
                    'If any earlier unmet criterion becomes satisfied, include resolved_findings '
                    '[{"criterion":1,"explanation":"Current source evidence resolving the earlier finding"}]. '
                    "Every concrete finding must remain or be explicitly resolved.\n"
                    + output
                )
        passed = decision == "pass" and not updates.get("error")
        if decision in {"invalid", "needs_evidence"} and not updates.get("error"):
            updates["error"] = "Reviewer could not resolve " + (
                "missing evidence" if decision == "needs_evidence" else "invalid verdict format"
            ) + " after a read-only recovery attempt."
    if passed and any(_read_current(state, p) != content for p, content in reviewed_content.items()):
        passed, notes = False, "Files changed during review. Verify current source again before accepting this task."
        updates["error"] = notes
    updates.update({"review_passed": passed, "review_notes": notes, "review_attempts": attempt})
    if not passed and attempt >= MAX_REVIEW_ATTEMPTS and not updates.get("error"):
        updates["error"] = f"{active['id']} failed review after {attempt} attempts: {notes[:1000]}"
    results = list(state.get("task_results") or [])
    result = {
        "task_id": active["id"], "status": ("implemented" if gate_only else "passed") if passed else "failed",
        "attempts": attempt, "summary": state.get("task_summary", ""),
        "behavior": completion.get("behavior", []),
        "verification": completion.get("verification", []),
        "unverified": completion.get("unverified", []),
        "changes": changes, "review": notes,
    }
    results = [r for r in results if r["task_id"] != active["id"]] + [result]
    updates["task_results"] = results
    updates["change_summary"] = "\n\n".join(f"{r['task_id']}: {r['summary']}" for r in results if r.get("summary"))
    updates["stage_log"] = _log_stage({**state, "stage_log": logs}, "Reviewing", "completed",
                                       f"{active['id']}: {'implemented; final review pending' if gate_only else ('passed' if passed else 'failed')}")
    _persist_task_results({**state, **updates})
    return updates


def advance_task_node(state: dict) -> dict:
    active = state["execution_tasks"][state["task_index"]]
    completed = any(r.get("task_id") == active["id"] and r.get("status") in {"implemented", "passed"}
                    for r in state.get("task_results", []))
    if state.get("error") or state.get("turn_exhausted") or not state.get("review_passed") or not completed:
        return {"error": state.get("error") or "Cannot advance an unfinished task.", "review_passed": False}
    index = state["task_index"] + 1
    integration = index >= len(state["execution_tasks"])
    return {
        "task_index": index, "integration_mode": integration,
        "task_baseline": {}, "review_attempts": 0, "review_notes": "",
        "review_passed": False, "turn_exhausted": False, "task_summary": "", "task_completion": {},
    }


def _get_bench_env() -> dict:
    """Build a subprocess environment with the correct Node.js on PATH.
    nvm installs Node under ~/.nvm/versions/node/<version>/bin but background
    workers inherit a bare PATH that points to the system Node (v12).
    This helper finds the nvm-managed Node and prepends it to PATH."""
    env = os.environ.copy()
    nvm_dir = env.get("NVM_DIR", os.path.expanduser("~/.nvm"))
    versions_dir = os.path.join(nvm_dir, "versions", "node")
    if os.path.isdir(versions_dir):
        candidates = sorted(
            (d for d in os.listdir(versions_dir) if d.startswith("v")),
            reverse=True,
        )
        for ver in candidates:
            bin_dir = os.path.join(versions_dir, ver, "bin")
            node_bin = os.path.join(bin_dir, "node")
            if os.path.isfile(node_bin):
                env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
                break
    return env


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_retry_implement(state: dict) -> str:
    if state.get("error"):
        return "done"
    if not state.get("review_passed"):
        return "implement"
    if state.get("integration_mode") or state.get("is_follow_up"):
        return "done"
    return "advance"


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_planning_graph():
    """
    Constructs the state machine for the Planning Phase.
    Flow: Understand the code → Draft a Plan.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("understand", understand_node)
    workflow.add_node("plan", plan_node)

    workflow.set_entry_point("understand")
    workflow.add_edge("understand", "plan")
    workflow.add_edge("plan", END)

    return workflow.compile()


def build_execution_graph():
    """Sequential implementation/gates, bounded repair, final semantic review."""
    workflow = StateGraph(AgentState)
    workflow.add_node("prepare", prepare_execution_node)
    workflow.add_node("implement", implement_node)
    workflow.add_node("review", review_node)
    workflow.add_node("advance", advance_task_node)
    workflow.set_entry_point("prepare")
    workflow.add_edge("prepare", "implement")
    workflow.add_edge("implement", "review")
    workflow.add_conditional_edges("review", should_retry_implement, {
        "implement": "implement", "advance": "advance", "done": END,
    })
    workflow.add_conditional_edges("advance", lambda s: "review" if s["integration_mode"] else "implement", {
        "review": "review", "implement": "implement",
    })
    return workflow.compile()
