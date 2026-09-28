# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# The agent's machinery: model access, the tool catalogue and tool-calling loop,
# execution setup and the independent review. session.py runs a request through
# them as one conversation; bench and deploy run in executor.py.

import hashlib
import html
import json
import os
import re as _re
from copy import deepcopy
from datetime import datetime

import frappe
from langchain_core.tools import StructuredTool, tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from ampower_koda.agent.prompt_caching import mark_message, native_cache_messages, openai_breakpoints, rolling_messages, terminal_model
from langchain_openai import ChatOpenAI

from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.cache_usage import persist_usage, provider_cost
from ampower_koda.agent import koda_core
from ampower_koda.agent import checkpoint
from ampower_koda.agent import recovery
from ampower_koda.agent import verification
from ampower_koda.agent import python_diagnostics
from ampower_koda.agent.advisor import Advisor, directive as advisor_directive
from ampower_koda.agent import repair_budget
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
    PLAN_PATCH_SCHEMA,
    PlanValidationError,
    apply_plan_patch,
    complete_rename_file_scope,
    nearest_paths,
    plan_to_markdown,
    repair_feedback,
    validate_plan,
)
from ampower_koda.agent.checks import run_health_checks, run_query_schema_checks, CheckResult, HealthReport
from ampower_koda.agent.execution_contract import (
    completion_report_problem,
    load_plan, read_snapshot, change_evidence, review_decision, completion_report, revision, renamed_sources,
)
from ampower_koda.agent.execution_evidence import source_context, SourceMemory
from ampower_koda.agent.history_prune import describe_call, drop_followups, prune_price, prune_rounds
from ampower_koda.agent.prompts import get_review_prompt, get_system_prompt


# Runaway guards, not work units: a turn ends when the model reports, and the
# request-wide call budget bounds the whole run.
MAX_TOOL_ROUNDS_EXECUTION = 60
MIN_REPAIR_ROUNDS = 30            # a pass with fewer calls left than this first extends the budget
MAX_TOOL_ROUNDS_REVIEW = 12       # includes live call_method checks, not only reads
MAX_TOOL_ROUNDS_REVIEW_RECOVERY = 6  # continues the first pass's history, so these are new reads only
# Review repairs stop on *no progress*, not on a magic attempt count. The
# total call budget remains the safety fence for genuinely novel findings.
MAX_REVIEW_ATTEMPTS = 2           # failed reviews of a repeated state before strategy changes
MAX_REPAIR_STRATEGIES = 2         # fresh diagnoses before a repeated failure may stop
# Novel findings produce cost telemetry; stalled findings first change strategy.
REVIEW_COST_PRESSURE_ATTEMPT = 4
# Retain review reads/deltas across repairs, then refresh from current source so
# an indefinitely recoverable run cannot accumulate an indefinitely large tail.
REVIEW_HISTORY_RECHECKS = 4
MAX_REVIEW_RECHECKS = 2           # source changed while being reviewed; take a fresh bounded snapshot
MAX_PLAN_AMENDMENTS = 2           # blocked-implementation plan patches per run
MAX_PLAN_AMENDMENTS_HARD = 6      # even a run that keeps finding new blockers stops here
BASE_EXECUTION_CALL_BUDGET = 56   # includes room for diagnosis and re-testing
# Each plan task adds room for two full implementation turns, each with its forced final call.
PER_TASK_CALL_BUDGET = 2 * (MAX_TOOL_ROUNDS_EXECUTION + 1)
# One repair must leave room for the reviewer's normal pass and its bounded
# evidence-recovery pass, including each forced conclusion call.
REPAIR_REVIEW_RESERVE = (MAX_TOOL_ROUNDS_REVIEW + 1) + (MAX_TOOL_ROUNDS_REVIEW_RECOVERY + 1)
MAX_AUTOMATIC_REPAIR_GRANTS = 2
AUTOMATIC_REPAIR_ALLOWANCE = MAX_TOOL_ROUNDS_EXECUTION + 1 + REPAIR_REVIEW_RESERVE
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
#: How the review notes that mean "carry on", not "fix a finding", begin. session.py keeps those
#: in the implementation conversation, so the notes are built from these and never retyped.
CALL_LIMIT_NOTE = "Implementation reached its call limit"
PLAN_RECOVERY_NOTE = "Plan recovery:"
INVALID_REPORT_NOTE = "Implementation did not return a valid"
CONTINUE_NOTES = (CALL_LIMIT_NOTE, PLAN_RECOVERY_NOTE, INVALID_REPORT_NOTE)

# Per-phase output stored in conversation_log. High so full phase text is retained
# (phase outputs are LLM summaries and are naturally well under this in practice).
MAX_PHASE_OUTPUT_CHARS = 60000

# Full-input pressure is the only trigger for rewriting retained tool history.
# A "round" is one assistant tool-call message plus all of its tool results.
MIN_KEEP_ROUNDS = 1           # latest call/result pair must survive into the next request
COMPACT_RESULT_PREVIEW = 140  # chars of each tool result kept in the compact summary
MAX_COMPACTED_HISTORY_CHARS = 12000
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
# Shorter results cost less than the call that would read them: at 1.5-2.5k chars its answer was
# 40-60% of the input, plus the call's own reasoning and latency.
READER_MIN_CHARS = 4000
READER_INPUT_CHARS = MAX_READ_RESULT_CHARS  # what the helper reads of one result, a whole read at most
# A purpose read of a whole file longer than this returns its outline instead of a helper reading it all.
PURPOSE_OUTLINE_LINES = 800
# The same in characters, for dense files: AGENT-0036's generated script was 126 lines but 26k characters.
PURPOSE_OUTLINE_CHARS = 40000
# The exact text of a file this request changes reaches the model as it is, never a summary of it.
READER_BYPASS = "[exact text: "
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
# Room for an adaptation's KEEP/CHANGE reference contract ahead of the change
# surface; the findings sit in the cached shared context, sent once per pass.
MAX_UNDERSTANDING_CONTEXT_CHARS = 12000
# A copy at least this long changes by edits: rewriting it whole drops what the reference does.
COPY_REWRITE_MIN_LINES = 150
CLIENT_SOURCE_SUFFIXES = (".js", ".css", ".html", ".vue")


def _server_calls(app_name: str, source: str) -> list[str]:
    """Dotted whitelisted-method paths of this app that a client source calls."""
    return _re.findall(rf"\b{_re.escape(app_name)}(?:\.[A-Za-z_][A-Za-z0-9_]*)+", source or "")
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

# Appended to the explore helper's question. A customized "Understand Prompt" may
# name the implementation tools (find_files, read_file), which the helper lacks.
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

There is no shell and no editing here: answer the question with what you found.
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


def _review_recovery_reason(decision: str, notes: str, payload, criteria: list[str]) -> str:
    """Say exactly what was wrong with the verdict, the way the plan repair does.

    ``review_decision`` already knows — a contradictory pass, a missing
    criterion, an unverified status — and a recovery prompt that only says
    "needs correction" makes the model guess at which rule it broke.
    """
    if decision == "needs_evidence" and isinstance(payload, dict):
        pending = [
            f"criterion {e.get('criterion')}: {str(e.get('evidence') or '').strip()[:200]}"
            for e in (payload.get("evidence") or []) if isinstance(e, dict) and e.get("status") == "unverified"
        ]
        return "Your verdict left these criteria unverified; fetch the source and decide each:\n- " + "\n- ".join(pending)
    if payload is None:
        return ("Your previous reply contained no JSON object with review_passed. Return the verdict as "
                f"JSON only, with exactly {len(criteria)} evidence entries, one per numbered criterion.")
    return "Your verdict was rejected: " + (notes or "invalid format") + (
        f" Expected exactly {len(criteria)} evidence entries, criteria numbered 1..{len(criteria)}."
    )


def _extract_review_json(text: str) -> dict | None:
    """Pull a {"review_passed": ...} object out of the model's response.

    Tries, in order: the whole response as JSON, a ```json fenced block,
    then the last such object embedded in prose (nested objects included).
    """
    candidates = [text.strip()]
    fence_match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and "review_passed" in parsed:
            return parsed

    decoder, found = json.JSONDecoder(), None
    for match in _re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(parsed, dict) and "review_passed" in parsed:
            found = parsed
    return found


# LLM factory — supports OpenAI, OpenRouter, Gemini and Claude

_OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1"


def _openrouter_base_url() -> str:
    """OpenRouter, or a local OpenAI-compatible proxy for testing (koda-local/codex-proxy).

    The override is honoured only on a loopback host: the OpenRouter key goes with every
    request, and an environment variable must not be able to send it anywhere else.
    """
    from urllib.parse import urlparse

    override = (os.environ.get("KODA_OPENROUTER_BASE_URL") or "").strip()
    if not override:
        return _OPENROUTER_DEFAULT_URL
    if (urlparse(override).hostname or "") in {"localhost", "127.0.0.1", "::1"}:
        return override
    print(f"[koda] KODA_OPENROUTER_BASE_URL ignored: {override!r} is not a loopback address")
    return _OPENROUTER_DEFAULT_URL


OPENROUTER_BASE_URL = _openrouter_base_url()


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


# Review turns (first pass, recovery pass, forced verdict, every recheck) run at this effort.
# Replays of run 5's verdict call found the same P1 at medium and low (6/6); through a text-only tool
# format low often claimed its tools were unavailable (AGENT-0034), medium did not.
# Agent Settings "Review Reasoning Effort" overrides it; it only ever lowers the configured effort.
REVIEW_REASONING_EFFORT = 'medium'
_EFFORT_ORDER = ('low', 'medium', 'high')


def _lower_effort(configured: str | None, wanted: str | None) -> str | None:
    """``wanted`` when it is a known effort below ``configured``; None keeps the configured one."""
    if configured not in _EFFORT_ORDER or wanted not in _EFFORT_ORDER:
        return None
    return wanted if _EFFORT_ORDER.index(wanted) < _EFFORT_ORDER.index(configured) else None


def _phase_reasoning_effort(phase: str, provider: str, model: str) -> str | None:
    """The effort override for one phase's turn, or None for the configured effort.

    Effort is part of the provider's prompt-cache key: in the replays, each change of effort on
    a byte-identical prompt read 0 cached tokens, even for the tools and system prefix. So effort
    changes only where a conversation starts (the review has its own), never inside one.
    """
    if phase != 'Reviewing':
        return None
    try:
        setting = frappe.db.get_single_value('Agent Settings', 'review_reasoning_effort')
    except Exception:  # noqa: BLE001 - an unmigrated site has no such column
        setting = None  # installations not yet migrated use the default
    setting = str(setting or REVIEW_REASONING_EFFORT)
    if setting.lower().startswith('same'):
        return None
    return _lower_effort(_configured_reasoning_effort(provider, model), setting)


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


SUBMIT_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "plan": PLAN_JSON_SCHEMA,
        "findings": {"type": "string", "description": "Verified path:line evidence for the user and reviewer."},
    },
    "required": ["plan", "findings"],
}

#: Repair rounds a failing test may trigger before it is reported instead of gating.
MAX_TEST_REPAIRS = 2
#: Checks that the verification suite was not weakened or broken; they always gate.
TEST_INTEGRITY_CHECKS = ("tests:configuration", "tests:contract:", "tests:frozen:", "tests:mocked:",
                         "tests:changed-during-run", "tests:environment")

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
            and len(result) >= READER_MIN_CHARS and not result.startswith(READER_BYPASS))


# Helpers

def _app_file_exists(app_name: str, rel_path: str) -> bool:
    """True if rel_path resolves to a real file inside the app (best-effort)."""
    try:
        return os.path.isfile(agent_tools._resolve_path(app_name, rel_path))
    except PermissionError:
        raise
    except (OSError, ValueError):
        return False


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
            # The form deduplicates on this; display text and a one-second
            # timestamp collapsed distinct calls to the same tool.
            "event_id": os.urandom(16).hex(),
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
    so read-only safety is enforced here, not by hiding schemas. A session adds
    ``submit_plan`` and ``explore``.

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
        # Refused before the snapshot: the before-content goes into the baseline the reviewer diffs.
        pattern = agent_tools.redaction_pattern(app_name, path, full)
        if pattern:
            raise ValueError(f"{path} matches the redaction pattern {pattern}; secrets are never read or written.")
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
        A file of up to 2000 lines comes back whole; read it once, not in slices. Longer files come back
        as a summary (signatures kept, long bodies elided); then fetch every span you need in ONE call with
        ranges="40-80,120-160". Edit_file receipts show the edited region."""
        full = agent_tools._resolve_path(app_name, path)
        snapshot = read_snapshot(full)
        # A final newline ends the last line; it does not start another (an 800-line file is not 801).
        lines = snapshot.count("\n") + (not snapshot.endswith("\n")) if snapshot is not None else 0
        whole = not (start_line or end_line or str(ranges or "").strip())
        chars = len(snapshot) if snapshot is not None else 0
        if (str(purpose or "").strip() and whole
                and (lines > PURPOSE_OUTLINE_LINES or chars > PURPOSE_OUTLINE_CHARS)):
            # A helper reading all of a long file costs as much as the file, and what the
            # purpose needs is a few spans of it (AGENT-0036 read a 1,624-line script whole twice).
            outline = agent_tools.get_file_outline(app_name, path)
            if not _tool_result_succeeded(outline):
                return outline  # a redacted file is refused, not outlined
            return (f"[outline only: {path} has {lines} lines ({chars} characters), too long to read whole "
                    "for a purpose. Its "
                    "outline follows: read the spans the purpose needs in ONE call with ranges=\"a-b,c-d\" (with "
                    "the purpose to have them explained, without it for the exact text).]\n"
                    + outline)
        result = agent_tools.read_file(app_name, path, start_line, end_line, ranges)
        canonical = os.path.relpath(full, os.path.realpath(agent_tools._app_root(app_name))).replace("\\", "/")
        if (str(purpose or "").strip() and not read_only and _tool_result_succeeded(result)
                and (canonical in copied or (before is not None and canonical in before))):
            # A file this request changes is read to be edited, and an edit needs the exact
            # text: a helper's summary would only force a second read of the same file.
            result = f"{READER_BYPASS}{path} is changed by this request, so this is its exact text.]\n{result}"
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
        For Python tests, database writes roll back, file writes, background jobs and email are discarded,
        and schema changes, commits and child processes are refused. Network calls to external services
        are NOT contained and do reach them.
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
        string of its keyword arguments, e.g. '{"customer": "CUST-0001", "limit": 20}'. To look at real
        records and columns before designing a query, method may also be a read probe: frappe.get_all,
        frappe.db.get_value, frappe.db.count or frappe.db.sql with '{"query": "SELECT ... LIMIT 20"}'
        (one read-only statement). Runs as Administrator
        with real records and schema: database writes are rolled back, file writes, background jobs and email
        are discarded (the result says which), and commits and child processes are refused. Network calls to
        external services are NOT rolled back, so do not call functions that post to one. Use it to see what a query
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
            if not path.endswith(CLIENT_SOURCE_SUFFIXES):
                return (f"WRITE_FAILED: {path} was copied from {reference} so that it keeps the reference's "
                        "behavior. Change it with edit_file on the parts the task changes (several edits in one "
                        "response are fine); rewriting it whole drops what the reference does.")
            # A client script may be redesigned whole (a request for a different UI), but not lose
            # the server contract the copy carried.
            # By method name inside a string or template literal: a rewrite may build the dotted path
            # from a base (`${methodBase}trace_batch`), which the full-path pattern misses, and a name
            # left only in a comment or an identifier is not a call.
            dropped = sorted(call for call in set(_server_calls(app_name, current))
                             if not _re.search(r"[\"'`][^\"'`\n]*\b" + _re.escape(call.rsplit(".", 1)[-1])
                                               + r"\b[^\"'`\n]*[\"'`]", content))
            if dropped:
                return (f"WRITE_FAILED: {path} was copied from {reference}; this rewrite no longer calls "
                        + ", ".join(dropped) + ". Keep every server call, its parameters and the states "
                        "around it (loading, empty, error) when you redesign the page.")
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
            # This tool wrote these bytes, so replacing the copy whole needs no read of it first:
            # AGENT-0029 was refused, then read a 60k-char copy it was about to overwrite.
            observed[destination] = read_snapshot(destination)
            if known is not None and not any("\n" in old + new for old, new in replacements.items()):
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
        target, destination, destination_content = capture_write(destination_path)
        known = current if current is not None else observed.get(full)
        if known is None and before is not None and before.get(destination) is None:
            known = before.get(source)
        expected = revision(known) if known is not None else ""
        if known is not None:
            checkpoint.write_intent({source: current, destination: destination_content},
                {source: None, destination: known}, move={"source": source, "destination": destination, "sha256": expected})
        result = agent_tools.rename_file(app_name, source_path, destination_path, expected_sha256=expected)
        if result.startswith("RENAME_OK:"):
            # The bytes the model read moved with the file: the new path is read, the old one is gone.
            observed.pop(full, None)
            moved = read_snapshot(target)
            if moved is not None:
                observed[target] = moved
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
    if session is None:
        return catalogue

    # A session keeps one catalogue in every phase (tools precede messages in
    # the provider's cache key); each phase decides what these two may do.
    plan_sink, explorer = session.get("plan_sink"), session.get("explorer")

    def submit_plan_call(plan, findings: str = "") -> str:
        if plan_sink is None:
            return "SUBMIT_FAILED: The plan is already approved. Implement it; do not submit another plan."
        return plan_sink.accept(plan, findings)

    # The plan travels as a typed object: escaping a whole plan inside a string argument fails.
    submit_plan = StructuredTool.from_function(
        func=submit_plan_call, name="submit_plan", args_schema=SUBMIT_PLAN_SCHEMA,
        description=(
            "Submit your implementation plan for the user's approval; only during investigation. "
            "plan follows the schema; findings is what you verified, as path:line evidence, for the user "
            "and the reviewer: current behavior, the reference contract (KEEP/CHANGE) for an adaptation, "
            "the full inventory for an exhaustive request, and what your tools could not verify. "
            "Validation errors come back as the result; fix them and submit again."))

    @tool
    def explore(question: str) -> str:
        """Hand a broad question to a helper that searches and reads many files in its own context and
        returns only path:line findings (about 1-2k tokens). Use it for questions that span the app ("where
        is X decided", "every place that hardcodes Y"); read a file yourself when you know which one."""
        if explorer is None:
            return "EXPLORE_UNAVAILABLE: search and read directly."
        return explorer(question)

    return [*catalogue, submit_plan, explore]


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
    # Keep both ends: a traceback's cause and a test run's verdict are at the tail.
    head = result[:limit // 2].rsplit('\n', 1)[0]
    tail = result[-(limit // 2):].split('\n', 1)[-1]
    omitted = len(result) - len(head) - len(tail)
    return (f'[{len(result):,} chars, {result.count(chr(10)) + 1:,} lines; the middle {omitted:,} chars are '
            'omitted. Request a narrower path or source range for them.]\n'
            + head + f'\n… [{omitted:,} chars omitted] …\n' + tail)


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
        follow = ("this file is long, so read the spans you need with ranges=\"a-b,c-d\""
                  if raw.startswith("[outline only: ") else "call without purpose for the exact text")
        return (f"[{what}, read for: {purpose[:200]}]\n{answer}\n"
                f"[A helper read {len(raw):,} characters for this answer; {follow}.]")

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
        # A compaction resends the whole retained history uncached, so leave room
        # for about two thirds of the limit before the next one.
        target = min(cleanup_target(limit), limit // 3)
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
                        # The exact-text header is not content: the same text read plainly is the same bytes.
                        seen_key = result.split("\n", 1)[-1] if result.startswith(READER_BYPASS) else result
                        result_round = seen_results.get(seen_key)
                        if result_round is not None and result_round in retained_numbers:
                            result = (
                                f"[{name} returned bytes identical to round {result_round}; "
                                "use the earlier result and move on.]"
                            )
                        else:
                            seen_results[seen_key] = label
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
        # Numbered with the round it is saved with, so a resumed pass does not reuse its label.
        history["rounds_done"] = label
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


def _implementation_advisor(state: dict, provider: str, model: str, request_name: str):
    """An advisor for this implementation pass, or None without a task."""
    if not state.get("execution_tasks"):
        return None
    _, criteria, _ = _execution_context(state)
    contract = ("## USER REQUEST\n" + str(state.get("user_message") or "")[:3000]
                + "\n\n## ACCEPTANCE CRITERIA\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, 1)))
    # Low effort: it reads one round's diff, it does not plan.
    return Advisor(_get_llm(provider=provider, model=model, session_id=request_name, reasoning_effort="low"),
                   contract)


def _run_agent_turn(state: dict, phase: str, prompt: str, read_only_tools: bool, max_rounds: int = 20,
                    history: dict | None = None, session: dict | None = None) -> dict:
    """Run one agent turn for the given phase. Returns state updates.

    ``history`` lets a follow-on turn (reviewer recovery) continue from the
    retained tool rounds of the previous one instead of starting cold.

    ``session`` runs the turn as one phase of the request-long conversation, extending one
    cached prefix. Keys: ``plan_sink``, ``explorer``, ``stop_when``, ``after_round``.
    """
    before = dict(state.get("execution_baseline") or {}) if not read_only_tools else {}
    file_moves = []
    progress = {"tokens": state.get("tokens_used", 0), "calls": 0}
    budget_updates = {}
    verified = {'value': state.get('verification_progress') or {},
                'receipts': state.get('verification_receipts') or []}
    def observe_verification(report, receipts):
        verified['value'] = repair_budget.observe(verified['value'], report, receipts)
        paths = set(before) | set(state.get('execution_baseline') or {})
        paths.update(path for task in state.get('execution_tasks', []) for path in task.get('files', []))
        revisions = {path: revision(_read_current(state, path)) for path in sorted(paths)}
        verified['receipts'] = receipts
        for receipt in receipts:
            receipt['source_revisions'] = dict(revisions)
        checkpoint.update(verification_progress=verified['value'], verification_receipts=receipts)
    try:
        app_name = state.get("target_app_name", "")
        provider = state.get("ai_provider", "OpenAI")
        model = state.get("ai_model", "gpt-4o-mini")
        request_name = state.get("request_name", "")

        remaining = state.get("tool_rounds_limit", 1000) - state.get("tool_rounds_used", 0)
        if not read_only_tools and state.get("execution_tasks"):
            remaining -= REPAIR_REVIEW_RESERVE  # a repair must leave the review that checks it room to run
        if remaining < min(max_rounds + 1, MIN_REPAIR_ROUNDS + 1):
            budget_updates = repair_budget.extend(state, allowance=AUTOMATIC_REPAIR_ALLOWANCE,
                                                  max_grants=MAX_AUTOMATIC_REPAIR_GRANTS)
            if budget_updates:
                state = {**state, **budget_updates}
                verified['value'] = state['verification_progress']
                remaining += AUTOMATIC_REPAIR_ALLOWANCE
                checkpoint.update(**budget_updates)
                _publish_agent_log(request_name, 'repair_budget_extended',
                    grants=state['automatic_repair_budget_grants'], call_limit=state['tool_rounds_limit'],
                    reason='Executed tests show improvement or completed behavior awaiting review.')
        if remaining < 2:
            return {"review_stopped": "The execution call budget ran out before the work passed review."}
        max_rounds = min(max_rounds, remaining - 1)
        verification_contract = state.get("verification_contract")
        if verification_contract is None:
            verification_contract = _prepare_verification_contract(state)
            checkpoint.update(verification_contract=verification_contract)
        copied_files = dict(state.get("copied_files") or {})
        reader = tool_reader(provider, model, request_name)
        tools = _make_tools(
            app_name, read_only=read_only_tools,
            before=before, file_moves=file_moves,
            delete_paths=_approved_deletions(state),
            verification_contract=verification_contract,
            verification_observer=observe_verification,
            copied_files=copied_files,
            session=session,
            reader=reader,
        )
        llm = _get_llm(provider=provider, model=model, session_id=request_name,
                       reasoning_effort=_phase_reasoning_effort(phase, provider, model))
        system_prompt = get_system_prompt(app_name or "target_app", request_name=request_name)
        shared_context = (_shared_request_context(state)
                          if state.get("execution_tasks") and (session is None or session.get("shared_context"))
                          else "")
        content, tool_edited_paths, total_tokens, rounds_used, exhausted = _run_tool_calling_loop(
            llm, tools, system_prompt, prompt,
            request_name=request_name,
            max_rounds=max_rounds,
            state=state,
            provider=provider,
            require_writes=not read_only_tools,
            progress=progress,
            history=history,
            validate_final=completion_report_problem if phase == "Implementing" else None,
            shared_context=shared_context,
            cache_phase=phase.lower(),
            advisor=_implementation_advisor(state, provider, model, request_name)
            if phase == "Implementing" and not read_only_tools else None,
            stop_when=(session or {}).get("stop_when"),
            reader=reader,
            after_round=(session or {}).get("after_round"),
        )
        content = _message_content_to_str(content)
        max_output = MAX_PHASE_OUTPUT_CHARS
        steps = list(state.get("intermediate_steps") or []) + [
            {"phase": phase, "output": (content[:max_output] if content else "")}
        ]
        result = {
            **budget_updates,
            "current_stage": phase,
            "intermediate_steps": steps,
            "tokens_used": total_tokens,
            "tool_rounds_used": state.get("tool_rounds_used", 0) + rounds_used,
            "turn_exhausted": exhausted,
            "_write_baseline": before,
            "_file_moves": file_moves,
            "copied_files": copied_files,
            "verification_contract": verification_contract,
            'verification_progress': verified['value'],
            'verification_receipts': verified['receipts'],
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
            **budget_updates,
            "current_stage": phase,
            "intermediate_steps": steps,
            "_write_baseline": before,
            "_file_moves": file_moves,
            "tokens_used": progress["tokens"],
            "tool_rounds_used": state.get("tool_rounds_used", 0) + progress["calls"],
            "error": str(e),
            'verification_progress': verified['value'],
            'verification_receipts': verified['receipts'],
        }


# Planner calls: the plan coverage check and blocked-implementation amendments

PLAN_PATCH_ROUNDS = 2
"""Repair rounds that ask for edits to a rejected plan amendment. Each costs a
cached-prefix read plus a few hundred new tokens."""


def _plan_run_config(run_name: str, *, round_number: int, mode: str) -> dict:
    """Trace name, tags and round metadata for one planner call."""
    return {
        "run_name": run_name,
        "tags": ["plan", f"plan:{mode}"],
        "metadata": {"plan_round": round_number, "plan_mode": mode},
    }


def _plan_path_candidates(app_name: str):
    """``path -> nearest existing files``, over one lazy walk of the app."""
    files: list[str] = []
    walked = False

    def candidates_for(path: str) -> list[str]:
        nonlocal walked
        if not walked:
            walked = True
            try:
                from ampower_koda.agent.core import LocalWorkspace
                from pathlib import Path
                files.extend(LocalWorkspace(Path(agent_tools._app_root(app_name))).list_files())
            except Exception:
                log_agent_error("Agent Graph: plan path candidates", frappe.get_traceback())
        return nearest_paths(path, files)

    return candidates_for


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


PLAN_COVERAGE_SCHEMA = {
    "title": "PlanCoverage",
    "description": "Each clause of the user's request, and whether the plan does what it says.",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "quote": {"type": "string"},
                    "status": {"type": "string", "enum": ["covered", "narrowed", "missing", "contradicted", "added"]},
                    "task": {"type": "string"},
                    "problem": {"type": "string"},
                },
                "required": ["quote", "status", "task", "problem"],
            },
        },
    },
    "required": ["clauses"],
}

PLAN_COVERAGE_PROMPT = """You check an implementation plan against the user's request before a human approves it.
You have not seen the codebase analysis the plan came from, on purpose: judge only whether the plan does
what the request's words say. Split the request into its clauses, one per distinct thing asked, including
what it asks to keep the same (for example "an identical copy of X" keeps X's behavior). For each clause,
quote the request's own words exactly, then give its status: covered; narrowed (the plan does less than
the words, or something different); missing; or contradicted. Name the task id that covers it, or "",
and in one sentence say what is wrong. Do not add requirements the words do not state. An ambiguous
clause the plan reads reasonably, and records in its assumptions, is covered, unless the assumptions
name another reading that keeps more of what already exists: the plan should build that one, so the
clause is narrowed. A control or limit the plan keeps from the thing being copied (a depth, page size,
date range or filter) does not narrow a clause the user can still reach through that control, for
example by raising the depth or expanding a node: mark narrowed only what the plan cannot deliver at all.
Then add one entry per task that changes existing code or behavior no clause asks to change, with status
"added" and the task's title as the quote; work a requested change needs (its own wiring, registration
or tests) is not added, and neither are the loading, error, empty and keyboard states any new page
needs."""


def _plan_text(text: str) -> str:
    """The request as plain lowercase words, for matching quoted clauses."""
    text = _re.sub(r"<[^>]+>", " ", html.unescape(str(text or "")))
    return " ".join(text.lower().split())


def _plan_coverage_gaps(plan: dict, user_message: str, *, llm, provider: str, budget: dict,
                        request_name: str, total_tokens: int) -> tuple[list[str], int]:
    """Clauses of the request the plan drops or reinterprets, from a model that never saw the investigation.

    A quote not in the request's own words is discarded, so the checker cannot add requirements.
    """
    request = _plan_text(user_message)
    if not request:
        return [], total_tokens
    messages = [SystemMessage(content=PLAN_COVERAGE_PROMPT),
                HumanMessage(content=f"## USER REQUEST\n{request}\n\n## PLAN\n{plan_to_markdown(plan)}")]
    try:
        response, total_tokens = _invoke_structured(
            llm, provider, PLAN_COVERAGE_SCHEMA, messages,
            _plan_run_config("plan:coverage", round_number=0, mode="coverage"),
            budget=budget, request_name=request_name, total_tokens=total_tokens, round_label=1)
        _, parsed = _unpack_structured_plan(response)
        clauses = [c for c in (parsed or {}).get("clauses") or [] if isinstance(c, dict)]
    except Exception:  # noqa: BLE001 - a second opinion must never block planning
        log_agent_error("Agent Graph: plan coverage check", frappe.get_traceback())
        return [], budget.get("total_tokens", total_tokens)
    titles = {_plan_text(task.get("title")): task.get("id", "") for task in plan.get("tasks") or []}
    gaps = []
    for clause in clauses:
        words, problem, task = (str(clause.get(key) or "").strip() for key in ("quote", "problem", "task"))
        quote = _plan_text(words)
        if clause.get("status") == "added":
            if quote in titles:
                gaps.append(f"{titles[quote]} \"{words}\" changes what the request does not ask to change: "
                            f"{problem} Drop the task and report the defect in risks, or state in assumptions "
                            "why the request needs it.")
            continue
        if clause.get("status") == "covered" or len(quote) < 8 or quote not in request:
            continue
        verb = {"narrowed": "narrows", "missing": "misses",
                "contradicted": "contradicts"}.get(clause.get("status"), "does not cover")
        gaps.append(f"The plan {verb} the request's words \"{words}\""
                    f"{' (' + task + ')' if task else ''}: {problem} "
                    "Edit the plan so it does what these words ask, or state in assumptions why this reading is right.")
    return gaps, total_tokens


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


def _check_plan_input(state: dict, messages: list, schema: dict, budget: dict) -> None:
    estimate = estimate_messages(messages) + serialized_tokens(schema)
    window, ceiling = koda_core.request_limits(state.get("target_app_name", ""), state.get("ai_model", "gpt-4o-mini"))
    limit = input_limit(window, budget["max_tokens"], ceiling)
    if estimate > limit:
        raise ValueError(f"Planning context exceeds the input budget ({estimate:,} > {limit:,} tokens).")


# Graph nodes — Execution phase

def _persist_execution_plan(request_name: str, plan: dict) -> None:
    if request_name:
        values = {
            "plan_json": json.dumps(plan, ensure_ascii=True),
            "approved_plan_json": json.dumps(plan, ensure_ascii=True),
            "agent_plan": plan_to_markdown(plan),
        }
        active = checkpoint.journal()
        if active:
            active.state.update(plan_object=plan, execution_tasks=plan["tasks"])
            if active.node == "review":
                active.node = "implement"
                active.state.update(review_attempts=0, task_completion={},
                    review_fingerprint="", review_fingerprints_seen=[],
                    review_failure_fingerprints_seen=[],
                    plan_amendments=int(active.state.get("plan_amendments") or 0) + 1,
                    review_notes="Continue the blocked task using the amended file scope.")
            active.refresh()
            values["execution_checkpoint"] = json.dumps(active.payload(), ensure_ascii=True)
        set_request_value(request_name, values)
        frappe.db.commit()


def prepare_execution_node(state: dict) -> dict:
    """Freeze the validated contract once, before any write: the session implements the whole plan."""
    try:
        scope_repairs = []
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
            app_name = state["target_app_name"]
            exists = lambda p: _app_file_exists(app_name, p)
            approved = load_plan(state.get("plan_object"))
            completed = complete_rename_file_scope(approved, exists, app_name=app_name)
            plan = validate_plan(completed.plan)
            scope_repairs = list(completed.repairs)
            if scope_repairs:
                _persist_execution_plan(state.get("request_name", ""), plan)
                _publish_agent_log(state.get("request_name", ""), "llm_response", round=0,
                                   preview="Approved rename scope completed before execution: " + "; ".join(scope_repairs))
            tasks = plan["tasks"]
        return {
            "plan_object": plan, "execution_tasks": tasks,
            "task_results": [], "execution_baseline": {},
            "file_moves": list(state.get("prior_file_moves") or []) if state.get("is_follow_up") else [], "plan_amendments": 0,
            "copied_files": {}, "test_repair_rounds": 0,
            "plan_scope_repairs": scope_repairs,
            "task_completion": {}, "turn_exhausted": False,
            "review_attempts": 0, "review_notes": "", "review_passed": False,
            "review_fingerprint": "", "review_fingerprints_seen": [],
            "review_failure_fingerprints_seen": [],
            "repair_strategy_level": 0, "review_format_retries": 0,
            "verification_contract": _prepare_verification_contract({**state, "execution_tasks": tasks}),
            "verification_receipts": [],
            "review_repairable": False, "review_retry_requested": False, "review_rechecks": 0,
            "review_history": {},
            "tool_rounds_used": 0,
            "tool_rounds_limit": BASE_EXECUTION_CALL_BUDGET + PER_TASK_CALL_BUDGET * len(tasks),
            'automatic_repair_budget_enabled': True, 'automatic_repair_budget_grants': 0,
            'verification_progress': {},
        }
    except Exception as exc:
        return {"error": str(exc)}


#: The only edits an amendment may make: extend file/context scope, or turn MODIFY into CREATE.
AMENDMENT_OPS = frozenset({"add_file", "add_context_ref", "set_action"})
#: A validation repair may also withdraw a reference; the approved ones are checked afterwards.
AMENDMENT_REPAIR_OPS = AMENDMENT_OPS | {"remove_context_ref"}
AMENDMENT_SCOPE_ONLY = "An amendment may only extend required file/context scope."


def _amendment_ops_only(parsed: dict, allowed=AMENDMENT_OPS) -> bool:
    return all(e.get("op") in allowed for e in parsed.get("edits", []) if isinstance(e, dict))


def _amendment_keeps_approved(tasks: list[dict], new_tasks: list[dict]) -> bool:
    """Every approved outcome, description, dependency, file and reference is still in the plan."""
    def kept(t):
        return t["id"], t["goal"], t["acceptance_criteria"], t.get("description"), t.get("depends_on")

    def refs(t):
        return {(r.get("path"), r.get("start"), r.get("end")) for r in t.get("context_refs") or []}

    return (len(tasks) == len(new_tasks) and all(
        kept(old) == kept(new) and set(old["files"]) <= set(new["files"]) and refs(old) <= refs(new)
        for old, new in zip(tasks, new_tasks)))


def _amend_plan_for_blocker(state: dict, completion: dict) -> tuple[dict | None, str]:
    """Ask the planner for edits that clear an implementation blocker.

    The planner sees the plan, the implementer's report and the nearest real files
    for each named path. Approved outcomes are frozen; the amended plan is validated,
    then persisted to the request.
    """
    if state.get("is_follow_up"):
        return None, "Follow-up work uses its own fixed outcome; no plan amendment attempted."
    spent = int(state.get("plan_amendments") or 0)
    blocker_key = " ".join(str(completion.get("summary") or "").split())[:500]
    repeated = bool(blocker_key) and blocker_key == (state.get("plan_last_blocker") or "")
    if spent >= MAX_PLAN_AMENDMENTS_HARD or (spent >= MAX_PLAN_AMENDMENTS and repeated):
        return None, (f"Plan already amended {spent} time(s) this run"
                      + (" for this same blocker" if repeated else "") + "; not amending again.")
    request_name = state.get("request_name", "")
    app_name = state.get("target_app_name", "")
    provider = state.get("ai_provider", "OpenAI")
    plan = state.get("plan_object") or {}
    tasks = list(state.get("execution_tasks") or [])
    tokens_used = int(state.get("tokens_used") or 0)
    try:
        completed = complete_rename_file_scope(plan, lambda p: _app_file_exists(app_name, p), app_name=app_name)
        if completed.repairs:
            amended = validate_plan(completed.plan)
            _persist_execution_plan(request_name, amended)
            return {
                "plan_object": amended, "execution_tasks": amended["tasks"],
                "plan_amendments": spent + 1, "plan_last_blocker": blocker_key, "tokens_used": tokens_used,
                "plan_scope_repairs": [*(state.get("plan_scope_repairs") or []), *completed.repairs],
            }, "Completed the already-approved rename's file scope without another model call. Retrying."
        llm = _get_llm(provider=provider, model=state.get("ai_model", "gpt-4o-mini"), session_id=request_name)
        budget = _plan_budget(provider, state)
        candidates_for = _plan_path_candidates(app_name)
        blocker = " ".join(
            [str(completion.get("summary") or "")] + [str(u) for u in completion.get("unverified") or []]
        ).strip()
        # Paths the report names, plus the plan's own files. Spaces are
        # excluded from the token so a sentence is not swallowed whole; a name
        # with a space inside still surfaces through the approved-files list.
        mentioned = _re.findall(r"[\w./-]+/[\w.-]+\.[A-Za-z0-9]{1,6}", blocker)
        lookups = []
        planned = [p for task in tasks for p in [*task.get("files", []),
                                                  *(ref.get("path", "") for ref in task.get("context_refs", []))]]
        for path in dict.fromkeys([*mentioned, *planned]):
            path = path.strip().lstrip("/")
            if not path:
                continue
            exists = _app_file_exists(app_name, path)
            near = candidates_for(path)
            lookups.append(f"- {path}: {'exists' if exists else 'does not exist'}"
                           + (f"; nearest existing: {', '.join(near)}" if near and not exists else ""))
        feedback = (
            "## IMPLEMENTATION BLOCKED — AMEND THE PLAN\n"
            "The approved plan could not be completed. The implementer reported:\n"
            f"{blocker}\n\n"
            "Files named in the report or the plan, checked on disk:\n" + ("\n".join(lookups) or "- (none)") + "\n\n"
            "Return PlanPatch edits that make the approved outcomes completable. All approved outcomes and "
            "existing task details are frozen: only add_file, add_context_ref, and set_action (MODIFY to CREATE "
            "for a newly added missing file) are allowed. Add only the missing paths needed to meet the existing "
            "approved outcomes. A task's `files` is the implementer's starting point, not a permission list, so "
            "do not add paths merely to unblock a write. Do not change DELETE tasks, remove paths, change "
            "dependencies, or change goals, scope, assumptions or risks; do not weaken the requested outcome to "
            "clear a blocker."
        )
        messages = [
            _build_system_message(provider, get_system_prompt(app_name or "target_app", request_name=request_name),
                                  state.get("ai_model", "")),
            HumanMessage(content="## CURRENT PLAN\n" + json.dumps(plan, ensure_ascii=False)),
            HumanMessage(content=feedback),
        ]
        _publish_agent_log(request_name, "llm_response", round=0,
                           preview=f"Implementation blocked; asking the planner for edits: {blocker[:300]}")
        _check_plan_input(state, messages, PLAN_PATCH_SCHEMA, budget)
        check_active(reserve=MODEL_TIME_RESERVE)
        response, tokens_used = _invoke_structured(
            llm, provider, PLAN_PATCH_SCHEMA, messages,
            _plan_run_config(f"plan:amend:{spent + 1}", round_number=spent + 1, mode="amend"),
            budget=budget, request_name=request_name, total_tokens=tokens_used, round_label=0,
        )
        check_active()
        raw_response, parsed = _unpack_structured_plan(response)
        failed = {"tokens_used": tokens_used, "plan_amendments": spent + 1, "plan_last_blocker": blocker_key}
        if not _amendment_ops_only(parsed):
            return failed, AMENDMENT_SCOPE_ONLY
        result = apply_plan_patch(plan, parsed)
        if not result.applied:
            return {"tokens_used": tokens_used, "plan_amendments": spent + 1, "plan_last_blocker": blocker_key}, (
                "Planner returned no applicable edits" + (": " + "; ".join(result.rejected) if result.rejected else "") + ".")
        # An amended plan that fails validation is sent back with the exact
        # issues, the same way plan repair works, instead of being discarded.
        for repair in range(PLAN_PATCH_ROUNDS + 1):
            try:
                amended = validate_plan(result.plan)
                break
            except PlanValidationError as exc:
                if repair >= PLAN_PATCH_ROUNDS:
                    return {"tokens_used": tokens_used, "plan_amendments": spent + 1, "plan_last_blocker": blocker_key}, (
                        "Plan amendment failed validation after " + str(PLAN_PATCH_ROUNDS) + " repair round(s): "
                        + "; ".join(exc.issues)[:300])
                issues = list(exc.issues) + list(result.rejected)
                messages.extend([
                    AIMessage(content=json.dumps(parsed, ensure_ascii=False)),
                    HumanMessage(content=repair_feedback(issues)),
                ])
                _publish_agent_log(request_name, "llm_response", round=0,
                                   preview=f"Amended plan rejected ({len(issues)} issue(s)); asking for corrections: {issues[0]}"[:400])
                _check_plan_input(state, messages, PLAN_PATCH_SCHEMA, budget)
                check_active(reserve=MODEL_TIME_RESERVE)
                response, tokens_used = _invoke_structured(
                    llm, provider, PLAN_PATCH_SCHEMA, messages,
                    _plan_run_config(f"plan:amend:{spent + 1}:repair:{repair + 1}", round_number=spent + 1, mode="amend"),
                    budget=budget, request_name=request_name, total_tokens=tokens_used, round_label=0,
                )
                check_active()
                raw_response, parsed = _unpack_structured_plan(response)
                if not _amendment_ops_only(parsed, AMENDMENT_REPAIR_OPS):
                    return {**failed, "tokens_used": tokens_used}, AMENDMENT_SCOPE_ONLY
                result = apply_plan_patch(result.plan, parsed)
        new_tasks = amended["tasks"]
        if (any(amended[key] != plan.get(key) for key in ("scope", "assumptions", "risks", "overview"))
                or not _amendment_keeps_approved(tasks, new_tasks)):
            return {"tokens_used": tokens_used, "plan_amendments": spent + 1, "plan_last_blocker": blocker_key}, "Planner edits changed the approved outcomes or scope; discarded."
        for old, new in zip(tasks, new_tasks):
            if (old["action"] == "DELETE" and old != new) or (
                new["action"] != old["action"] and not (old["action"] == "MODIFY" and new["action"] == "CREATE")
            ):
                return {"tokens_used": tokens_used, "plan_amendments": spent + 1, "plan_last_blocker": blocker_key}, "An amendment cannot change approved removal operations."
        _persist_execution_plan(request_name, amended)
        summary = "; ".join(
            f"{e.get('op')} {e.get('task_id')} {e.get('from') or e.get('path') or e.get('field') or ''}".strip()
            for e in (parsed.get("edits") or []) if isinstance(e, dict)
        )[:600]
        _publish_agent_log(request_name, "llm_response", round=0,
                           preview=f"Plan amended ({result.applied} edit(s)): {summary}")
        return {
            "plan_object": amended, "execution_tasks": new_tasks,
            "plan_amendments": spent + 1, "plan_last_blocker": blocker_key, "tokens_used": tokens_used,
        }, f"Plan amended with {result.applied} edit(s): {summary}. Retrying against the amended plan."
    except Exception as exc:
        log_agent_error("Agent Graph: plan amendment", f"request={request_name}\n{exc}\n{frappe.get_traceback()}")
        return {"tokens_used": tokens_used, "plan_amendments": spent + 1, "plan_last_blocker": blocker_key}, f"Plan amendment failed: {str(exc)[:300]}"


def _execution_context(state: dict) -> tuple[dict, list[str], list[str] | None]:
    """The whole plan as one unit, every task's acceptance criteria, and the paths the plan names.

    The paths seed the before/after baseline the reviewer reads; they do not
    bound what the implementer may write.
    """
    tasks = state["execution_tasks"]
    active = {"id": "PLAN", "tasks": tasks}
    criteria = [f"{t['id']}: {c}" for t in tasks for c in t["acceptance_criteria"]]
    paths = list(dict.fromkeys(p for t in tasks for p in t["files"]))
    return active, criteria, None if state.get("is_follow_up") else paths


def _approved_plan_context(state: dict) -> dict:
    plan = state.get("plan_object") or {}
    return {key: plan.get(key, [] if key in {"assumptions", "risks"} else {})
            for key in ("overview", "scope", "assumptions", "risks")}


def _model_receipts(receipts) -> list[dict]:
    """Receipts as the reviewer reads them: an inline runner program is named by its hash, not pasted.

    The unittest runner is ~17.7k characters of Koda's own code in every argv; the stored receipt keeps it
    whole, because freezing tests checks the test paths against that argv.
    """
    compact = []
    for receipt in receipts or []:
        if not isinstance(receipt, dict):
            continue
        argv = [f"<{receipt.get('name') or 'runner'} program, {len(arg)} chars, sha256 "
                f"{hashlib.sha256(arg.encode()).hexdigest()[:12]}>"
                if isinstance(arg, str) and ("\n" in arg or len(arg) > 500) else arg
                for arg in receipt.get("argv") or []]
        compact.append({**receipt, "argv": argv} if "argv" in receipt else receipt)
    return compact


def _compact_execution_results(results) -> list[dict]:
    """The model's claims without past reviews or receipts, which the review supplies fresh."""
    compact = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        item = {key: result.get(key) for key in ("task_id", "status")
                if result.get(key) not in (None, "")}
        summary = str(result.get("summary") or "").strip()
        if summary:
            item["summary"] = _bounded_text(summary, 1200)
        for key in ("behavior", "unverified"):
            values = result.get(key) or []
            kept = [_bounded_text(str(value), 1200) for value in values[:8]
                    if str(value).strip()]
            if kept:
                item[key] = kept
        changes = []
        for change in (result.get("changes") or [])[:20]:
            if not isinstance(change, dict):
                continue
            changes.append({key: change.get(key) for key in ("path", "summary", "before", "after")
                            if change.get(key) not in (None, "")})
        if changes:
            item["changes"] = changes
        compact.append(item)
    return compact


def _shared_request_context(state: dict) -> str:
    """The stable request-level message a review sends before its changing prompt.

    The approved plan context, the request and the investigation's findings form
    a byte-identical prefix with its own cache boundary, reused by every recheck.
    """
    blocks = [
        "## APPROVED PLAN CONTEXT\n" + json.dumps(_approved_plan_context(state), ensure_ascii=True),
    ]
    user_message = str(state.get("follow_up_message") or state.get("user_message") or "").strip()
    if user_message:
        blocks.append("## USER REQUEST\n" + user_message)
    findings = _bounded_text(
        state.get("implementation_memory") if state.get("is_follow_up")
        else state.get("understanding_summary", ""),
        MAX_UNDERSTANDING_CONTEXT_CHARS,
    ).strip()
    if findings:
        blocks.append("## CODEBASE FINDINGS\n" + findings)
    verification_context = verification.contract_context(state.get('verification_contract') or {})
    if verification_context:
        blocks.append(verification_context)
    return "\n\n".join(blocks) + "\n\n"


def _approved_deletions(state: dict) -> set[str]:
    tasks = state.get("execution_tasks") or []
    actions = {path: task.get("action") for task in tasks for path in task.get("files", [])}
    return {path for path, action in actions.items() if action == "DELETE"}


def _review_paths(state: dict, task_paths) -> list[str]:
    # A follow-up names no task paths; its read/check scope is what it changed before.
    return list(dict.fromkeys(task_paths if task_paths is not None else state.get("prior_changed_paths") or []))


def _integration_neighbor_paths(state: dict, paths: list[str], changed_paths: list[str]) -> list[str]:
    """Add unchanged JS callers of changed Python endpoint modules.

    Plan files are a starting point, so an unchanged caller can legitimately be
    absent from them.  Final integration must still run deterministic wiring
    checks on that caller instead of hoping the reviewer happens to search for
    it.  Both the app-root and nested-package spellings are searched because a
    missing inner package segment is itself a common Frappe wiring defect.
    """
    app_name = state.get("target_app_name", "")
    prefixes = set()
    for path in changed_paths:
        normalized = path.replace("\\", "/")
        if not normalized.endswith(".py"):
            continue
        module = normalized[:-3].replace("/", ".")
        prefixes.add(f"{app_name}.{module}")
        if module.startswith(app_name + "."):
            prefixes.add(module)
    if not prefixes:
        return list(dict.fromkeys(paths))
    root = frappe.get_app_path(app_name)
    neighbors = []
    ignored = set(getattr(agent_tools, "IGNORE_DIRS", ()))
    for directory, dirnames, filenames in os.walk(root):
        check_active(reserve=10)
        dirnames[:] = [name for name in dirnames if name not in ignored]
        for filename in filenames:
            if not filename.endswith(".js"):
                continue
            full = os.path.join(directory, filename)
            try:
                with open(full, encoding="utf-8", errors="replace") as source:
                    content = source.read()
            except OSError:
                continue  # an unreadable caller only narrows the extra wiring checks
            if any(prefix in content for prefix in prefixes):
                neighbors.append(os.path.relpath(full, root).replace("\\", "/"))
    return list(dict.fromkeys([*paths, *neighbors]))


def _read_current(state: dict, path: str):
    """A file's current content for baselines and review evidence; None for a redacted path,
    whose content must not reach the reviewer even when the plan names it."""
    app_name = state["target_app_name"]
    full = agent_tools._resolve_path(app_name, path)
    if agent_tools.redaction_pattern(app_name, path, full):
        return None
    return read_snapshot(full)


def _without_redacted(state: dict, baseline: dict) -> dict:
    """The baseline minus redacted paths. Content recorded before a path was redacted (an older
    checkpoint, a glob added since) would otherwise come back as a "Deleted" diff."""
    app_name = state["target_app_name"]
    matcher = agent_tools._redaction_matcher(app_name)
    return {path: content for path, content in baseline.items() if not agent_tools.redaction_pattern(
        app_name, path, agent_tools._resolve_path(app_name, path), matcher)}


def _persist_task_results(state: dict):
    """Persist audit results, not a claim that filesystem writes are replayable."""
    if state.get("request_name"):
        set_request_value(state["request_name"], "execution_results", json.dumps({
            "tasks": state.get("task_results", []),
            "file_moves": state.get("file_moves", []),
            "plan_scope_repairs": state.get("plan_scope_repairs", []),
            "final_review_passed": bool(state.get("review_passed")),
            "review_notes": state.get("review_notes", ""),
            "verification_receipts": state.get("verification_receipts", []),
            "repair_strategy_level": state.get("repair_strategy_level", 0),
            "model_calls": state.get("tool_rounds_used", 0),
            "error": state.get("error", ""),
        }, ensure_ascii=True))
        frappe.db.commit()


def _implementation_updates(state: dict, updates: dict, logs: list) -> dict:
    """Fold an implementation turn's writes into the baseline, change evidence and completion report."""
    before = updates.pop("_write_baseline", {})
    updates.pop("_tool_edited_paths", None)
    file_moves = list(state.get("file_moves") or [])
    for move in updates.pop("_file_moves", []):
        if move not in file_moves:
            file_moves.append(move)
    # The state's baseline already holds the plan's files, so a no-op change is still reviewed.
    baseline = dict(state.get("execution_baseline") or {})
    for path, content in before.items():
        baseline.setdefault(path, content)
    baseline = _without_redacted(state, baseline)
    edits, _ = change_evidence(baseline, lambda p: _read_current(state, p))
    steps = updates.get("intermediate_steps") or []
    output = steps[-1].get("output", "") if steps else ""
    completion = completion_report(output)
    updates.update({
        "execution_baseline": baseline,
        "file_moves": file_moves,
        "edits_made": edits,
        "task_completion": completion,
        "task_summary": completion.get("summary", _bounded_text(output, 4000)),
        "stage_log": _log_stage({**state, "stage_log": logs}, "Implementing", "completed" if not updates.get("error") else "failed",
                                f"{len(edits)} changed file(s); review pending"),
    })
    return updates


def _review_fingerprint(changes: list) -> str:
    """The changed-file state a failed review saw; equal to the last one means no progress."""
    import hashlib
    entries = sorted(f"{c.get('path')}\x00{c.get('after')}" for c in changes if isinstance(c, dict))
    return hashlib.sha1("\x01".join(entries).encode("utf-8")).hexdigest() if entries else "none"


def _review_outcome_fingerprint(changes: list, notes: str) -> str:
    """Identify a failed state by both bytes and the finding to act on.

    Returning to old bytes is only a repair cycle when the same evidence is
    asking for the same correction. A new finding on those bytes is new
    information and must get one implementation attempt.
    """
    import hashlib
    source = _review_fingerprint(changes)
    finding = " ".join(str(notes or "").split())
    return hashlib.sha1(f"{source}\x00{finding}".encode("utf-8")).hexdigest()


def _record_review_fingerprint(fingerprint: str, prior: list[str] | None) -> tuple[bool, list[str]]:
    """Return whether this file state was already reviewed and the updated history."""
    seen = list(prior or [])
    cycled = fingerprint in seen
    if not cycled:
        seen.append(fingerprint)
    return cycled, seen


def _health_failure_fingerprint(health: HealthReport) -> str:
    """Identity of deterministic checker failures, independent of file churn."""
    def stable(detail):
        detail = '\n'.join(line for line in detail.splitlines() if not line.startswith(
            ('LIVE_ACCEPTANCE ', 'BROWSER DIAGNOSTICS ', 'KODA_TEST_SUMMARY ')))
        detail = _re.sub(r"Ran (\d+) tests? in [\d.]+s", r"Ran \1 tests", detail)
        return _re.sub(r"(duration_ms:|# duration_ms)\s*[\d.]+", r"\1 <elapsed>", detail)
    failures = sorted(
        (result.name, stable(result.detail), result.owner)
        for result in health.failures
        if result.owner == "implementation"
    )
    if not failures:
        return ""
    import hashlib
    return hashlib.sha1(json.dumps(failures, ensure_ascii=True).encode("utf-8")).hexdigest()


def _prepare_verification_contract(state: dict) -> dict:
    # Explicitly approved test changes may update expectations for changed
    # requirements. Other pre-existing regressions remain fixed during repair.
    return verification.prepare_contract(state["target_app_name"], editable_tests=(
        path for task in state.get("execution_tasks", []) for path in task.get("files", [])
    ))


def review_node(state: dict) -> dict:
    """Check the whole implementation, run its tests, then review it in an independent context."""
    if state.get("error"):
        _persist_task_results(state)
        return {"error": state["error"]}
    if state.get("review_stopped"):
        return {}  # repair stopped (its call budget ran out); the last review's findings stand
    active, criteria, allowed = _execution_context(state)
    logs = _log_stage(state, "Reviewing", "started", "Reviewing the changes")
    baseline = _without_redacted(state, state.get("execution_baseline") or {})
    changes, diff = change_evidence(baseline, lambda p: _read_current(state, p))
    changed_paths = [e["path"] for e in changes]
    paths = _integration_neighbor_paths(
        state, list(dict.fromkeys(_review_paths(state, allowed) + changed_paths)), changed_paths)
    reviewed_content = {p: _read_current(state, p) for p in paths}
    moved = renamed_sources(state.get("file_moves") or [], lambda p: _read_current(state, p))
    deletions = _approved_deletions(state)
    prior_deleted = set(state.get("prior_deleted_paths") or []) if state.get("is_follow_up") else set()
    # A task's files are a starting point, so a path the plan named but the work
    # did not need is not a defect — only a path that was actually written, or
    # that exists, is checked.
    check_paths = [p for p in paths if p not in moved and p not in deletions
                   and reviewed_content.get(p) is not None
                   and not (p in prior_deleted and reviewed_content.get(p) is None)]
    health = run_health_checks(state["target_app_name"], [{"path": p} for p in check_paths])
    health.results.extend(
        CheckResult(f"exists:{p}", False, "A file this task changed is missing")
        for p in changed_paths
        if p not in moved and p not in deletions and p not in prior_deleted
        and not _app_file_exists(state["target_app_name"], p)
    )
    health.results.extend(CheckResult(f"deleted:{p}", reviewed_content.get(p) is None,
                                      "Approved DELETE path must be absent") for p in deletions)
    attempt = state.get("review_attempts", 0) + 1
    updates = {"review_repairable": False, "review_retry_requested": False}

    def halted() -> bool:
        """Repair stopped: the work goes out with its findings as warnings."""
        return bool(updates.get("review_stopped"))

    completion = completion_report(json.dumps(state.get("task_completion") or {}))
    # The implementer's own report, for the reviewer to verify rather than trust.
    claims = [{"task_id": active["id"], **completion}]
    if health.passed:
        contract = state.get("verification_contract")
        if contract is None:  # no implementation turn prepared one
            contract = _prepare_verification_contract(state)
        updates["verification_contract"] = contract
        required = verification.needs_tests(
            [*check_paths, *changed_paths, *(state.get("prior_changed_paths") or [])])
        # The host owns re-testing every coherent repair; a blocked/unfinished
        # model report must not postpone this until the call budget has run out.
        test_health, receipts = verification.run_verification(
            state["target_app_name"], contract, env=_get_bench_env(), required=required,
        )
        # The warning reflects these tests only: cleared here, set again below if they still fail
        # after the repair cap. Earlier failures stay in the receipts of the reviews that saw them.
        updates["tests_unresolved"] = ""
        health.results.extend(test_health.results)
        for receipt in receipts:
            receipt["source_revisions"] = {p: revision(content) for p, content in reviewed_content.items()}
        updates["verification_receipts"] = receipts
        updates['verification_progress'] = repair_budget.observe(state.get('verification_progress'), test_health, receipts)
    if state.get("turn_exhausted") and completion.get("status") == "complete":
        # The turn boundary is not a correctness verdict. A complete report
        # still goes through the mandatory checks and independent final review.
        updates["turn_exhausted"] = False
    # Tests still failing after MAX_TEST_REPAIRS rounds are reported, not gated;
    # suite-integrity checks and environment failures keep gating.
    test_repairs = int(state.get("test_repair_rounds") or 0)
    test_warnings, demoted = "", []
    if test_repairs >= MAX_TEST_REPAIRS:
        demoted = [r for r in health.failures if r.name.startswith("tests:")
                   and not r.name.startswith(TEST_INTEGRITY_CHECKS) and getattr(r, "owner", "") != "environment"]
        if demoted:
            test_warnings = "\n".join(f"- {r.name}: {str(r.detail)[:400]}" for r in demoted)
            health.results = [r for r in health.results if r not in demoted]
            updates["tests_unresolved"] = test_warnings
            _publish_agent_log(state.get("request_name", ""), "tests_reported_as_warnings",
                               tests=[r.name for r in demoted], repairs=test_repairs)
    plan_recovery_attempted = False
    unfinished_at_cap = False
    if health.environment_failures:
        passed, notes = False, (
            "Review could not access the required source/checker environment; code repair is not the owner: "
            + health.summary()
        )
        updates["review_stopped"] = notes
    elif state.get("turn_exhausted") and completion.get("status") != "complete":
        # Unfinished, not failed: the continuation gets what is left and every
        # check result, and a check it has not reached yet is not a stall.
        passed, notes = False, f"{CALL_LIMIT_NOTE} without finishing. Inspect current changes and complete the plan."
        updates["review_repairable"] = True
        unfinished_at_cap = True
        # A blocked report from the forced final call says what is left; hand
        # it to the retry instead of making it rediscover the state.
        remaining = [completion.get("summary", "")] + list(completion.get("unverified") or [])
        remaining = [item for item in remaining if isinstance(item, str) and item.strip()]
        if completion.get("status") == "blocked" and remaining:
            notes += " Reported remaining work: " + " | ".join(remaining)[:1500]
        notes += "\nChecks already executed against the current files:\n" + health.summary()
    elif any(result.name.startswith('tests:') for result in health.failures):
        # An older completion report may claim testing was blocked. Freshly
        # executed failing cases are stronger evidence: repair those cases,
        # rather than sending a now-stale testing blocker back to the planner.
        passed, notes = False, health.summary()
        updates["review_repairable"] = True
        updates["test_repair_rounds"] = test_repairs + 1
    elif completion.get("status") == "blocked":  # an exhausted turn took the branch above
        # A scope blocker must reach the planner even when an uncreated destination
        # also fails the existence check. Repeating the same contract cannot fix it.
        passed = False
        plan_recovery_attempted = True
        amended, note = _amend_plan_for_blocker(state, completion)
        # Keep the repair outcome before the model's often-long blocker report,
        # so dashboard previews and bounded errors cannot hide why recovery failed.
        notes = f"{PLAN_RECOVERY_NOTE} {note}\nImplementation blocked: {completion['summary']}"
        _publish_agent_log(state.get("request_name", ""), "llm_response", round=0,
                           preview=f"{active['id']} plan recovery: {note}")
        if amended:
            updates.update(amended)
        if amended and "execution_tasks" in amended:
            attempt = 0  # A repaired contract gets a fresh, still-bounded review budget.
            updates["review_fingerprint"] = ""
            updates["review_fingerprints_seen"] = []
            updates["review_failure_fingerprints_seen"] = []
        if not health.passed:
            notes += " Current checks: " + health.summary()
    elif not health.passed:
        passed, notes = False, health.summary()
        updates["review_repairable"] = True
    elif (state.get("review_attempts", 0) > 0
          and not state.get("review_retry_requested")
          and completion.get("status") != "complete"):
        passed, notes = False, (f"{INVALID_REPORT_NOTE} complete JSON report. Finish the task and report its "
                                "behavior and verification.")
        updates["review_repairable"] = True
    else:
        # _run_agent_turn sends the shared request context as its own message.
        prompt = get_review_prompt([{"path": p} for p in paths], request_name=state.get("request_name"))
        prompt += "\n\n## REVIEW CONTRACT (authoritative)\n" + json.dumps({
            "phase": "final review of the whole plan",
            "criteria": [{"criterion": i, "requirement": c} for i, c in enumerate(criteria, 1)],
            "implementation_claims": _compact_execution_results(claims),
        }, ensure_ascii=True)
        # Changed paths go in only as their diff; hashes bind every reviewed path,
        # and omitted source stays reachable through the read tools.
        prompt += "\n\n## CURRENT CHANGE EVIDENCE\n" + diff
        unchanged_paths = [path for path in paths if path not in set(changed_paths)]
        if unchanged_paths:
            # No-op/follow-up review still needs actual source, while changed
            # paths must not be duplicated beside their diff.
            prompt += "\n\n## UNCHANGED SOURCE EVIDENCE\n" + source_context(
                unchanged_paths, reviewed_content.get, baseline, limit=8000
            )
        prompt += "\n\n## SOURCE REVISION MANIFEST\n" + json.dumps({
            path: revision(reviewed_content.get(path)) for path in paths
        }, ensure_ascii=True)
        prompt += "\n\n## STATIC CHECKS\n" + health.summary()
        prompt += "\n\n## EXECUTED BEHAVIORAL CHECKS\n" + json.dumps(
            _model_receipts(updates.get("verification_receipts", [])), ensure_ascii=True)
        prompt += (
            "\nAssess every criterion against current source. Behavior is proven by running it: use "
            "call_method to run the changed endpoints and helpers against the live site (database writes are "
            "rolled back), and read the executed test receipts and their source. Python tests run against the live "
            "site; a test that mocks the behavior under test proves nothing. Implementation claims are not "
            "proof. Do not invent requirements outside the approved scope. For a copy or adaptation, the "
            "reference is part of the specification: before marking a criterion unmet, check what the reference "
            "does. Behavior the plan does not name as a change or a fix is correct when the change does what the "
            "reference does, even where a criterion's wording seems stricter; record that conflict as P2, never "
            "P0/P1. Missing or truncated context is a reason to use your tools, not a defect.\n"
            "Rank each issue: P0 the request does not work (crash, wrong data, broken page); P1 a criterion "
            "is not met in real use or a regression; P2 an edge case, missing test for working behavior, "
            "or a better design; P3 style. Only P0/P1 send the task back; P2/P3 are recorded and pass. "
            "Client/UI behavior is verified by reading its source and wiring; never require a Node test for "
            "it. Name a test file in an issue only when that test itself is wrong. "
            'Return ONLY JSON: {"review_passed":true,"issues":[{"criterion":1,"severity":"P1",'
            '"issue":"what fails, the concrete input and the observed vs expected result"}],"evidence":'
            '[{"criterion":1,"status":"satisfied","evidence":"path:symbol and what you ran or read"}]}. '
            'Include exactly one evidence entry per numbered criterion. "unmet" needs a P0/P1 issue; mark a '
            'criterion with only P2/P3 issues "satisfied". Use "unverified" only for evidence your tools can '
            "still fetch; get it first. review_passed is false exactly when a P0/P1 issue or an unverified "
            "criterion remains."
        )
        histories = state.get("review_history") or {}
        prior_attempts = int(state.get("review_attempts", 0) or 0)
        reuse_history = bool(prior_attempts and prior_attempts % REVIEW_HISTORY_RECHECKS)
        history = histories.get(active["id"]) if reuse_history else None
        history = history if isinstance(history, dict) else {}
        opening_prompt = history.get("task_prompt")
        previous_snapshot = history.get("reviewed_content")
        if isinstance(opening_prompt, str) and isinstance(previous_snapshot, dict):
            # Continue the same reviewer conversation after implementation has
            # repaired its finding. The previous source/tool reads remain a
            # cacheable prefix; only the actual repair delta and current receipts
            # are appended. Explicit supersession prevents stale source from
            # being treated as current.
            delta_baseline = dict(previous_snapshot)
            for path in paths:
                delta_baseline.setdefault(path, None)
            _, repair_diff = change_evidence(
                delta_baseline, reviewed_content.get, limit=12000
            )
            directive = (
                "## REVIEW RECHECK AFTER IMPLEMENTATION REPAIR\n"
                "Continue the same complete review. The current repair diff below supersedes any older "
                "source lines or tool results for those paths. Preserve previously satisfied criteria, "
                "verify the concrete prior finding is resolved, and inspect the changed regions for regressions. "
                "Converge: a new finding in code this repair did not change is P2 unless it makes the request "
                "fail outright (P0); do not deepen a criterion the previous pass accepted. "
                "Use read-only tools only when this delta is truncated or a connected dependency is missing.\n\n"
                "### PRIOR FINDING\n" + str(state.get("review_notes") or "")[:4000]
                + "\n\n### CURRENT REPAIR DIFF\n" + repair_diff
                + "\n\n### CURRENT SOURCE REVISIONS\n" + json.dumps({
                    path: revision(reviewed_content.get(path)) for path in paths
                }, ensure_ascii=True)
                + "\n\n### CURRENT IMPLEMENTATION CLAIM\n" + json.dumps(
                    _compact_execution_results(claims), ensure_ascii=True)
                + "\n\n### CURRENT STATIC CHECKS\n" + health.summary()
                + "\n\n### CURRENT EXECUTED BEHAVIORAL CHECKS\n" + json.dumps(
                    _model_receipts(updates.get("verification_receipts", [])), ensure_ascii=True)
            )
            # The repair changed the source: earlier reads, and calls that failed
            # against the old revision, may run again.
            for key in ("seen_calls", "seen_results", "failed_calls", "failure_causes"):
                history.setdefault(key, {}).clear()
            # Each recheck restates the prior finding and the full current delta.
            drop_followups(history, "review_recheck")
            _queue_followup(
                history,
                str(history.get("last_verdict") or state.get("review_notes") or ""),
                directive,
                kind="review_recheck",
            )
            prompt = opening_prompt
        else:
            history["task_prompt"] = prompt
        history["reviewed_content"] = dict(reviewed_content)
        decision, notes = "invalid", ""
        prior_unmet = set()
        # The recovery pass continues the first pass's retained tool rounds, so
        # its smaller round budget is spent on reads that have not happened yet.
        output = ""
        for recovery_pass in range(2):
            turn_updates = _run_agent_turn(
                {**state, **updates}, "Reviewing", prompt, read_only_tools=True,
                max_rounds=MAX_TOOL_ROUNDS_REVIEW if recovery_pass == 0 else MAX_TOOL_ROUNDS_REVIEW_RECOVERY,
                history=history,
                # The same catalogue as planning and implementation, so the review
                # reuses the cached tools+system prefix; both session tools refuse here.
                session={"plan_sink": None, "explorer": None, "shared_context": True},
            )
            updates.update(turn_updates)
            updates.pop("_write_baseline", None)
            updates.pop("_tool_edited_paths", None)
            updates.pop("_file_moves", None)
            if updates.get("review_stopped"):
                notes = str(state.get("review_notes") or "")  # the last findings go out with the warning
                break
            # A reviewer turn that failed (a provider error, a full context) is retried
            # like missing evidence, and after its bounded retries reported, never fatal.
            turn_error = updates.pop("error", None)
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
            if updates.get("turn_exhausted") and not turn_error and decision in {"pass", "repair"}:
                # The forced no-tools final call still produced a complete,
                # well-formed verdict. Reaching the round cap is not a reason
                # to discard it; only an incomplete or invalid verdict is.
                updates["turn_exhausted"] = False
                _publish_agent_log(state.get("request_name", ""), "review_verdict_after_cap",
                                   decision=decision, recovery=recovery_pass)
            if updates.get("turn_exhausted") and recovery_pass == 0 and not turn_error:
                # A per-pass cap is why the bounded evidence recovery exists.
                updates["turn_exhausted"] = False
            if turn_error or updates.get("turn_exhausted"):
                notes = turn_error or "Reviewer exhausted its call limit; verification is incomplete."
                # Exhausting this read-only pass is recoverable if the overall
                # execution budget still has room for a fresh evidence pass.
                updates["turn_exhausted"] = False
                decision = "needs_evidence"
                break
            if decision not in {"invalid", "needs_evidence"}:
                break
            if recovery_pass == 0:
                entries = payload.get("evidence", []) if isinstance(payload, dict) else []
                # A malformed envelope may still contain a valid concrete finding.
                prior_unmet = {e["criterion"] for e in entries if isinstance(e, dict)
                    and type(e.get("criterion")) is int and 1 <= e["criterion"] <= len(criteria)
                    and e.get("status") == "unmet"} if isinstance(entries, list) else set()
                _queue_followup(history, output, (
                    "## REVIEWER RECOVERY\n"
                    + _review_recovery_reason(decision, notes, payload, criteria) + "\n\n"
                    "The tool results from your first pass are retained above; do not re-read them. "
                    "Use read-only tools only for current source you have not fetched yet, preserve all "
                    "concrete findings, then return the complete verdict as JSON only. Read test sources "
                    "and execution receipts for runtime criteria. If changed server logic has no executed "
                    "evidence, run it with call_method, or report the scenario implementation must add and "
                    "run; verify client/UI behavior from source. "
                    'If any earlier unmet criterion becomes satisfied, include resolved_findings '
                    '[{"criterion":1,"explanation":"Current source evidence resolving the earlier finding"}]. '
                    "Every concrete finding must remain or be explicitly resolved."
                ))
        history["last_verdict"] = output
        updates["review_history"] = {active["id"]: history}
        passed = decision == "pass" and not halted()
        if decision == "repair" and not halted():
            updates["review_repairable"] = True
            # A test the reviewer calls wrong must be correctable; every other
            # green test stays frozen.
            contract = dict(updates.get("verification_contract") or state.get("verification_contract") or {})
            contract["frozen_tests"] = dict(contract.get("frozen_tests") or {})
            if verification.release_frozen(contract, notes):
                updates["verification_contract"] = contract
        if decision in {"invalid", "needs_evidence"} and not halted():
            retries = int(state.get("review_format_retries", 0) or 0)
            if retries < MAX_REVIEW_RECHECKS:
                updates.update(review_retry_requested=True, review_format_retries=retries + 1,
                               review_history={})
                attempt = int(state.get("review_attempts", 0) or 0)
                notes = "Reviewer evidence/format recovery needed: " + notes
            else:
                updates["review_stopped"] = "The independent review could not reach a verdict (" + (
                    "missing evidence" if decision == "needs_evidence" else "invalid verdict format"
                ) + ") after recovery and two fresh review attempts: " + notes[:1500]
        elif decision in {"pass", "repair"}:
            updates["review_format_retries"] = 0
    if (not halted()
            and any(_read_current(state, p) != content for p, content in reviewed_content.items())):
        passed, notes = False, "Files changed during review. Verify current source again before accepting the work."
        updates["review_repairable"] = False
        rechecks = int(state.get("review_rechecks", 0) or 0) + 1
        if rechecks <= MAX_REVIEW_RECHECKS:
            # This is neither an implementation defect nor a terminal error:
            # the verdict simply describes an obsolete snapshot.  Re-enter
            # review with current bytes without spending a repair attempt.
            updates["review_retry_requested"] = True
            updates["review_rechecks"] = rechecks
            attempt = int(state.get("review_attempts", 0) or 0)
        else:
            updates["review_stopped"] = (
                f"Source kept changing during review after {rechecks} snapshots; "
                "automatic re-review stopped to avoid approving unstable files."
            )
    elif passed or updates.get("review_repairable"):
        updates["review_rechecks"] = 0
    if test_warnings:
        notes = f"{notes}\nTests still failing after {test_repairs} repair rounds (reported, not blocking):\n{test_warnings}"
    updates.update({"review_passed": passed, "review_notes": notes, "review_attempts": attempt})
    # Repeated evidence stops repair, not an attempt count. An unfinished turn
    # words its remaining work differently each time, so its progress is the
    # files and check results moving, not the notes.
    failure_fingerprint = _health_failure_fingerprint(health)
    fingerprint = ("unfinished\x00" + _review_fingerprint(changes) + "\x00" + failure_fingerprint
                   if unfinished_at_cap else _review_outcome_fingerprint(changes, notes))
    # An amended plan reset the histories above; read them from there.
    history_source = updates if "review_fingerprints_seen" in updates else state
    prior_fingerprints = list(history_source.get("review_fingerprints_seen") or [])
    if not prior_fingerprints and history_source.get("review_fingerprint"):
        prior_fingerprints.append(history_source["review_fingerprint"])
    track_failure_progress = not passed and not halted() and not updates.get("review_retry_requested")
    cycled, seen = (_record_review_fingerprint(fingerprint, prior_fingerprints)
                     if track_failure_progress else (False, prior_fingerprints))
    prior_failure_fingerprints = list(history_source.get("review_failure_fingerprints_seen") or [])
    # An unfinished turn stalls only when its files and checks both stand still (``cycled``).
    repeated_failure = bool(track_failure_progress and failure_fingerprint and not unfinished_at_cap
                            and failure_fingerprint in prior_failure_fingerprints)
    if track_failure_progress and failure_fingerprint and not repeated_failure:
        prior_failure_fingerprints.append(failure_fingerprint)
    stalled_plan_recovery = (plan_recovery_attempted
                             and "execution_tasks" not in updates)
    updates["review_fingerprint"] = fingerprint
    updates["review_fingerprints_seen"] = seen
    updates["review_failure_fingerprints_seen"] = prior_failure_fingerprints
    if repeated_failure and completion.get("status") == "blocked":
        # Blocked and the same check failed again: deliver with the failure as a warning.
        # (An exhausted turn's "blocked" means unfinished; repeated_failure excludes it.)
        updates["review_stopped"] = ("The work is blocked and the same "
                                     f"deterministic check failed again: {completion.get('summary', '')[:1000]}\n"
                                     f"{notes[:2000]}")
    elif (track_failure_progress and attempt >= MAX_REVIEW_ATTEMPTS
            and (repeated_failure or cycled or stalled_plan_recovery)):
        why = ("after the same deterministic check failed again" if repeated_failure else
               "after plan recovery produced no executable contract change" if stalled_plan_recovery else
               "after returning to a changed-file state already reviewed")
        strategy = int(state.get("repair_strategy_level", 0) or 0)
        if strategy < MAX_REPAIR_STRATEGIES:
            updates.update(repair_strategy_level=strategy + 1, review_repairable=True,
                           review_notes=notes + f"\nPrevious repair was ineffective {why}.")
            _publish_agent_log(state.get("request_name", ""), "repair_strategy_changed",
                               task_id=active["id"], strategy=strategy + 1, preview=notes[:1000])
        else:
            updates["review_stopped"] = (f"The work did not pass review after "
                                         f"{attempt} attempts and {strategy} changed repair strategies {why}: "
                                         f"{notes[:2000]}")
    elif (not passed and not halted()
          and updates.get("review_repairable")
          and attempt % REVIEW_COST_PRESSURE_ATTEMPT == 0):
        # Telemetry only: many serial reviews never stop the run.
        _publish_agent_log(
            state.get("request_name", ""),
            "review_cost_pressure",
            task_id=active["id"],
            attempts=attempt,
            action="continue repair with compact evidence",
        )
    results = list(state.get("task_results") or [])
    result = {
        "task_id": active["id"], "status": "passed" if passed else "failed",
        "attempts": attempt, "summary": state.get("task_summary", ""),
        "behavior": completion.get("behavior", []),
        "verification": completion.get("verification", []),
        "unverified": list(completion.get("unverified", []))
        + [f"Failing test (after {test_repairs} repairs): {r.name}" for r in demoted],
        "changes": changes, "review": notes,
        "executed_checks": updates.get("verification_receipts", []),
    }
    results = [r for r in results if r["task_id"] != active["id"]] + [result]
    updates["task_results"] = results
    updates["change_summary"] = "\n\n".join(r["summary"] for r in results if r.get("summary"))
    updates["stage_log"] = _log_stage({**state, "stage_log": logs}, "Reviewing", "completed",
                                       "Review passed" if passed else "Review failed")
    _persist_task_results({**state, **updates})
    return updates


def _get_bench_env() -> dict:
    """Use one Node environment for syntax, behavioral tests and bench commands."""
    return agent_tools.command_environment()


# Conditional edge and checkpointing, for the session's execution graph

def should_retry_implement(state: dict) -> str:
    if state.get("error") or state.get("review_stopped"):
        return "done"
    if state.get("review_retry_requested"):
        return "review"
    if not state.get("review_passed"):
        return "implement"
    return "done"


def _checkpoint_next_node(name: str, merged: dict, updates: dict) -> str:
    """Persist the node that can make progress, not merely the node that failed."""
    if updates.get("error"):
        if name == "review" and merged.get("review_repairable"):
            return "implement"
        return name
    if name == "review":
        return should_retry_implement(merged)
    return {"prepare": "implement", "implement": "review"}[name]


def _checkpointed_node(name, fn):
    def run(state):
        check_active(reserve=5)
        # A node reached with an earlier node's error only passes it on: the checkpoint keeps the
        # node that failed, or a restart would start here (a failed implement resumed at review).
        active = None if state.get("error") else checkpoint.journal()
        if active:
            active.begin(state, name)
        updates = fn(state)
        if active:
            merged = {**state, **updates}
            next_node = _checkpoint_next_node(name, merged, updates)
            active.begin(merged, next_node)
        return updates
    return run
