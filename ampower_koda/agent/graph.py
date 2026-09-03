# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# LangGraph workflows: Planning (Understand -> Plan) and Execution (Implement -> Review).
# A local syntax check gates the implement pass; review validates clean code/Frappe
# standards before bench and deploy run in executor.py.

import json
import os
import re as _re
from datetime import datetime

import frappe
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.state import AgentState
from ampower_koda.agent import koda_core
from ampower_koda.agent import tools as agent_tools
from ampower_koda.agent.plan_contract import (
    MAX_PLAN_TASKS,
    PLAN_JSON_SCHEMA,
    PlanValidationError,
    plan_to_markdown,
    validate_plan,
)
from ampower_koda.agent.checks import run_health_checks, needs_llm_review
from ampower_koda.agent.prompts import (
    get_system_prompt,
    get_understand_system_prompt,
    get_understand_prompt,
    get_plan_prompt,
    get_implement_prompt,
    get_follow_up_implement_prompt,
    get_review_prompt,
)


MAX_TOOL_ROUNDS_EXECUTION = 18
MAX_TOOL_ROUNDS_REVIEW = 4        # local checks already ran; keep model review surgical
MAX_TOOL_ROUNDS_SYNTAX_FIX = 6    # focused "fix the syntax errors" pass
MAX_SYNTAX_FIX_ATTEMPTS = 2       # how many times to re-invoke to fix remaining syntax errors
MAX_REVIEW_ATTEMPTS = 2           # implement + review retries before failing
WRITE_TOOLS = {"replace_lines", "insert_lines", "edit_file", "write_file"}
REPLAYABLE_TOOLS = {
    "find_files", "list_directory", "read_file", "search_code",
    "read_doctype_schema", "get_file_outline", "validate_code",
}

# Per-phase output stored in conversation_log. High so full phase text is retained
# (phase outputs are LLM summaries and are naturally well under this in practice).
MAX_PHASE_OUTPUT_CHARS = 60000

# History trimming: keep recent tool rounds verbatim, compact older ones to text.
# A "round" is one assistant tool-call message plus all of its tool results.
KEEP_TOOL_ROUNDS = 4          # recent rounds retained in full
MIN_KEEP_ROUNDS = 1           # latest call/result pair must survive into the next request
TRIM_CHAR_BUDGET = 48000      # includes tool-call arguments, not only result text
COMPACT_RESULT_PREVIEW = 140  # chars of each tool result kept in the compact summary
MAX_COMPACTED_HISTORY_CHARS = 12000
MAX_TASK_PROMPT_CHARS = 60000
MAX_TOOL_RESULT_CHARS = 8000
MAX_UNDERSTANDING_CONTEXT_CHARS = 8000
MODEL_ROUND_OUTPUT_TOKENS = 4096
MODEL_FINAL_OUTPUT_TOKENS = 2048
# Sized for the largest valid plan plus its overview and scope.
PLAN_OUTPUT_TOKENS_PER_TASK = 450
MODEL_PLAN_OUTPUT_TOKENS = 1000 + MAX_PLAN_TASKS * PLAN_OUTPUT_TOKENS_PER_TASK


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


def _parse_review_verdict(text: str) -> tuple[bool, str]:
    """Parse the review verdict as structured JSON, with a soft-pass fallback.

    Preferred format: {"review_passed": true/false, "issues": ["...", ...]}

    If the model didn't produce valid JSON, this is treated as a formatting
    slip, not a real failure — by the time review_node calls this, the
    mechanical health checks have already passed, so an unparseable verdict
    is far more likely to be "forgot the format" than "found a real problem".
    A genuine failure must be an *explicit* review_passed: false.
    """
    notes = (text or "").strip()
    payload = _extract_review_json(notes)

    if payload is not None and isinstance(payload.get("review_passed"), bool):
        passed = payload["review_passed"]
        issues = payload.get("issues") or []
        if passed:
            return True, notes
        if isinstance(issues, list) and issues:
            formatted = "\n".join(f"- {issue}" for issue in issues if str(issue).strip())
            return False, formatted or notes
        return False, notes or "Review reported issues without details."

    # No parseable, well-formed verdict. Mechanical checks already passed
    # before this was ever called — don't fail a real run over a formatting
    # slip. Soft pass, but say plainly that the verdict was unparseable so
    # it's visible in review_notes rather than silently swallowed.
    return True, (
        "Verdict could not be parsed as structured JSON; treated as a soft "
        f"pass since mechanical checks already passed. Raw response: {notes[:300]}"
    )


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
    

def _syntax_check_edits(app_name: str, edits: list[dict]) -> tuple[bool, str]:
    """Validate edited .py/.js locally so broken syntax never reaches bench."""
    paths = [e.get("path", "") for e in (edits or []) if e.get("path")]
    code_paths = [p for p in paths if p.endswith((".py", ".js"))]
    if not code_paths:
        return True, ""

    failures = []
    summary_lines = []
    for path in code_paths:
        result = agent_tools.validate_code(app_name, path)
        # A path that doesn't resolve to a real file is not a syntax error — it's
        # usually a phantom path parsed from the model's summary text. Skip it.
        if "Not a file" in result:
            continue
        first_line = (result or "").split("\n", 1)[0][:240]
        summary_lines.append(f"- {path}: {first_line}")
        # Only genuine syntax problems should block the run.
        if "SYNTAX_ERROR" in result:
            failures.append(result)

    summary = "\n".join(summary_lines)
    if failures:
        return False, "\n\n".join(failures)
    return True, summary


# ---------------------------------------------------------------------------
# LLM factory — supports OpenAI, OpenRouter, Gemini and Claude
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _openai_uses_responses_api(model: str) -> bool:
    """Codex and newer pro models require OpenAI's /v1/responses endpoint."""
    name = (model or "").lower()
    return "codex" in name or "gpt-5.2-pro" in name or name.startswith("gpt-5.4")


def _get_llm(provider: str = "OpenAI", model: str = "gpt-4o-mini"):
    """Build the chat model for the given provider and model."""
    provider = (provider or "OpenAI").strip()
    model = (model or "gpt-4o-mini").strip()
    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    if provider == "Claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    if provider == "OpenRouter":
        # OpenRouter speaks the OpenAI wire format, so ChatOpenAI drives it — only
        # the base URL and key differ. `usage.include` asks OpenRouter to return the
        # real billed cost of each call in the usage block; without it there is no
        # cost field at all. Never enable the Responses API here: OpenRouter only
        # implements /v1/chat/completions.
        return ChatOpenAI(
            model=model,
            temperature=0,
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ.get("OPENROUTER_API_KEY") or "",
            extra_body={"usage": {"include": True}},
        )
    if provider not in ("OpenAI", "Gemini", "Claude", "OpenRouter"):
        log_agent_error(
            "Agent LLM Warning",
            f"Unknown AI provider '{provider}', defaulting to OpenAI",
        )
    openai_kwargs = {"model": model, "temperature": 0}
    if _openai_uses_responses_api(model):
        openai_kwargs["use_responses_api"] = True
    return ChatOpenAI(**openai_kwargs)


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
    tool rounds). OpenAI caches stable prefixes automatically, so a plain system
    message is enough there. Gemini/others fall back to a plain system message.
    Caching only reduces cost; it never changes model output.
    """
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


def _invoke_limited(model, messages: list, max_tokens: int):
    """Apply an output ceiling where the provider supports a per-call limit."""
    try:
        return model.invoke(messages, max_tokens=max_tokens)
    except TypeError:
        return model.invoke(messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            frappe.db.set_value(DOCTYPE_NAME, request_name, {
                "status": stage if status == "started" else state.get("current_stage", stage),
                "stage_log": stage_text[:50000],
            })
            if status == "started":
                frappe.db.set_value(DOCTYPE_NAME, request_name, "status", stage)
            frappe.db.commit()

            user = frappe.db.get_value(DOCTYPE_NAME, request_name, "owner") or "Administrator"
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


def _publish_agent_log(request_name: str, log_type: str, **kwargs):
    """Publish a detailed agent_log realtime event."""
    if not request_name:
        return
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

def _persist_token_usage(request_name: str, total_tokens: int):
    """Write the running token total to the request so the form shows live usage."""
    if not request_name:
        return
    try:
        frappe.db.set_value(DOCTYPE_NAME, request_name, "tokens_used", int(total_tokens or 0))
        frappe.db.commit()
    except Exception:
        log_agent_error(
            "Agent Graph: persist token usage",
            f"request={request_name}\n{frappe.get_traceback()}",
        )


def _make_tools(app_name: str, read_only: bool = False):
    """Build LangChain tools bound to a specific app_name."""

    @tool
    def find_files(pattern: str = "", max_depth: int = 6) -> str:
        """Recursively list the app directory tree using a glob-style pattern.
        Returns an indented tree view. max_depth defaults to 6.
        Call this FIRST to map the codebase structure before reading individual files."""
        return agent_tools.find_files(app_name, pattern, max_depth)

    @tool
    def list_directory(path: str) -> str:
        """List files and directories at path (relative to app root). 
        Directories are prefixed with [DIR] in the output."""
        return agent_tools.list_directory(app_name, path)

    @tool
    def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
        """Read a file (path relative to app root) with line numbers.
        - If start_line and end_line are both > 0, reads only that range (1-indexed, inclusive).
        - Otherwise reads the full file.
        ALWAYS use this to see exact line numbers before editing."""
        return agent_tools.read_file(app_name, path, start_line, end_line)

    @tool
    def search_code(pattern: str, path: str = "") -> str:
        """Search for a regex pattern in the codebase. 
        Returns matches with 3 lines of surrounding context. path is optional directory filter."""
        return agent_tools.search_code(app_name, pattern, path)

    @tool
    def read_doctype_schema(doctype_name: str) -> str:
        """Read DocType JSON schema file. doctype_name e.g. 'Sales Order'."""
        return agent_tools.read_doctype_schema(app_name, doctype_name)

    @tool
    def get_file_outline(path: str) -> str:
        """Get lightweight outline of a Python/JS/TS file — class/function signatures with line numbers.
        Much cheaper than reading the whole file. Use to understand structure before reading specific sections."""
        return agent_tools.get_file_outline(app_name, path)

    @tool
    def validate_code(path: str) -> str:
        """Check for syntax errors in a .py or .js file. Path relative to app root.
        ALWAYS call this after editing to verify that no syntax errors were introduced."""
        return agent_tools.validate_code(app_name, path)

    out = [find_files, list_directory, read_file, search_code, read_doctype_schema, get_file_outline, validate_code]
    if not read_only:
        @tool
        def write_file(path: str, content: str) -> str:
            """Write or overwrite a file at the given relative path. Parent directories are created if needed."""
            return agent_tools.write_file(app_name, path, content)

        @tool
        def edit_file(path: str, old_string: str, new_string: str) -> str:
            """Replace FIRST occurrence of old_string with new_string in file.
            Requires EXACT match including whitespace. For large files, prefer replace_lines."""
            return agent_tools.edit_file(app_name, path, old_string, new_string)

        @tool
        def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
            """Replace lines start_line through end_line (1-indexed, inclusive) with new_content.
            The most reliable tool for editing. Use read_file first to identify the exact line range."""
            return agent_tools.replace_lines(app_name, path, start_line, end_line, new_content)

        @tool
        def insert_lines(path: str, after_line: int, new_content: str) -> str:
            """Insert new_content AFTER the specified 1-indexed line number.
            Use after_line=0 to insert at the beginning of the file."""
            return agent_tools.insert_lines(app_name, path, after_line, new_content)

        out.extend([write_file, edit_file, replace_lines, insert_lines])
    return out


# ---------------------------------------------------------------------------
# Tool-calling loop with detailed realtime logging
# ---------------------------------------------------------------------------

def _compact_round_summary(round_entry: dict) -> list[str]:
    """One short line per tool call in a round, for the compacted-history block."""
    return round_entry.get("summary", [])


def _run_tool_calling_loop(llm, tools, system_prompt: str, task_prompt: str,
                           request_name: str = "", max_rounds: int = 20,
                           state: dict = None, provider: str = "OpenAI",
                           require_writes: bool = False) -> tuple[str, list[str], int]:
    """
    Run a tool-calling loop, publishing every tool call and LLM response via realtime.
    Returns (final_text, list_of_edited_file_paths, total_tokens_used).

    Context control (token savings without quality loss):
      - The stable system prompt is a separate, cache-friendly message.
      - Older tool rounds are compacted to a short text summary while the most
        recent rounds are kept verbatim, so the model keeps working context but
        we stop re-sending large, already-consumed tool outputs every round.

    A "round" is one assistant tool-call message plus all of its tool results;
    rounds are trimmed atomically so no tool_call_id is ever left dangling.

    When require_writes is set (implementation phase), the loop nudges the model
    to stop exploring and actually edit files if it reads for too many rounds
    without writing, or tries to finish without having made any edit. This
    prevents the "read forever, never write" failure that yields no file changes.

    If max_rounds is reached without the LLM stopping, one final llm.invoke() is
    fired WITHOUT tools bound to force a plain-text conclusion.
    """
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    model_id = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "")
    original_task_chars = len(task_prompt or "")
    task_prompt = _bounded_text(task_prompt, MAX_TASK_PROMPT_CHARS)
    if len(task_prompt) < original_task_chars:
        _publish_agent_log(
            request_name, "prompt_trim",
            original_chars=original_task_chars,
            kept_chars=len(task_prompt),
        )
    system_msg = _build_system_message(provider, system_prompt, model_id)
    task_msg = _build_task_message(provider, task_prompt, model_id)

    rounds: list[dict] = []      # each: {"number", "ai", "tools", "summary"}
    compacted: list[str] = []    # summary lines for dropped (older) rounds
    injected: list = []          # directive nudges appended after the latest round
    edited_paths: list[str] = []
    seen_calls: dict[str, int] = {}
    seen_results: dict[str, int] = {}
    total_tokens = (state or {}).get("tokens_used", 0)  # carry forward from prior phases
    usage_missing_logged = False  # warn once per loop, not once per round

    wrote_anything = False
    read_only_streak = 0
    write_nudges = 0
    MAX_WRITE_NUDGES = 3
    READ_STREAK_LIMIT = 3
    NUDGE_TEXT = (
        "STOP exploring. You have read enough — the files you need are already "
        "identified by the approved plan and the reads above. "
        "Do NOT call read_file, search_code, list_directory, get_file_outline or "
        "read_doctype_schema again unless an edit actually fails. Make the planned "
        "changes NOW: call replace_lines, insert_lines, edit_file or write_file to "
        "apply real edits, then validate_code on each changed .py/.js file. Your "
        "next message MUST include an edit tool call."
    )

    def inject_nudge(text: str) -> None:
        # The latest directive supersedes an earlier identical one. Accumulating
        # three copies makes every later request larger without adding a rule.
        injected[:] = [HumanMessage(content=text)]

    def build_messages():
        msgs = [system_msg, task_msg]
        if compacted:
            msgs.append(HumanMessage(content=(
                "## Earlier tool activity (older rounds, summarized to save context)\n"
                "These tools already ran; re-read a file only if you need details not captured here.\n"
                + "\n".join(compacted)
            )))
        for r in rounds:
            msgs.append(r["ai"])
            msgs.extend(r["tools"])
        msgs.extend(injected)
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

    def maybe_trim():
        trimmed = False
        while len(rounds) > KEEP_TOOL_ROUNDS:
            compacted.extend(_compact_round_summary(rounds.pop(0)))
            trimmed = True
        while len(rounds) > MIN_KEEP_ROUNDS and estimate_chars() > TRIM_CHAR_BUDGET:
            compacted.extend(_compact_round_summary(rounds.pop(0)))
            trimmed = True
        if trimmed:
            cap_compacted()
            _publish_agent_log(request_name, "history_trim",
                kept_rounds=len(rounds),
                compacted_lines=len(compacted),
                retained_chars=estimate_chars(),
            )

    def account_tokens(response, round_label, context_chars: int = 0):
        nonlocal total_tokens, usage_missing_logged
        usage = getattr(response, "usage_metadata", None)
        if usage:
            total_tokens += int(usage.get("total_tokens") or 0)
            details = usage.get("input_token_details") or {}
            cache_read = int(details.get("cache_read") or 0)
            cache_write = int(details.get("cache_creation") or 0)
            uncached_input = max(
                0, int(usage.get("input_tokens") or 0) - cache_read - cache_write
            )
            _publish_agent_log(request_name, "token_usage",
                round=round_label,
                tokens_this_round=usage.get("total_tokens", 0),
                tokens_total=total_tokens,
                input_tokens=uncached_input,
                output_tokens=usage.get("output_tokens", 0),
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                context_chars=context_chars,
            )
            _persist_token_usage(request_name, total_tokens)
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

    for round_num in range(max_rounds):
        maybe_trim()
        messages = build_messages()
        context_chars = (
            len(system_prompt) + len(task_prompt) + estimate_chars()
            + sum(len(line) + 1 for line in compacted)
        )
        response = _invoke_limited(llm_with_tools, messages, MODEL_ROUND_OUTPUT_TOKENS)
        account_tokens(response, round_num + 1, context_chars)

        response_text = _llm_response_text(response)
        if response_text:
            _publish_agent_log(request_name, "llm_response",
                preview=response_text[:4000],
                round=round_num + 1,
            )

        if not getattr(response, "tool_calls", None):
            # Model wants to finish. In the implement phase, don't let it stop
            # without ever editing a file — push it to actually write.
            if require_writes and not wrote_anything and write_nudges < MAX_WRITE_NUDGES:
                write_nudges += 1
                inject_nudge(NUDGE_TEXT)
                _publish_agent_log(request_name, "write_nudge",
                    round=round_num + 1, attempt=write_nudges, trigger="no_tool_calls")
                continue
            return response_text, edited_paths, total_tokens

        round_entry = {"number": round_num + 1, "ai": response, "tools": [], "summary": []}
        round_had_write = False
        for tc in response.tool_calls:
            _publish_agent_log(request_name, "tool_call",
                tool_name=tc["name"],
                tool_args={k: (str(v)[:200] if len(str(v)) > 200 else v) for k, v in tc.get("args", {}).items()},
                round=round_num + 1,
            )

            name = tc["name"]
            arguments = tc.get("args", {})
            retained_numbers = {entry["number"] for entry in rounds}
            call_key = _tool_call_key(name, arguments)
            prior_round = seen_calls.get(call_key) if name in REPLAYABLE_TOOLS else None
            duplicate_call = prior_round is not None and prior_round in retained_numbers

            fn = tool_map.get(name)
            if duplicate_call:
                result = (
                    f"[{name} with these exact arguments already returned in round "
                    f"{prior_round}; use that result. Nothing has changed.]"
                )
                _publish_agent_log(
                    request_name, "duplicate_tool_call",
                    tool_name=name, original_round=prior_round, round=round_num + 1,
                )
            elif fn:
                try:
                    result = str(fn.invoke(arguments))
                    if name in REPLAYABLE_TOOLS and len(result) >= 400:
                        result_round = seen_results.get(result)
                        if result_round is not None and result_round in retained_numbers:
                            result = (
                                f"[{name} returned bytes identical to round {result_round}; "
                                "use the earlier result and move on.]"
                            )
                        else:
                            seen_results[result] = round_num + 1
                    if len(result) > MAX_TOOL_RESULT_CHARS:
                        result = (
                            result[:MAX_TOOL_RESULT_CHARS]
                            + f"\n... (truncated at {MAX_TOOL_RESULT_CHARS} characters; "
                              "request a narrower path or line range for anything missing)"
                        )
                except Exception as e:
                    log_agent_error(
                        f"Agent Graph: tool {tc['name']}",
                        f"request={request_name}\n{e}\n{frappe.get_traceback()}",
                    )
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {name}"

            if name in REPLAYABLE_TOOLS and not duplicate_call:
                seen_calls[call_key] = round_num + 1

            if name in WRITE_TOOLS:
                if not result.startswith("Tool error:") and not result.startswith("Error:"):
                    round_had_write = True
                    wrote_anything = True
                    # Every cached read/search is stale after a mutation.
                    seen_calls.clear()
                    seen_results.clear()
                    path_arg = arguments.get("path", "")
                    if path_arg and path_arg not in edited_paths:
                        edited_paths.append(path_arg)

            _publish_agent_log(request_name, "tool_result",
                tool_name=name,
                result_preview=result[:500],
                round=round_num + 1,
            )
            round_entry["tools"].append(ToolMessage(content=result, tool_call_id=tc["id"]))

            arg_preview = ", ".join(
                f"{k}={str(v)[:60]}" for k, v in list(arguments.items())[:3]
            )
            round_entry["summary"].append(
                f"[r{round_num + 1}] {name}({arg_preview}) -> {result[:COMPACT_RESULT_PREVIEW]}"
            )

        rounds.append(round_entry)

        # Proactively break analysis-paralysis: if the model keeps only reading
        # in the implement phase, tell it to start editing.
        if round_had_write:
            read_only_streak = 0
        else:
            read_only_streak += 1
        if (require_writes and not wrote_anything
                and read_only_streak >= READ_STREAK_LIMIT
                and write_nudges < MAX_WRITE_NUDGES):
            write_nudges += 1
            read_only_streak = 0
            inject_nudge(NUDGE_TEXT)
            _publish_agent_log(request_name, "write_nudge",
                round=round_num + 1, attempt=write_nudges, trigger="read_streak")

    maybe_trim()
    final_context_chars = (
        len(system_prompt) + len(task_prompt) + estimate_chars()
        + sum(len(line) + 1 for line in compacted)
    )
    final = _invoke_limited(llm, build_messages(), MODEL_FINAL_OUTPUT_TOKENS)
    account_tokens(final, max_rounds + 1, final_context_chars)
    return _llm_response_text(final), edited_paths, total_tokens


# ---------------------------------------------------------------------------
# Agent turn helper
# ---------------------------------------------------------------------------

def _run_agent_turn(state: dict, phase: str, prompt: str, read_only_tools: bool, max_rounds: int = 20) -> dict:
    """Run one agent turn for the given phase. Returns state updates."""
    try:
        app_name = state.get("target_app_name", "")
        provider = state.get("ai_provider", "OpenAI")
        model = state.get("ai_model", "gpt-4o-mini")
        request_name = state.get("request_name", "")

        tools = _make_tools(app_name, read_only=read_only_tools)
        llm = _get_llm(provider=provider, model=model)
        system_prompt = get_system_prompt(app_name or "target_app", request_name=request_name)
        content, tool_edited_paths, total_tokens = _run_tool_calling_loop(
            llm, tools, system_prompt, prompt,
            request_name=request_name,
            max_rounds=max_rounds,
            state=state,
            provider=provider,
            require_writes=not read_only_tools,
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

        plan_object = validate_plan(
            plan_object,
            path_exists=lambda path: _app_file_exists(app_name, path),
        )

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

        # Derive the editable plan consumed by the current executor.
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

def implement_node(state: dict) -> dict:
    """Apply the approved plan by editing and creating files with write tools.

    Runs an implementation pass, then a local syntax check on the edited
    .py/.js files so broken code never reaches bench. A review stage can
    send corrections back into a follow-up implement pass.
    """
    if state.get("error"):
        return {"error": state["error"]}
    is_follow_up = bool(state.get("is_follow_up"))
    attempt = (state.get("review_attempts") or 0) + 1
    logs = _log_stage(
        state, "Implementing", "started",
        "Applying follow-up patch" if is_follow_up else f"Applying code changes (attempt {attempt})"
    )

    app_name = state.get("target_app_name", "")
    plan_text = _message_content_to_str(state.get("plan", ""))
    understanding_text = _message_content_to_str(state.get("understanding_summary", ""))

    if is_follow_up:
        follow_up_message = _message_content_to_str(state.get("follow_up_message", ""))
        prior_paths = list(state.get("prior_changed_paths") or [])
        implementation_memory = _message_content_to_str(state.get("implementation_memory", ""))
        context_paths = list(dict.fromkeys(prior_paths + _extract_file_paths(follow_up_message)))
        file_contents = _file_manifest(context_paths)
        prior_files = "\n".join(f"- {p}" for p in context_paths) if context_paths else "- (none recorded)"
        logs = _log_stage(
            {**state, "stage_log": logs}, "Implementing", "progress",
            f"Follow-up context: {len(context_paths)} target files identified"
        )
        base_prompt = get_follow_up_implement_prompt(
            follow_up_message,
            plan_text,
            _bounded_text(implementation_memory, MAX_UNDERSTANDING_CONTEXT_CHARS),
            prior_files,
            file_contents,
            request_name=state.get("request_name"),
        )
    else:
        all_text = plan_text + "\n" + understanding_text
        file_paths = _extract_file_paths(all_text)
        file_contents = _file_manifest(file_paths)
        logs = _log_stage(
            {**state, "stage_log": logs}, "Implementing", "progress",
            f"Identified {len(file_paths)} target files: {', '.join(file_paths[:5])}"
        )
        base_prompt = get_implement_prompt(
            plan_text,
            _bounded_text(understanding_text, MAX_UNDERSTANDING_CONTEXT_CHARS),
            state.get("user_message", ""),
            file_contents,
            request_name=state.get("request_name"),
        )
        review_notes = (state.get("review_notes") or "").strip()
        if attempt > 1 and review_notes:
            base_prompt += f"""

## PRIOR REVIEW FINDINGS — FIX THESE NOW
The last test stage failed. Fix every issue listed below before doing anything else.
Focus on the exact files and problems described; do not add new features.

{review_notes}
"""

    updates = _run_agent_turn(state, "Implementing", base_prompt, read_only_tools=False, max_rounds=MAX_TOOL_ROUNDS_EXECUTION)
    if updates.get("error"):
        logs = _log_stage({**state, "stage_log": logs}, "Implementing", "failed", updates["error"][:200])
        updates["stage_log"] = logs
        return updates

    tool_edited = updates.pop("_tool_edited_paths", [])
    steps = updates.get("intermediate_steps") or []
    last_out = _message_content_to_str(steps[-1].get("output", "") if steps else "")
    # Only keep summary-mentioned paths that actually exist on disk — the model
    # often names files it merely considered, which would otherwise be recorded
    # as phantom edits and break validation ("Not a file").
    text_paths = [p for p in _extract_file_paths(last_out) if _app_file_exists(app_name, p)]

    all_paths = list(dict.fromkeys(tool_edited + text_paths))

    edits = list(state.get("edits_made") or [])
    for p in all_paths:
        if not any(e.get("path") == p for e in edits):
            edits.append({"path": p, "summary": "Modified"})
    if not all_paths:
        edits.append({"summary": last_out[:300]})
    updates["edits_made"] = edits
    updates["change_summary"] = _extract_change_summary(last_out)

    logs = _log_stage(
        {**state, "stage_log": logs}, "Implementing", "progress",
        f"Files edited: {', '.join(all_paths[:8]) if all_paths else 'see summary'}"
    )

    # Local syntax gate. If real syntax errors exist, try to auto-fix them by
    # feeding the exact errors back to the model, then re-check — instead of
    # failing the whole run on the first broken edit.
    syntax_ok, syntax_report = _syntax_check_edits(app_name, edits)
    fix_state = {
        **state,
        "tokens_used": updates.get("tokens_used", state.get("tokens_used", 0)),
        "intermediate_steps": updates.get("intermediate_steps") or [],
        "stage_log": logs,
    }
    attempt = 0
    while not syntax_ok and attempt < MAX_SYNTAX_FIX_ATTEMPTS:
        attempt += 1
        logs = _log_stage(
            {**state, "stage_log": logs}, "Implementing", "progress",
            f"Auto-fixing syntax errors (attempt {attempt})"
        )
        fix_state["stage_log"] = logs
        fix_prompt = (
            "One or more files you just edited have SYNTAX ERRORS. Fix ONLY these "
            "errors now. For each affected file: read it to see the current state, "
            "correct the syntax with replace_lines/insert_lines/edit_file, then call "
            "validate_code on it to confirm it passes. Do NOT change unrelated code "
            "and do NOT create any new files.\n\n"
            f"## SYNTAX ERRORS TO FIX\n{syntax_report}"
        )
        fix_updates = _run_agent_turn(
            fix_state, "Implementing", fix_prompt,
            read_only_tools=False, max_rounds=MAX_TOOL_ROUNDS_SYNTAX_FIX,
        )
        # Carry forward token/step accounting whatever the outcome.
        fix_state["tokens_used"] = fix_updates.get("tokens_used", fix_state["tokens_used"])
        fix_state["intermediate_steps"] = fix_updates.get("intermediate_steps") or fix_state["intermediate_steps"]
        updates["tokens_used"] = fix_state["tokens_used"]
        updates["intermediate_steps"] = fix_state["intermediate_steps"]
        if fix_updates.get("error"):
            break
        for p in fix_updates.pop("_tool_edited_paths", []):
            if not any(e.get("path") == p for e in edits):
                edits.append({"path": p, "summary": "Modified (syntax fix)"})
        updates["edits_made"] = edits
        syntax_ok, syntax_report = _syntax_check_edits(app_name, edits)

    if not syntax_ok:
        logs = _log_stage(
            {**state, "stage_log": logs}, "Implementing", "failed",
            f"Syntax check failed: {syntax_report[:150]}"
        )
        updates["error"] = (
            "Implementation produced invalid syntax that could not be auto-fixed:\n"
            f"{syntax_report[:1000]}"
        )
        updates["stage_log"] = logs
        return updates

    logs = _log_stage(
        {**state, "stage_log": logs}, "Implementing", "completed",
        "Changes applied and syntax validated"
        + (f" (auto-fixed on attempt {attempt})" if attempt else "")
    )
    updates["stage_log"] = logs
    return updates


def review_node(state: dict) -> dict:
    """Test the implementation for clean code and Frappe standards."""
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Reviewing", "started", "Testing implementation quality")
    edits = state.get("edits_made", [])
    app_name = state.get("target_app_name", "")
    base_branch = state.get("base_branch", "")
    branch_name = state.get("branch_name", "")
    attempt = (state.get("review_attempts") or 0) + 1

    health = run_health_checks(app_name, edits)

    if not health.passed:
        # A real, mechanical failure — no ambiguity, no need for an LLM
        # opinion. Same retry/error semantics as before.
        report = health.summary()
        updates = {
            "review_passed": False,
            "review_notes": report[:1000],
            "review_attempts": attempt,
        }
        logs = _log_stage(
            {**state, "stage_log": logs}, "Reviewing", "completed",
            f"Testing FAILED (health check): {report[:120]}"
        )
        if attempt >= MAX_REVIEW_ATTEMPTS:
            updates["error"] = f"Health checks failed after {attempt} test attempt(s)."
        updates["stage_log"] = logs
        return updates

    decision = needs_llm_review(app_name, base_branch, branch_name, edits)
    if not decision.needs_llm:
        # Checks passed, and the change is small/low-risk enough that a
        # model's judgment isn't worth the tokens for this run.
        updates = {
            "review_passed": True,
            "review_notes": f"Mechanical checks passed; LLM review skipped ({decision.reason}).",
            "review_attempts": attempt,
        }
        logs = _log_stage(
            {**state, "stage_log": logs}, "Reviewing", "completed",
            f"Testing PASSED (mechanical only — {decision.reason})"
        )
        updates["stage_log"] = logs
        return updates

    # Checks passed, but size/risk says this is worth a model's attention —
    # run the existing LLM review, informed by what's already been verified
    # so it doesn't spend rounds re-checking syntax/JSON/imports/wiring.
    prompt = get_review_prompt(edits, state.get("user_message", ""), request_name=state.get("request_name"))
    short_report = "\n".join(health.summary().splitlines()[:6])
    prompt += (
        "\n\n## LOCAL HEALTH CHECKS (already passed)\n"
        f"{short_report}\n"
        f"## WHY THIS NEEDS REVIEW\n{decision.reason}\n"
        "Do not re-verify syntax, JSON validity, imports, or wiring — focus on logic and quality."
    )
    updates = _run_agent_turn(state, "Reviewing", prompt, read_only_tools=True, max_rounds=MAX_TOOL_ROUNDS_REVIEW)
    if updates.get("error"):
        logs = _log_stage({**state, "stage_log": logs}, "Reviewing", "failed", updates["error"][:200])
        updates["stage_log"] = logs
        return updates

    steps = updates.get("intermediate_steps") or []
    last_out = _message_content_to_str(steps[-1].get("output", "") if steps else "")
    passed, notes = _parse_review_verdict(last_out)
    updates["review_passed"] = passed
    updates["review_notes"] = notes[:1200]
    updates["review_attempts"] = attempt

    result_msg = "Testing PASSED" if passed else f"Testing FAILED: {notes[:100]}"
    logs = _log_stage({**state, "stage_log": logs}, "Reviewing", "completed", result_msg)
    if not passed and attempt >= MAX_REVIEW_ATTEMPTS:
        updates["error"] = f"Review failed after {attempt} attempt(s)."
    updates["stage_log"] = logs
    return updates


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
    if state.get("review_passed"):
        return "done"
    if (state.get("review_attempts") or 0) >= MAX_REVIEW_ATTEMPTS:
        return "done"
    return "implement"


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
    """
    Constructs the state machine for the Implementation Phase.
    Flow: Implement Changes → Review/Testing → (Retry if needed).
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("implement", implement_node)
    workflow.add_node("review", review_node)

    workflow.set_entry_point("implement")
    workflow.add_edge("implement", "review")
    workflow.add_conditional_edges("review", should_retry_implement, {
        "implement": "implement",
        "done": END,
    })

    return workflow.compile()
