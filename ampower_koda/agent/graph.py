# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# LangGraph workflows: Planning (Understand -> Plan) and Execution (Implement).
# A local syntax check gates the implement pass; bench and deploy run in executor.py.

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
from ampower_koda.agent import tools as agent_tools
from ampower_koda.agent.prompts import (
    get_system_prompt,
    get_understand_prompt,
    get_plan_prompt,
    get_implement_prompt,
)


MAX_TOOL_ROUNDS_PLANNING = 40
MAX_TOOL_ROUNDS_EXECUTION = 30
MAX_TOOL_ROUNDS_SYNTAX_FIX = 12   # rounds for a focused "fix the syntax errors" pass
MAX_SYNTAX_FIX_ATTEMPTS = 2       # how many times to re-invoke to fix remaining syntax errors
WRITE_TOOLS = {"replace_lines", "insert_lines", "edit_file", "write_file"}

# Per-phase output stored in conversation_log. High so full phase text is retained
# (phase outputs are LLM summaries and are naturally well under this in practice).
MAX_PHASE_OUTPUT_CHARS = 60000

# History trimming: keep recent tool rounds verbatim, compact older ones to text.
# A "round" is one assistant tool-call message plus all of its tool results.
KEEP_TOOL_ROUNDS = 8          # recent rounds retained in full
MIN_KEEP_ROUNDS = 3           # never trim below this many, even under char pressure
TRIM_CHAR_BUDGET = 100000     # approx history chars before char-based trimming kicks in
COMPACT_RESULT_PREVIEW = 200  # chars of each tool result kept in the compact summary

# Provider-native prompt caching for the stable system prefix (safe no-op when unsupported).
ENABLE_PROMPT_CACHE = True

DOCTYPE_NAME = "Agent Request"


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
# LLM factory — supports OpenAI and Gemini
# ---------------------------------------------------------------------------

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
    if provider not in ("OpenAI", "Gemini", "Claude"):
        log_agent_error(
            "Agent LLM Warning",
            f"Unknown AI provider '{provider}', defaulting to OpenAI",
        )
    openai_kwargs = {"model": model, "temperature": 0}
    if _openai_uses_responses_api(model):
        openai_kwargs["use_responses_api"] = True
    return ChatOpenAI(**openai_kwargs)


def _build_system_message(provider: str, system_prompt: str) -> SystemMessage:
    """Build the system message, adding provider-native prompt caching where supported.

    Anthropic supports an explicit `cache_control` breakpoint on the system block,
    which caches the large, stable instruction prefix (big cost saver on repeated
    tool rounds). OpenAI caches stable prefixes automatically, so a plain system
    message is enough there. Gemini/others fall back to a plain system message.
    Caching only reduces cost; it never changes model output.
    """
    if ENABLE_PROMPT_CACHE and (provider or "").strip() == "Claude":
        try:
            return SystemMessage(content=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }])
        except Exception:
            log_agent_error(
                "Agent Graph: anthropic cache system message",
                frappe.get_traceback(),
            )
    return SystemMessage(content=system_prompt)


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


def _pre_read_files(app_name: str, paths: list[str], max_files: int = 25) -> str:
    """Read files by path with line numbers for prompt injection.
    Large files are read in full — the agent needs complete context to avoid bad edits."""
    sections = []
    total_chars = 0
    max_per_file = 20000
    max_total = 120000

    for path in paths[:max_files]:
        content = agent_tools.read_file(app_name, path)
        if content.startswith("Error:") or content.startswith("Not a file:"):
            continue
        if len(content) > max_per_file:
            content = content[:max_per_file] + f"\n... (truncated at {max_per_file} chars — call read_file for remaining lines)"
        total_chars += len(content)
        if total_chars > max_total:
            sections.append(f"(skipping remaining files — {max_total} char limit reached, use read_file for more)")
            break
        sections.append(f"### FILE: {path}\n```\n{content}\n```")
    return "\n\n".join(sections) if sections else "(no files pre-loaded)"


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

    system_msg = _build_system_message(provider, system_prompt)
    task_msg = HumanMessage(content=task_prompt)

    rounds: list[dict] = []      # each: {"ai": AIMessage, "tools": [ToolMessage], "summary": [str]}
    compacted: list[str] = []    # summary lines for dropped (older) rounds
    injected: list = []          # directive nudges appended after the latest round
    edited_paths: list[str] = []
    total_tokens = (state or {}).get("tokens_used", 0)  # carry forward from prior phases

    wrote_anything = False
    read_only_streak = 0
    write_nudges = 0
    MAX_WRITE_NUDGES = 3
    READ_STREAK_LIMIT = 5
    NUDGE_TEXT = (
        "STOP exploring. You have read enough — the files you need are already "
        "in context (pre-loaded in the instructions and/or from the reads above). "
        "Do NOT call read_file, search_code, list_directory, get_file_outline or "
        "read_doctype_schema again unless an edit actually fails. Make the planned "
        "changes NOW: call replace_lines, insert_lines, edit_file or write_file to "
        "apply real edits, then validate_code on each changed .py/.js file. Your "
        "next message MUST include an edit tool call."
    )

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

    def estimate_chars():
        return sum(len(str(getattr(m, "content", ""))) for r in rounds for m in [r["ai"], *r["tools"]])

    def maybe_trim():
        trimmed = False
        while len(rounds) > KEEP_TOOL_ROUNDS:
            compacted.extend(_compact_round_summary(rounds.pop(0)))
            trimmed = True
        while len(rounds) > MIN_KEEP_ROUNDS and estimate_chars() > TRIM_CHAR_BUDGET:
            compacted.extend(_compact_round_summary(rounds.pop(0)))
            trimmed = True
        if trimmed:
            _publish_agent_log(request_name, "history_trim",
                kept_rounds=len(rounds),
                compacted_lines=len(compacted),
            )

    def account_tokens(response, round_label):
        nonlocal total_tokens
        usage = getattr(response, "usage_metadata", None)
        if usage:
            total_tokens += int(usage.get("total_tokens") or 0)
            _publish_agent_log(request_name, "token_usage",
                round=round_label,
                tokens_this_round=usage.get("total_tokens", 0),
                tokens_total=total_tokens,
            )
            _persist_token_usage(request_name, total_tokens)

    for round_num in range(max_rounds):
        maybe_trim()
        response = llm_with_tools.invoke(build_messages())
        account_tokens(response, round_num + 1)

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
                injected.append(HumanMessage(content=NUDGE_TEXT))
                _publish_agent_log(request_name, "write_nudge",
                    round=round_num + 1, attempt=write_nudges, trigger="no_tool_calls")
                continue
            return response_text, edited_paths, total_tokens

        round_entry = {"ai": response, "tools": [], "summary": []}
        round_had_write = False
        for tc in response.tool_calls:
            _publish_agent_log(request_name, "tool_call",
                tool_name=tc["name"],
                tool_args={k: (str(v)[:200] if len(str(v)) > 200 else v) for k, v in tc.get("args", {}).items()},
                round=round_num + 1,
            )

            fn = tool_map.get(tc["name"])
            if fn:
                try:
                    result = str(fn.invoke(tc["args"]))
                    if len(result) > 15000:
                        result = result[:15000] + "\n... (truncated)"
                except Exception as e:
                    log_agent_error(
                        f"Agent Graph: tool {tc['name']}",
                        f"request={request_name}\n{e}\n{frappe.get_traceback()}",
                    )
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tc['name']}"

            if tc["name"] in WRITE_TOOLS:
                if not result.startswith("Tool error:") and not result.startswith("Error:"):
                    round_had_write = True
                    wrote_anything = True
                    path_arg = tc.get("args", {}).get("path", "")
                    if path_arg and path_arg not in edited_paths:
                        edited_paths.append(path_arg)

            _publish_agent_log(request_name, "tool_result",
                tool_name=tc["name"],
                result_preview=result[:500],
                round=round_num + 1,
            )
            round_entry["tools"].append(ToolMessage(content=result, tool_call_id=tc["id"]))

            arg_preview = ", ".join(
                f"{k}={str(v)[:60]}" for k, v in list(tc.get("args", {}).items())[:3]
            )
            round_entry["summary"].append(
                f"[r{round_num + 1}] {tc['name']}({arg_preview}) -> {result[:COMPACT_RESULT_PREVIEW]}"
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
            injected.append(HumanMessage(content=NUDGE_TEXT))
            _publish_agent_log(request_name, "write_nudge",
                round=round_num + 1, attempt=write_nudges, trigger="read_streak")

    maybe_trim()
    final = llm.invoke(build_messages())
    account_tokens(final, max_rounds + 1)
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
    """Explore the codebase with read-only tools to gather context for planning."""
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Understanding", "started", "Exploring codebase to understand the request")
    prompt = get_understand_prompt(
        state.get("user_message", ""),
        state.get("request_type", "Improvement"),
        request_name=state.get("request_name"),
    )
    # Read-only tools keep exploration side-effect free.
    updates = _run_agent_turn(state, "Understanding", prompt, read_only_tools=True, max_rounds=MAX_TOOL_ROUNDS_PLANNING)
    if updates.get("error"):
        logs = _log_stage({**state, "stage_log": logs}, "Understanding", "failed", updates["error"][:200])
        updates["stage_log"] = logs
        return updates
    steps = updates.get("intermediate_steps") or []
    summary = _message_content_to_str(steps[-1].get("output", "") if steps else "")
    updates["understanding_summary"] = summary
    logs = _log_stage({**state, "stage_log": logs}, "Understanding", "completed", summary[:200])
    updates["stage_log"] = logs
    return updates


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

        response = llm.invoke([
            _build_system_message(provider, system_prompt),
            HumanMessage(content=plan_prompt),
        ])
        plan = _llm_response_text(response)

        total_tokens = state.get("tokens_used", 0)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            total_tokens += int(usage.get("total_tokens") or 0)
            _publish_agent_log(request_name, "token_usage",
                round=1,
                tokens_this_round=usage.get("total_tokens", 0),
                tokens_total=total_tokens,
            )
            _persist_token_usage(request_name, total_tokens)

        if not plan.strip():
            logs = _log_stage({**state, "stage_log": logs}, "Planning", "failed",
                "LLM returned empty plan")
            return {"error": "Plan generation returned empty response", "stage_log": logs}

        _publish_agent_log(request_name, "llm_response",
            preview=plan[:500], round=2)

        steps = list(state.get("intermediate_steps") or []) + [
            {"phase": "Planning", "output": plan[:MAX_PHASE_OUTPUT_CHARS]}
        ]

        logs = _log_stage({**state, "stage_log": logs}, "Planning", "completed",
            f"Plan generated ({len(plan)} chars)")

        return {
            "current_stage": "Planning",
            "plan": plan,
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

    Runs a single implementation pass, then a local syntax check on the edited
    .py/.js files so broken code never reaches bench. There is no LLM review loop.
    """
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Implementing", "started", "Applying code changes")

    app_name = state.get("target_app_name", "")
    plan_text = _message_content_to_str(state.get("plan", ""))
    understanding_text = _message_content_to_str(state.get("understanding_summary", ""))
    all_text = plan_text + "\n" + understanding_text
    file_paths = _extract_file_paths(all_text)
    file_contents = _pre_read_files(app_name, file_paths)
    logs = _log_stage(
        {**state, "stage_log": logs}, "Implementing", "progress",
        f"Pre-loaded {len(file_paths)} files: {', '.join(file_paths[:5])}"
    )

    base_prompt = get_implement_prompt(
        plan_text,
        understanding_text,
        state.get("user_message", ""),
        file_contents,
        request_name=state.get("request_name"),
    )

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
    Flow: Implement Changes (with a local syntax gate) → END.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("implement", implement_node)

    workflow.set_entry_point("implement")
    workflow.add_edge("implement", END)

    return workflow.compile()
