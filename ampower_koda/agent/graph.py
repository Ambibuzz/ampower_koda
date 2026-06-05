# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# LangGraph workflows: Planning (Understand→Plan) and Execution (Implement→Review→Bench→Deploy)

import json
import os
import re as _re
import subprocess
from datetime import datetime
from functools import partial

import frappe
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

from ampower_koda.agent.state import AgentState
from ampower_koda.agent import tools as agent_tools
from ampower_koda.agent.prompts import (
    get_system_prompt,
    get_understand_prompt,
    get_plan_prompt,
    get_implement_prompt,
    get_review_prompt,
)
from ampower_koda.agent.git_ops import (
    create_branch,
    commit_changes,
    push_branch,
    create_pull_request,
    generate_branch_name,
    run_git,
    get_repo_root,
)


MAX_TOOL_ROUNDS_PLANNING = 40
MAX_TOOL_ROUNDS_EXECUTION = 30
MAX_TOOL_ROUNDS_REVIEW = 10
WRITE_TOOLS = {"replace_lines", "insert_lines", "edit_file", "write_file"}

DOCTYPE_NAME = "AI Agent Request"


# ---------------------------------------------------------------------------
# LLM factory — supports OpenAI and Gemini
# ---------------------------------------------------------------------------

def _get_llm(provider: str = "OpenAI", model: str = "gpt-4o-mini"):
    """
    Factory function to initialize the appropriate Large Language Model (LLM) 
    based on the user's preferred provider and model.
    """
    provider = (provider or "OpenAI").strip()
    model = (model or "gpt-4o-mini").strip()
    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    if provider == "Claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    if provider not in ("OpenAI", "Gemini", "Claude"):
        frappe.log_error(f"Unknown AI provider '{provider}', defaulting to OpenAI", "Agent LLM Warning")
    return ChatOpenAI(model=model, temperature=0)


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
            pass

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
        pass


# ---------------------------------------------------------------------------
# Tool builders — closure over app_name
# ---------------------------------------------------------------------------

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

def _run_tool_calling_loop(llm, tools, prompt: str, request_name: str = "", max_rounds: int = 20,state: dict = None) -> tuple[str, list[str], int]:
    """
    Run a tool-calling loop, publishing every tool call and LLM response via realtime.
    Returns (final_text, list_of_edited_file_paths, total_tokens_used).

    On each round the LLM is invoked with the full message history. If the LLM
    returns no tool calls, the loop exits early and returns that response as the
    final text.

    If max_rounds is reached without the LLM stopping, one final llm.invoke()
    is fired WITHOUT tools bound — this forces a plain text conclusion rather
    than an infinite loop. Edited paths and token counts collected up to that
    point are still returned.
    """
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    messages = [HumanMessage(content=prompt)]
    edited_paths = []
    total_tokens = (state or {}).get("tokens_used", 0) # carry forward from prior phases

    for round_num in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        usage = getattr(response, "usage_metadata", None)
        if usage:
            total_tokens += int(usage.get("total_tokens") or 0)
            _publish_agent_log(request_name, "token_usage",
                round=round_num + 1,
                tokens_this_round=usage.get("total_tokens", 0),
                tokens_total=total_tokens,
            )

        response_text = getattr(response, "content", "") or ""
        if response_text:
            _publish_agent_log(request_name, "llm_response",
                preview=response_text[:4000],
                round=round_num + 1,
            )

        if not getattr(response, "tool_calls", None):
            return response_text, edited_paths, total_tokens

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
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {tc['name']}"

            if tc["name"] in WRITE_TOOLS:
                if not result.startswith("Tool error:") and not result.startswith("Error:"):
                    path_arg = tc.get("args", {}).get("path", "")
                    if path_arg and path_arg not in edited_paths:
                        edited_paths.append(path_arg)

            _publish_agent_log(request_name, "tool_result",
                tool_name=tc["name"],
                result_preview=result[:500],
                round=round_num + 1,
            )
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    final = llm.invoke(messages)
    usage = getattr(final, "usage_metadata", None)
    if usage:
        total_tokens += int(usage.get("total_tokens") or 0)
    return (getattr(final, "content", "") or ""), edited_paths, total_tokens


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
        full_prompt = f"{system_prompt}\n\n{prompt}"
        content, tool_edited_paths, total_tokens = _run_tool_calling_loop(
            llm, tools, full_prompt,
            request_name=request_name,
            max_rounds=max_rounds,
            state=state,      
        )
        max_output = 30000 if phase in ("Understanding", "Planning") else 8000
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
    """
    The 'Understanding' node. Here, the agent explores the codebase to build 
     a comprehensive mental model of the relevant files and logic before 
     attempting any planning.
    """
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Understanding", "started", "Exploring codebase to understand the request")
    prompt = get_understand_prompt(
        state.get("user_message", ""),
        state.get("request_type", "Improvement"),
        request_name=state.get("request_name"),
    )
    # The agent is given read-only tools for this phase to ensure a safe exploration.
    updates = _run_agent_turn(state, "Understanding", prompt, read_only_tools=True, max_rounds=MAX_TOOL_ROUNDS_PLANNING)
    if updates.get("error"):
        logs = _log_stage({**state, "stage_log": logs}, "Understanding", "failed", updates["error"][:200])
        updates["stage_log"] = logs
        return updates
    steps = updates.get("intermediate_steps") or []
    summary = steps[-1].get("output", "") if steps else ""
    updates["understanding_summary"] = summary
    logs = _log_stage({**state, "stage_log": logs}, "Understanding", "completed", summary[:200])
    updates["stage_log"] = logs
    return updates


def plan_node(state: dict) -> dict:
    """
    The 'Planning' node. The agent synthesizes its discoveries from the 
    Understanding phase into a concrete, step-by-step implementation plan.
    """
    if state.get("error"):
        return {"error": state["error"]}

    logs = _log_stage(state, "Planning", "started", "Creating detailed implementation plan from codebase analysis")

    try:
        provider = state.get("ai_provider", "OpenAI")
        model = state.get("ai_model", "gpt-4o-mini")
        app_name = state.get("target_app_name", "target_app")
        request_name = state.get("request_name", "")
        understanding = state.get("understanding_summary", "")
        user_message = state.get("user_message", "")

        if not understanding.strip():
            logs = _log_stage({**state, "stage_log": logs}, "Planning", "failed",
                "No understanding summary available — explore phase produced no output")
            return {"error": "Understanding phase produced no output", "stage_log": logs}

        llm = _get_llm(provider=provider, model=model)
        system_prompt = get_system_prompt(app_name, request_name=request_name)
        plan_prompt = get_plan_prompt(understanding, user_message, request_name=request_name)
        full_prompt = f"{system_prompt}\n\n{plan_prompt}"

        _publish_agent_log(request_name, "llm_response",
            preview="Generating detailed plan from codebase analysis...", round=1)

        response = llm.invoke([HumanMessage(content=full_prompt)])
        plan = getattr(response, "content", "") or ""

        if not plan.strip():
            logs = _log_stage({**state, "stage_log": logs}, "Planning", "failed",
                "LLM returned empty plan")
            return {"error": "Plan generation returned empty response", "stage_log": logs}

        _publish_agent_log(request_name, "llm_response",
            preview=plan[:500], round=2)

        steps = list(state.get("intermediate_steps") or []) + [
            {"phase": "Planning", "output": plan[:30000]}
        ]

        logs = _log_stage({**state, "stage_log": logs}, "Planning", "completed",
            f"Plan generated ({len(plan)} chars)")

        return {
            "current_stage": "Planning",
            "plan": plan,
            "intermediate_steps": steps,
            "stage_log": logs,
        }
    except Exception as e:
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
    """
    The 'Implementing' node. In this phase, the agent applies the approved plan 
    by writing code and creating new files as needed.
    """
    if state.get("error"):
        return {"error": state["error"]}
    attempt = (state.get("review_attempts") or 0) + 1
    logs = _log_stage(state, "Implementing", "started", f"Applying code changes (attempt {attempt})")

    app_name = state.get("target_app_name", "")
    all_text = (state.get("plan", "") + "\n" + state.get("understanding_summary", ""))
    file_paths = _extract_file_paths(all_text)
    file_contents = _pre_read_files(app_name, file_paths)
    logs = _log_stage(
        {**state, "stage_log": logs}, "Implementing", "progress",
        f"Pre-loaded {len(file_paths)} files: {', '.join(file_paths[:5])}"
    )

    base_prompt = get_implement_prompt(
        state.get("plan", ""),
        state.get("understanding_summary", ""),
        state.get("user_message", ""),
        file_contents,
        request_name=state.get("request_name"),
    )

    review_notes = state.get("review_notes", "")
    if attempt > 1 and review_notes:
        base_prompt += f"""

## PREVIOUS REVIEW FEEDBACK — FIX THESE ISSUES
The previous implementation attempt was reviewed and FAILED. Here is what the reviewer found wrong:
{review_notes}

You MUST fix ALL of these issues in this attempt. Read the affected files first to see the current state."""

    updates = _run_agent_turn(state, "Implementing", base_prompt, read_only_tools=False, max_rounds=MAX_TOOL_ROUNDS_EXECUTION)
    if updates.get("error"):
        logs = _log_stage({**state, "stage_log": logs}, "Implementing", "failed", updates["error"][:200])
        updates["stage_log"] = logs
        return updates

    tool_edited = updates.pop("_tool_edited_paths", [])
    steps = updates.get("intermediate_steps") or []
    last_out = steps[-1].get("output", "") if steps else ""
    text_paths = _extract_file_paths(last_out)

    all_paths = list(dict.fromkeys(tool_edited + text_paths))

    edits = list(state.get("edits_made") or [])
    for p in all_paths:
        if not any(e.get("path") == p for e in edits):
            edits.append({"path": p, "summary": f"Modified in attempt {attempt}"})
    if not all_paths:
        edits.append({"summary": last_out[:300]})
    updates["edits_made"] = edits

    logs = _log_stage(
        {**state, "stage_log": logs}, "Implementing", "completed",
        f"Files edited: {', '.join(all_paths[:8]) if all_paths else 'see summary'}"
    )
    updates["stage_log"] = logs
    return updates


def review_node(state: dict) -> dict:
    """
    The 'Reviewing' node. The agent takes an adversarial look at its own 
    implementation to identify potential bugs, syntax errors, or missed requirements.
    """
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Reviewing", "started", "Reviewing implemented changes")
    edits = state.get("edits_made", [])
    prompt = get_review_prompt(edits, state.get("user_message", ""), request_name=state.get("request_name"))
    updates = _run_agent_turn(state, "Reviewing", prompt, read_only_tools=True, max_rounds=MAX_TOOL_ROUNDS_REVIEW)
    if updates.get("error"):
        logs = _log_stage({**state, "stage_log": logs}, "Reviewing", "failed", updates["error"][:200])
        updates["stage_log"] = logs
        return updates

    steps = updates.get("intermediate_steps") or []
    last_out = (steps[-1].get("output", "") if steps else "")

    if "REVIEW_PASSED" not in last_out.upper():
        try:
            provider = state.get("ai_provider", "OpenAI")
            model = state.get("ai_model", "gpt-4o-mini")
            llm = _get_llm(provider=provider, model=model)
            verdict_response = llm.invoke([HumanMessage(
                content=(
                    "Based on the review you just performed, give your final verdict.\n"
                    "Reply with EXACTLY one of:\n"
                    "- REVIEW_PASSED=yes (if all changes are correct)\n"
                    "- REVIEW_PASSED=no (followed by what's wrong)\n\n"
                    f"Review output so far: {last_out[:2000]}"
                )
            )])
            verdict = getattr(verdict_response, "content", "") or ""
            if verdict.strip():
                last_out = verdict
        except Exception:
            pass

    last_out_upper = last_out.upper()
    passed = "REVIEW_PASSED=YES" in last_out_upper
    updates["review_passed"] = passed
    updates["review_notes"] = last_out[:500]
    updates["review_attempts"] = (state.get("review_attempts") or 0) + 1
    result_msg = "Review PASSED" if passed else f"Review FAILED: {last_out[:100]}"
    logs = _log_stage({**state, "stage_log": logs}, "Reviewing", "completed", result_msg)
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


def bench_node(state: dict) -> dict:
    """
    The 'Building' node. This node executes the necessary bench commands 
    (like migrations or asset builds) to integrate the changes into the site.
    """
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Building", "started", "Running bench commands to apply changes")

    edits = state.get("edits_made", [])
    edited_paths = [e.get("path", "") for e in edits if e.get("path")]
    app_name = state.get("target_app_name", "")
    request_name = state.get("request_name", "")

    bench_root = os.path.join(frappe.get_app_path("frappe"), "..", "..", "..")
    bench_root = os.path.normpath(bench_root)

    site_name = frappe.local.site
    bench_env = _get_bench_env()

    has_doctype_changes = any(
        p.endswith(".json") and "/doctype/" in p for p in edited_paths
    )
    has_js_css_changes = any(
        p.endswith((".js", ".css", ".html")) for p in edited_paths
    )

    cmds = []
    if has_doctype_changes:
        cmds.append(f"bench --site {site_name} migrate")
    if has_js_css_changes:
        cmds.append(f"bench build --app {app_name}")
    cmds.append(f"bench --site {site_name} clear-cache")
    cmds.append("supervisorctl restart all")

    bench_output_parts = []
    for cmd in cmds:
        _log_stage({**state, "stage_log": logs}, "Building", "progress", f"Running: {cmd}")
        _publish_agent_log(request_name, "bench_command", command=cmd)
        try:
            result = subprocess.run(
                cmd.split(),
                cwd=bench_root,
                capture_output=True,
                text=True,
                timeout=900,
                env=bench_env,
            )
            output = (result.stdout or "") + (result.stderr or "")
            success = result.returncode == 0
            status_str = "OK" if success else f"FAILED (exit {result.returncode})"
            entry = f"$ {cmd}\n{status_str}\n{output.strip()}\n"
            bench_output_parts.append(entry)
            _publish_agent_log(request_name, "bench_result",
                command=cmd,
                success=success,
                output_preview=output[:500],
            )
        except subprocess.TimeoutExpired:
            entry = f"$ {cmd}\nTIMEOUT after 900s\n"
            bench_output_parts.append(entry)
            _publish_agent_log(request_name, "bench_result",
                command=cmd, success=False, output_preview="Timed out after 900s")
        except Exception as e:
            entry = f"$ {cmd}\nERROR: {e}\n"
            bench_output_parts.append(entry)

    bench_log = "\n".join(bench_output_parts)

    if request_name:
        try:
            frappe.db.set_value(DOCTYPE_NAME, request_name, "bench_log", bench_log[:50000])
            frappe.db.commit()
        except Exception:
            pass

    logs = _log_stage({**state, "stage_log": logs}, "Building", "completed",
        f"Ran {len(cmds)} bench commands")
    return {
        "current_stage": "Building",
        "bench_log": bench_log,
        "stage_log": logs,
    }


def deploy_node(state: dict) -> dict:
    """
    The 'Pushing' node. Finalizes the work by committing the changes to a branch, 
    pushing to the remote repository, and optionally opening a pull request.
    """
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Pushing", "started", "Creating branch, committing, pushing and opening PR")

    app_name = state.get("target_app_name", "")
    request_name = state.get("request_name", "AGENT-0000")
    request_type = state.get("request_type", "Improvement")
    plan = state.get("plan", "")
    user_message = state.get("user_message", "")[:500]
    base_branch = state.get("base_branch", "main")
    branch_prefix = state.get("branch_prefix", "ai-agent/")
    repo_url = state.get("github_repo_url", "")
    token = state.get("github_token", "")
    git_user_name = state.get("git_user_name", "AI Agent")
    git_user_email = state.get("git_user_email", "ai-agent@ampower.com")

    branch_name = generate_branch_name(request_name, branch_prefix)

    repo_root = get_repo_root(app_name)
    ok, diff_out = run_git(["diff", "--stat", "HEAD"], cwd=repo_root)
    ok2, untracked = run_git(["ls-files", "--others", "--exclude-standard"], cwd=repo_root)
    has_changes = bool((diff_out or "").strip()) or bool((untracked or "").strip())
    if not has_changes:
        logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed",
            "No file changes detected on disk. The implement phase did not produce any actual edits.")
        return {
            "current_stage": "Pushing",
            "error": "No code changes were produced. The implement phase did not modify any files on disk.",
            "branch_name": "",
            "stage_log": logs,
        }

    logs = _log_stage({**state, "stage_log": logs}, "Pushing", "progress",
        f"Verified file changes exist: {(diff_out or untracked or '')[:100]}")

    ok, msg = create_branch(app_name, branch_name, base_branch)
    if not ok:
        logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed", f"Branch creation failed: {msg[:150]}")
        return {"current_stage": "Pushing", "error": f"create_branch: {msg}", "branch_name": branch_name, "stage_log": logs}

    logs = _log_stage({**state, "stage_log": logs}, "Pushing", "progress", f"Branch created: {branch_name}")

    commit_msg = f"[AI Agent] {request_type}: {request_name}\n\n{user_message[:200]}"
    ok, msg = commit_changes(app_name, commit_msg, git_user_name, git_user_email)
    if not ok:
        if "No changes" in msg:
            logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed",
                "No files were actually changed. The implement phase edits may have all failed.")
            return {"current_stage": "Pushing", "error": "No code changes were produced.", "branch_name": branch_name, "stage_log": logs}
        logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed", f"Commit failed: {msg[:150]}")
        return {"current_stage": "Pushing", "error": f"commit: {msg}", "branch_name": branch_name, "stage_log": logs}

    logs = _log_stage({**state, "stage_log": logs}, "Pushing", "progress", "Changes committed")

    ok, msg = push_branch(app_name, branch_name, repo_url, token)
    if not ok:
        logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed", f"Push failed: {msg[:150]}")
        return {"current_stage": "Pushing", "error": f"push: {msg}", "branch_name": branch_name, "stage_log": logs}

    logs = _log_stage({**state, "stage_log": logs}, "Pushing", "progress", "Branch pushed to remote")

    pr_title = f"[AI Agent] {request_type}: {request_name}"
    pr_body = f"## Request\n{user_message}\n\n## Plan\n{plan}"
    ok, msg, pr_url, pr_number = create_pull_request(pr_title, pr_body, branch_name, repo_url, token, base_branch)
    if not ok:
        logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed", f"PR creation failed: {msg[:150]}")
        return {"current_stage": "Pushing", "error": f"PR: {msg}", "branch_name": branch_name, "pr_url": None, "pr_number": None, "stage_log": logs}

    logs = _log_stage({**state, "stage_log": logs}, "Pushing", "completed", f"PR #{pr_number} created: {pr_url}")
    return {
        "current_stage": "Completed",
        "branch_name": branch_name,
        "pr_url": pr_url,
        "pr_number": pr_number,
        "error": None,
        "stage_log": logs,
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_retry_implement(state: dict) -> str:
    if state.get("error"):
        return "done"
    if state.get("review_passed"):
        return "done"
    if (state.get("review_attempts") or 0) >= 3:
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
    Flow: Implement Changes → Review Work → (Retry if needed).
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
