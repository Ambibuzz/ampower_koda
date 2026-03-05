# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# LangGraph state machine: Understand -> Plan -> Implement -> Review -> Deploy

import json
import re as _re
from datetime import datetime

import frappe
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

from ampower_ai_agents.agent.state import AgentState
from ampower_ai_agents.agent import tools as agent_tools
from ampower_ai_agents.agent.prompts import (
    get_system_prompt,
    get_understand_prompt,
    get_plan_prompt,
    get_implement_prompt,
    get_review_prompt,
)
from ampower_ai_agents.agent.git_ops import (
    create_branch,
    commit_changes,
    push_branch,
    create_pull_request,
    generate_branch_name,
    run_git,
    get_repo_root,
)


MAX_TOOL_ROUNDS = 14

DOCTYPE_NAME = "AI Agent Request"


def _get_target_app_name() -> str:
    """Read the target app name from AI Agents Settings."""
    try:
        settings = frappe.get_single("AI Agents Settings")
        return (settings.target_app_name or "").strip()
    except Exception:
        return ""


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


def _pre_read_files(paths: list[str], max_files: int = 10) -> str:
    """Read files by path and format them for prompt injection."""
    sections = []
    total_chars = 0
    for path in paths[:max_files]:
        content = agent_tools.read_file(path)
        if content.startswith("Error:") or content.startswith("Not a file:"):
            continue
        if len(content) > 8000:
            content = content[:8000] + f"\n... (file truncated at 8000/{len(content)} chars)"
        total_chars += len(content)
        if total_chars > 50000:
            sections.append("(skipping remaining files — context limit reached)")
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
            frappe.publish_realtime("agent_progress", {
                "request_name": request_name,
                "status": stage,
                "stage": stage,
                "stage_status": status,
                "message": summary[:200],
            })
        except Exception:
            pass

    return logs


def _get_llm(model: str = "gpt-4o-mini"):
    return ChatOpenAI(model=model, temperature=0)


def _make_tools(read_only: bool = False):
    """Build LangChain tools. read_only excludes write_file and edit_file."""
    @tool
    def list_directory(path: str) -> str:
        """List files and directories at path (relative to app root)."""
        return agent_tools.list_directory(path)

    @tool
    def read_file(path: str) -> str:
        """Read file content. Path relative to app root."""
        return agent_tools.read_file(path)

    @tool
    def search_code(pattern: str, path: str = "") -> str:
        """Search for regex pattern in codebase. path is optional directory."""
        return agent_tools.search_code(pattern, path)

    @tool
    def read_doctype_schema(doctype_name: str) -> str:
        """Read DocType JSON schema. doctype_name e.g. TM Task."""
        return agent_tools.read_doctype_schema(doctype_name)

    out = [list_directory, read_file, search_code, read_doctype_schema]
    if not read_only:
        @tool
        def write_file(path: str, content: str) -> str:
            """Write or overwrite a file. Path relative to app root."""
            return agent_tools.write_file(path, content)

        @tool
        def edit_file(path: str, old_string: str, new_string: str) -> str:
            """Replace old_string with new_string in file (first occurrence)."""
            return agent_tools.edit_file(path, old_string, new_string)

        out.extend([write_file, edit_file])
    return out


def _run_tool_calling_loop(llm, tools, prompt: str) -> str:
    """Run a tool-calling loop manually (no nested LangGraph graph).
    Returns the final text response from the LLM."""
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    messages = [HumanMessage(content=prompt)]

    for _ in range(MAX_TOOL_ROUNDS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return getattr(response, "content", "") or ""

        for tc in response.tool_calls:
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
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    final = llm.invoke(messages)
    return getattr(final, "content", "") or ""


def _run_agent_turn(
    state: dict,
    phase: str,
    prompt: str,
    read_only_tools: bool,
    model: str,
) -> dict:
    """Run one agent turn for the given phase. Returns state updates."""
    try:
        tools = _make_tools(read_only=read_only_tools)
        llm = _get_llm(model=model)
        app_name = _get_target_app_name()
        system_prompt = get_system_prompt(app_name or "target_app")
        full_prompt = f"{system_prompt}\n\n{prompt}"
        content = _run_tool_calling_loop(llm, tools, full_prompt)
        steps = list(state.get("intermediate_steps") or []) + [
            {"phase": phase, "output": (content[:8000] if content else "")}
        ]
        return {
            "current_stage": phase,
            "intermediate_steps": steps,
        }
    except Exception as e:
        steps = list(state.get("intermediate_steps") or []) + [
            {"phase": phase, "output": f"Error: {e}"}
        ]
        return {
            "current_stage": phase,
            "intermediate_steps": steps,
            "error": str(e),
        }


def understand_node(state: dict) -> dict:
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Understanding", "started", "Exploring codebase to understand the request")
    prompt = get_understand_prompt(
        state.get("user_message", ""),
        state.get("request_type", "Improvement"),
    )
    updates = _run_agent_turn(
        state, "Understanding", prompt, read_only_tools=True, model=state.get("ai_model", "gpt-4o-mini")
    )
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
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Planning", "started", "Creating implementation plan")
    prompt = get_plan_prompt(state.get("understanding_summary", ""))
    updates = _run_agent_turn(
        state, "Planning", prompt, read_only_tools=True, model=state.get("ai_model", "gpt-4o-mini")
    )
    if updates.get("error"):
        logs = _log_stage({**state, "stage_log": logs}, "Planning", "failed", updates["error"][:200])
        updates["stage_log"] = logs
        return updates
    steps = updates.get("intermediate_steps") or []
    plan = steps[-1].get("output", "") if steps else ""
    updates["plan"] = plan
    logs = _log_stage({**state, "stage_log": logs}, "Planning", "completed", plan[:200])
    updates["stage_log"] = logs
    return updates


def implement_node(state: dict) -> dict:
    if state.get("error"):
        return {"error": state["error"]}
    attempt = (state.get("review_attempts") or 0) + 1
    logs = _log_stage(state, "Implementing", "started", f"Applying code changes (attempt {attempt})")

    all_text = (state.get("plan", "") + "\n" + state.get("understanding_summary", ""))
    file_paths = _extract_file_paths(all_text)
    file_contents = _pre_read_files(file_paths)
    logs = _log_stage(
        {**state, "stage_log": logs}, "Implementing", "progress",
        f"Pre-loaded {len(file_paths)} files: {', '.join(file_paths[:5])}"
    )

    prompt = get_implement_prompt(
        state.get("plan", ""),
        state.get("understanding_summary", ""),
        state.get("user_message", ""),
        file_contents,
    )
    updates = _run_agent_turn(
        state, "Implementing", prompt, read_only_tools=False, model=state.get("ai_model", "gpt-4o-mini")
    )
    if updates.get("error"):
        logs = _log_stage({**state, "stage_log": logs}, "Implementing", "failed", updates["error"][:200])
        updates["stage_log"] = logs
        return updates
    steps = updates.get("intermediate_steps") or []
    last_out = steps[-1].get("output", "") if steps else ""

    edited_paths = _extract_file_paths(last_out)
    edits = list(state.get("edits_made") or [])
    for p in edited_paths:
        edits.append({"path": p, "summary": f"Modified in attempt {attempt}"})
    if not edited_paths:
        edits.append({"summary": last_out[:300]})
    updates["edits_made"] = edits

    logs = _log_stage(
        {**state, "stage_log": logs}, "Implementing", "completed",
        f"Files edited: {', '.join(edited_paths[:5]) if edited_paths else 'see summary'}"
    )
    updates["stage_log"] = logs
    return updates


def review_node(state: dict) -> dict:
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Reviewing", "started", "Reviewing implemented changes")
    edits = state.get("edits_made", [])
    prompt = get_review_prompt(edits, state.get("user_message", ""))
    updates = _run_agent_turn(
        state, "Reviewing", prompt, read_only_tools=True, model=state.get("ai_model", "gpt-4o-mini")
    )
    if updates.get("error"):
        logs = _log_stage({**state, "stage_log": logs}, "Reviewing", "failed", updates["error"][:200])
        updates["stage_log"] = logs
        return updates
    steps = updates.get("intermediate_steps") or []
    last_out = (steps[-1].get("output", "") if steps else "").upper()
    passed = "REVIEW_PASSED=YES" in last_out
    updates["review_passed"] = passed
    updates["review_notes"] = last_out[:500]
    updates["review_attempts"] = (state.get("review_attempts") or 0) + 1
    result_msg = "Review PASSED" if passed else f"Review FAILED: {last_out[:100]}"
    logs = _log_stage({**state, "stage_log": logs}, "Reviewing", "completed", result_msg)
    updates["stage_log"] = logs
    return updates


def deploy_node(state: dict) -> dict:
    """Create branch, commit, push, open PR."""
    if state.get("error"):
        return {"error": state["error"]}
    logs = _log_stage(state, "Pushing", "started", "Creating branch, committing, pushing and opening PR")
    request_name = state.get("request_name", "AGENT-0000")
    request_type = state.get("request_type", "Improvement")
    plan = state.get("plan", "")
    user_message = state.get("user_message", "")[:500]
    branch_name = generate_branch_name(request_name, request_type)

    ok, diff_out = run_git(["diff", "--stat", "HEAD"], cwd=get_repo_root())
    ok2, untracked = run_git(["ls-files", "--others", "--exclude-standard"], cwd=get_repo_root())
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

    ok, msg = create_branch(branch_name)
    if not ok:
        logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed", f"Branch creation failed: {msg[:150]}")
        return {"current_stage": "Pushing", "error": f"create_branch: {msg}", "branch_name": branch_name, "stage_log": logs}

    logs = _log_stage({**state, "stage_log": logs}, "Pushing", "progress", f"Branch created: {branch_name}")

    commit_msg = f"[AI Agent] {request_type}: {request_name}\n\n{user_message[:200]}"
    ok, msg = commit_changes(commit_msg)
    if not ok:
        if "No changes" in msg:
            logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed",
                "No files were actually changed. The implement phase edits may have all failed.")
            return {"current_stage": "Pushing", "error": "No code changes were produced.", "branch_name": branch_name, "stage_log": logs}
        logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed", f"Commit failed: {msg[:150]}")
        return {"current_stage": "Pushing", "error": f"commit: {msg}", "branch_name": branch_name, "stage_log": logs}

    logs = _log_stage({**state, "stage_log": logs}, "Pushing", "progress", "Changes committed")

    ok, msg = push_branch(branch_name)
    if not ok:
        logs = _log_stage({**state, "stage_log": logs}, "Pushing", "failed", f"Push failed: {msg[:150]}")
        return {"current_stage": "Pushing", "error": f"push: {msg}", "branch_name": branch_name, "stage_log": logs}

    logs = _log_stage({**state, "stage_log": logs}, "Pushing", "progress", "Branch pushed to remote")

    pr_title = f"[AI Agent] {request_type}: {request_name}"
    pr_body = f"## Request\n{user_message}\n\n## Plan\n{plan}"
    ok, msg, pr_url, pr_number = create_pull_request(pr_title, pr_body, branch_name)
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


def should_retry_implement(state: dict) -> str:
    if state.get("error"):
        return "deploy"
    if state.get("review_passed"):
        return "deploy"
    if (state.get("review_attempts") or 0) >= 2:
        return "deploy"
    return "implement"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("understand", understand_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("implement", implement_node)
    workflow.add_node("review", review_node)
    workflow.add_node("deploy", deploy_node)

    workflow.set_entry_point("understand")
    workflow.add_edge("understand", "plan")
    workflow.add_edge("plan", "implement")
    workflow.add_edge("implement", "review")
    workflow.add_conditional_edges("review", should_retry_implement, {"implement": "implement", "deploy": "deploy"})
    workflow.add_edge("deploy", END)

    return workflow.compile()
