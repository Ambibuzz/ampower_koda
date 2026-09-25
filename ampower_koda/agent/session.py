"""One append-only conversation per request: investigation, approval, implementation, repairs.

Each prompt is an exact prefix of the next, so the provider cache covers it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import frappe
from langchain_core.messages import messages_from_dict, messages_to_dict
from langgraph.graph import END, StateGraph

from ampower_koda.agent import graph, koda_core, verification
from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.plan_contract import PlanValidationError, ground_plan_references, plan_to_markdown, validate_plan
from ampower_koda.agent.prompts import (
    get_explore_prompt,
    get_plan_feedback_prompt,
    get_session_request_prompt,
    get_explore_system_prompt,
)
from ampower_koda.agent.state import AgentState

#: Investigation calls before the model is told to submit; rounds, not tokens,
#: because reads stay cached for implementation.
PLAN_ROUNDS = 40
#: Calls allowed after that reminder to produce a valid plan.
PLAN_SUBMIT_ROUNDS = 4
#: The explore helper's report, bounded so it cannot flood the session.
EXPLORE_REPORT_CHARS = 8000

#: History keys that survive between jobs. Derived caches (seen results,
#: source memory, calibrator) are rebuilt by the loop.
PERSISTED = ("compacted", "task_prompt", "rounds_done", "failed_calls", "failure_causes",
             "write_generation", "tool_progress_generation", "round_cap_tokens", "final_cap_tokens",
             "evidence_chars", "proposed_plan", "approved", "investigated", "follow_up",
             "strategy_level_sent")

PLANNING = "plan"
IMPLEMENTATION = "work"


def _path(request_name: str, kind: str = IMPLEMENTATION) -> Path:
    name = f"{request_name}.plan.json" if kind == PLANNING else f"{request_name}.json"
    return Path(frappe.get_site_path("private", "koda_sessions", name))


def save(request_name: str, history: dict, kind: str = IMPLEMENTATION) -> None:
    """Persist the conversation so the next job continues it byte for byte."""
    data = {key: history[key] for key in PERSISTED if key in history}
    data["rounds"] = [{"number": r["number"], "ai": messages_to_dict([r["ai"]])[0],
                       "tools": messages_to_dict(r["tools"]), "summary": r.get("summary", [])}
                      for r in history.get("rounds", [])]
    data["followups"] = [{**{k: v for k, v in f.items() if k != "messages"},
                          "messages": messages_to_dict(f["messages"])}
                         for f in history.get("followups", [])]
    path = _path(request_name, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load(request_name: str, kind: str = IMPLEMENTATION) -> dict:
    """The saved conversation, or an empty history when there is none."""
    path = _path(request_name, kind)
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        history = {key: data[key] for key in PERSISTED if key in data}
        history["rounds"] = [{"number": r["number"], "ai": messages_from_dict([r["ai"]])[0],
                              "tools": messages_from_dict(r["tools"]), "summary": r.get("summary", [])}
                             for r in data.get("rounds", [])]
        history["followups"] = [{**{k: v for k, v in f.items() if k != "messages"},
                                 "messages": messages_from_dict(f["messages"])}
                                for f in data.get("followups", [])]
    except (OSError, ValueError, KeyError, TypeError):
        # An unreadable conversation starts over rather than failing every later pass.
        log_agent_error("Koda session: unreadable conversation", f"{path}\n{frappe.get_traceback()}")
        return {}
    return history


class PlanSink:
    """What ``submit_plan`` does during investigation: validate, check, accept."""

    def __init__(self, state: dict, coverage=None):
        self.state = state
        self.coverage = coverage
        self.checked = False
        self.plan = None
        self.findings = ""

    def accept(self, plan, findings: str) -> str:
        raw = plan
        if isinstance(plan, str):
            # Text plan: take the first complete object and ignore what trails it.
            try:
                raw, _ = json.JSONDecoder().raw_decode(plan.strip())
            except ValueError as error:
                return f"SUBMIT_FAILED: plan is not a valid JSON object ({error}). Send plan as an object."
        try:
            plan = validate_plan(raw)
            app_name = self.state.get("target_app_name")
            if app_name:
                plan = ground_plan_references(plan, lambda path: graph._app_file_exists(app_name, path),
                                              app_name=app_name)
        except PlanValidationError as error:
            return "SUBMIT_FAILED: fix these and submit again:\n- " + "\n- ".join(error.issues)
        if self.coverage is not None and not self.checked:
            # Checked once; the checker sees only the request and the plan.
            self.checked = True
            gaps = self.coverage(plan)
            if gaps:
                return ("SUBMIT_FAILED: an independent check of the plan against the request found:\n- "
                        + "\n- ".join(gaps)
                        + "\nRevise the plan, or state in assumptions why your reading is right, and submit again.")
        self.plan, self.findings = plan, str(findings or "").strip()
        return f"PLAN_ACCEPTED: {len(plan['tasks'])} task(s) sent to the user for approval. Stop here."


def _explorer(state: dict, llm, provider: str, request_name: str):
    """``explore(question)``: the retrieval core as a sub-agent that returns findings only."""
    app_name = state.get("target_app_name", "")

    def explore(question: str) -> str:
        question = str(question or "").strip()
        if not question:
            return "EXPLORE_FAILED: ask a question."
        spent = int(frappe.db.get_value("Agent Request", request_name, "tokens_used") or 0) if request_name else 0
        try:
            # Local ranking only: the paid reranker's spend never reaches the request's ledger.
            result = koda_core.understand(
                question=get_explore_prompt(question, request_name=request_name) + graph.CORE_TOOL_NOTE,
                app_name=app_name, llm=llm, provider=provider, request_name=request_name,
                system_prompt=get_explore_system_prompt(app_name or "target_app"),
                retrieval_query=question, utility_llm=llm, spent=spent, rerank=False,
            )
        except Exception:  # noqa: BLE001 - a helper failure is a tool result, not a failed request
            log_agent_error("Koda session: explore", frappe.get_traceback())
            return "EXPLORE_FAILED: the helper could not answer; search and read directly."
        graph.add_side_tokens(request_name, result.tokens - spent)
        if not result.ok:
            return f"EXPLORE_FAILED: {result.why}. Search and read directly."
        return graph._bounded_text(result.summary, EXPLORE_REPORT_CHARS)

    return explore


def plan_node(state: dict) -> dict:
    """Phase 1: investigate in the session and end with an accepted plan.

    With ``plan_feedback``, the user's answer is appended to the same investigation.
    """
    if state.get("error"):
        return {"error": state["error"]}
    request_name = state.get("request_name", "")
    feedback = str(state.get("plan_feedback") or "").strip()
    logs = graph._log_stage(state, "Planning", "started",
                            "Revising the plan from feedback" if feedback else "Investigating in one session")
    provider = state.get("ai_provider", "OpenAI")
    model = state.get("ai_model", "gpt-4o-mini")
    app_name = state.get("target_app_name", "")
    try:
        llm = graph._get_llm(provider=provider, model=model, session_id=request_name)

        def coverage(plan):
            # Book the check's tokens as side tokens, or the loop's next write overwrites them.
            start = (int(frappe.db.get_value("Agent Request", request_name, "tokens_used") or 0) if request_name
                     else int(state.get("tokens_used") or 0))
            checker = graph._get_llm(provider=provider, model=model, session_id=request_name,
                                     reasoning_effort="medium")
            gaps, total = graph._plan_coverage_gaps(
                plan, state.get("user_message", ""), llm=checker, provider=provider,
                budget=graph._plan_budget(provider, state), request_name=request_name, total_tokens=start)
            graph.add_side_tokens(request_name, total - start)
            return gaps

        history = load(request_name, PLANNING) if feedback else {}
        if not history.get("task_prompt"):
            history = {"task_prompt": get_session_request_prompt(
                state.get("user_message", ""), state.get("request_type", "Improvement"),
                verification.contract_context(verification.prepare_contract(app_name)), request_name=request_name)}
        if feedback:
            current = state.get("plan_object")
            edited = (json.dumps(current, ensure_ascii=False, indent=1)
                      if current and current != history.get("proposed_plan") else "")
            graph._queue_directive(history, get_plan_feedback_prompt(feedback, edited))
        # The user's answer outranks a second opinion on the request's words.
        sink = PlanSink(state, None if feedback else coverage)
        prompt = history["task_prompt"]
        session = {"plan_sink": sink, "explorer": _explorer(state, llm, provider, request_name),
                   "stop_when": lambda: sink.plan is not None}
        updates = graph._run_agent_turn(state, "Planning", prompt, read_only_tools=True,
                                        max_rounds=PLAN_ROUNDS, history=history, session=session)
        if sink.plan is None and not updates.get("error"):
            graph._queue_directive(history, (
                "Investigation time is up. Call submit_plan now with the plan and your findings; record "
                "what you could not verify in the findings and the plan's assumptions."))
            updates.update(graph._run_agent_turn({**state, **updates}, "Planning", prompt, read_only_tools=True,
                                                 max_rounds=PLAN_SUBMIT_ROUNDS, history=history, session=session))
        if sink.plan is None:
            error = updates.get("error") or "Investigation ended without a valid plan."
            logs = graph._log_stage({**state, "stage_log": logs}, "Planning", "failed", error[:200])
            return {"error": error, "stage_log": logs, "tokens_used": updates.get("tokens_used", 0)}
        history["proposed_plan"] = sink.plan
        save(request_name, history, PLANNING)
        # An implementation of an older plan must never continue under this one.
        _path(request_name).unlink(missing_ok=True)
        plan = plan_to_markdown(sink.plan)
        steps = list(state.get("intermediate_steps") or []) + [{"phase": "Planning", "output": plan}]
        logs = graph._log_stage({**state, "stage_log": logs}, "Planning", "completed",
                                f"Plan submitted in session ({len(sink.plan['tasks'])} task(s))")
        return {
            "current_stage": "Planning",
            "plan": plan,
            "plan_object": sink.plan,
            "understanding_summary": sink.findings,
            "intermediate_steps": steps,
            "stage_log": logs,
            "tokens_used": int(updates.get("tokens_used") or 0),
        }
    except Exception as error:  # noqa: BLE001 - reported on the request, never a traceback death
        log_agent_error("Koda session: planning", f"request={request_name}\n{frappe.get_traceback()}")
        logs = graph._log_stage({**state, "stage_log": logs}, "Planning", "failed", str(error)[:200])
        return {"error": str(error), "stage_log": logs}


def build_planning_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", END)
    return workflow.compile()

