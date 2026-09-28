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

from ampower_koda.agent import checkpoint, graph, koda_core, verification
from ampower_koda.agent.errors import log_agent_error
from ampower_koda.agent.plan_contract import PlanValidationError, ground_plan_references, plan_to_markdown, validate_plan
from ampower_koda.agent.prompts import (
    get_explore_prompt,
    get_plan_feedback_prompt,
    get_session_approval_prompt,
    get_session_follow_up_prompt,
    get_session_implemented_plan_prompt,
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


#: Submits whose independent coverage check may fail before the plan is accepted without it.
MAX_COVERAGE_FAILURES = 2


class PlanSink:
    """What ``submit_plan`` does during investigation: validate, check, accept."""

    def __init__(self, state: dict, coverage=None):
        self.state = state
        self.coverage = coverage
        self.checked = False
        self.failures = 0
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
            # Checked once; the checker sees only the request and the plan. Marked only once it
            # answers: a checker that raised runs again on the next submit, and one that keeps
            # failing is logged and skipped, since a second opinion must not block planning.
            try:
                gaps = self.coverage(plan)
            except Exception:
                self.failures += 1
                if self.failures < MAX_COVERAGE_FAILURES:
                    raise
                log_agent_error("Koda session: plan coverage check unavailable", frappe.get_traceback())
                gaps = []
            self.checked = True
            if gaps:
                return ("SUBMIT_FAILED: an independent check of the plan against the request found:\n- "
                        + "\n- ".join(gaps)
                        + "\nThe checker saw only the request and the plan, not the code. Revise the plan where "
                        "it drops what the user asked for; where it already delivers it (for example through "
                        "a control kept from the reference), say so in assumptions and submit it unchanged.")
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
            prompt = get_session_request_prompt(
                state.get("user_message", ""), state.get("request_type", "Improvement"),
                verification.contract_context(verification.prepare_contract(app_name)), request_name=request_name)
            # Ranked from the request alone, so the first model call already knows where to look.
            points = koda_core.starting_points(app_name, state.get("user_message", ""), request_name=request_name)
            history = {"task_prompt": f"{prompt}\n\n{points}" if points else prompt}
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


#: After a repair that did not work, the next one starts from the evidence, not the last patch.
STRATEGY_CHANGE = (
    "\n\n## CHANGE OF REPAIR STRATEGY {level}/{limit}\nThe previous patch did not resolve the failure. "
    "Reproduce it with run_tests, read the exact failing path and its caller/consumer, and identify why the "
    "prior patch was ineffective. Use a concrete input and expected output to choose a different, minimal "
    "correction. Do not churn unrelated files, repeat the previous patch, or suppress the check. Run the "
    "regression and connected tests again before reporting complete."
)
#: The static import check's hint that names the module a broken import should use.
IMPORT_HINT = "; import it as '"
IMPORT_RECOVERY = (
    "\n\n## IMPORT RECOVERY PROCEDURE\nThe static checker has already searched the app and included exact "
    "path:line import references and the existing canonical module in the finding. Edit those current "
    "references now. Also use search_code for the missing dotted prefix to catch other live references. Do "
    "not create a guessed package or __init__.py at the missing path, and do not report complete until "
    "validate_code succeeds for every edited Python file."
)


def _repair_directive(state: dict, history: dict) -> str:
    """The review's findings as the implementer's next message in the session."""
    notes = state["review_notes"]
    text = "## FINDINGS TO REPAIR\n" + notes
    level = int(state.get("repair_strategy_level") or 0)
    if level > int(history.get("strategy_level_sent") or 0):
        # Once per new level: the conversation already holds the earlier ones.
        text += STRATEGY_CHANGE.format(level=level, limit=graph.MAX_REPAIR_STRATEGIES)
        history["strategy_level_sent"] = level
    if IMPORT_HINT in notes:
        text += IMPORT_RECOVERY
    return (text + "\n\nFix these in the current files, run the affected checks again, then return the JSON "
                   "completion report.")


def _conversation(state: dict) -> dict:
    """The history this pass continues: the investigation when fresh, else the implementation.

    A plan without a saved investigation is implemented from the plan alone.
    """
    request_name = state.get("request_name", "")
    # Calls or reviews already made mean a pass saved its conversation.
    fresh = not (state.get("tool_rounds_used") or state.get("review_attempts")
                 or state.get("is_follow_up") or state.get("resuming"))
    history = load(request_name, PLANNING) if fresh else (load(request_name) or load(request_name, PLANNING))
    if history.get("task_prompt"):
        return history
    return {"investigated": False, "task_prompt": get_session_request_prompt(
        state.get("user_message", ""), state.get("request_type", "Improvement"),
        verification.contract_context(state.get("verification_contract") or {}), request_name=request_name)}


def implement_node(state: dict) -> dict:
    """Phase 2, and every repair or follow-up: continue the session with writes enabled."""
    if state.get("error"):
        return {"error": state["error"]}
    request_name = state.get("request_name", "")
    history = _conversation(state)
    _, criteria, allowed = graph._execution_context(state)
    logs = graph._log_stage(state, "Implementing", "started", f"session attempt {state.get('review_attempts', 0) + 1}")
    # The plan's files enter the baseline before any write, so a no-op is still reviewed.
    baseline = dict(state.get("execution_baseline") or {})
    for path in graph._review_paths(state, allowed):
        if path not in baseline:
            baseline[path] = graph._read_current(state, path)
    state = {**state, "execution_baseline": baseline}
    checkpoint.update(execution_baseline=baseline)

    plan = state["plan_object"]
    plan_json = json.dumps(plan, ensure_ascii=False, indent=1)
    follow_up = (state.get("follow_up_message") or "").strip() if state.get("is_follow_up") else ""
    # The worktree at the follow-up's start tells a repeated message apart from a resumed run.
    follow_up_key = f"{state.get('follow_up_worktree_before', '')}\n{follow_up}"
    if follow_up and history.get("follow_up") != follow_up_key:
        if not history.get("approved"):
            graph._queue_directive(history, get_session_implemented_plan_prompt(plan_json))
            history["approved"] = True
        # The user's next message, after the last run's own report, in the same conversation.
        graph._queue_followup(history, state.get("implementation_memory") or "",
                              get_session_follow_up_prompt(follow_up, request_name=request_name), kind="follow_up")
        history["follow_up"] = follow_up_key
    elif not history.get("approved"):
        graph._queue_directive(history, get_session_approval_prompt(
            plan_json, criteria, edited=history.get("proposed_plan") not in (None, plan),
            investigated=history.get("investigated", True), request_name=request_name))
        history["approved"] = True
    elif state.get("review_notes"):
        graph._queue_followup(history, json.dumps(state.get("task_completion") or {}, ensure_ascii=False),
                              _repair_directive(state, history), kind="repair")
    if state.get("resuming"):
        # The interrupted pass may have written more than its saved conversation shows.
        _, applied = graph.change_evidence(graph._without_redacted(state, baseline),
                                           lambda p: graph._read_current(state, p), limit=12000)
        graph._queue_directive(history, "## RESUMED AFTER AN INTERRUPTION\nThese changes are on disk now; "
                                        "continue from them:\n" + (applied or "(no changes yet)"))

    provider = state.get("ai_provider", "OpenAI")
    model = state.get("ai_model", "gpt-4o-mini")
    llm = graph._get_llm(provider=provider, model=model, session_id=request_name)
    # Saved at start, after every round and on interrupt, so a resume never continues an older one.
    work = _path(request_name)
    prior = work.read_bytes() if work.is_file() else None
    session = {"plan_sink": None, "explorer": _explorer(state, llm, provider, request_name),
               "after_round": lambda: save(request_name, history)}
    save(request_name, history)
    try:
        updates = graph._run_agent_turn(state, "Implementing", history["task_prompt"], read_only_tools=False,
                                        max_rounds=graph.MAX_TOOL_ROUNDS_EXECUTION, history=history, session=session)
    except BaseException:
        save(request_name, history)
        raise
    if updates.get("review_stopped"):
        # Roll back the conversation so a follow-up does not inherit an unanswered repair.
        if prior is None:
            work.unlink(missing_ok=True)
        else:
            work.write_bytes(prior)
        return {"review_stopped": updates["review_stopped"],
                "stage_log": graph._log_stage({**state, "stage_log": logs}, "Implementing", "stopped",
                                              updates["review_stopped"][:200])}
    save(request_name, history)
    return {**graph._implementation_updates(state, updates, logs), "resuming": False}


def build_planning_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", END)
    return workflow.compile()


def build_execution_graph():
    """Implement the whole plan in the session, review it, repair in the session."""
    workflow = StateGraph(AgentState)
    workflow.add_node("prepare", graph._checkpointed_node("prepare", graph.prepare_execution_node))
    workflow.add_node("implement", graph._checkpointed_node("implement", implement_node))
    workflow.add_node("review", graph._checkpointed_node("review", graph.review_node))
    workflow.set_conditional_entry_point(lambda s: s.get("resume_node") or "prepare", {
        "prepare": "prepare", "implement": "implement", "review": "review", "done": END,
    })
    workflow.add_edge("prepare", "implement")
    workflow.add_edge("implement", "review")
    workflow.add_conditional_edges("review", graph.should_retry_implement, {
        "review": "review", "implement": "implement", "done": END,
    })
    return workflow.compile()
