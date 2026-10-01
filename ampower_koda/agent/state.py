# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Agent state schema for LangGraph workflow

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """
    State passed through the agent graph.
    All fields are optional (total=False) — each graph node returns only
    the fields it updates; LangGraph merges them into the running state.
    """

    # User input
    user_message: str
    request_type: str
    request_name: str

    # Per-request configuration
    target_app_name: str
    ai_provider: str
    ai_model: str
    github_repo_url: str
    base_branch: str
    branch_prefix: str
    git_user_name: str
    git_user_email: str

    # Plan phase
    understanding_summary: str  # the investigation's findings, submitted with the plan
    plan: str
    plan_object: dict
    plan_feedback: str  # the user's answer to a proposed plan; planning continues its session with it

    # Implementation phase
    edits_made: list  # [{path, summary}, ...]
    change_summary: str  # plain-English summary of what the implement phase changed
    execution_tasks: list[dict]
    task_results: list[dict]
    execution_baseline: dict  # path -> content before this execution wrote it; review diffs against it
    file_moves: list[dict]  # verified source/destination pairs reported by the rename tool
    copied_files: dict  # destination -> reference for copy_file copies; rewritten only by edits
    called_methods: list[str]  # dotted paths call_method ran in this execution; review lists changed ones never run
    test_repair_rounds: int  # repairs failing tests have triggered; past the limit they are reported
    task_completion: dict
    task_summary: str
    tool_rounds_used: int
    tool_rounds_limit: int
    automatic_repair_budget_enabled: bool
    automatic_repair_budget_grants: int  # bounded continuations earned by executed test progress
    verification_progress: dict
    turn_exhausted: bool
    resume_node: str
    resuming: bool
    prior_file_moves: list[dict]
    prior_deleted_paths: list[str]
    follow_up_worktree_before: str

    # Follow-up patch mode
    is_follow_up: bool
    follow_up_message: str
    prior_changed_paths: list
    implementation_memory: str

    # Review/Test phase
    review_passed: bool
    review_notes: str
    # Why repair stopped short of a passing review: the work is delivered with this and the last
    # findings as warnings, never failed. Only a crash or an unloadable plan ends a run in "error".
    review_stopped: str
    tests_unresolved: str  # tests still failing after the repair cap: delivered as a warning
    review_attempts: int
    review_fingerprint: str  # changed-file state at the last failed review; unchanged = no progress
    review_fingerprints_seen: list[str]  # detects A→B→A repair oscillation before the hard valve
    review_failure_fingerprints_seen: list[str]  # repeated deterministic check failure despite unrelated edits
    review_repairable: bool  # failed review is owned by implementation; resume there instead of re-reviewing
    review_retry_requested: bool  # review snapshot/provider state changed; retry review rather than edit source
    review_rechecks: int  # bounded review-only retries that do not consume implementation repair attempts
    review_history: dict  # active reviewer rounds/source snapshot retained across repair rechecks
    review_format_retries: int  # fresh read-only attempts after retained verdict repair fails
    repair_strategy_level: int  # bounded fresh diagnoses after ineffective code repairs
    verification_contract: dict  # commands and pre-existing test hashes frozen before edits
    verification_receipts: list[dict]  # actual exit codes/output from the latest review gate
    plan_amendments: int  # plan patch rounds spent on implementation blockers this run
    plan_last_blocker: str  # the blocker the last amendment answered; the same one again is not progress
    plan_scope_repairs: list[str]  # file inventory completed from explicit, already-approved renames

    # Pending approvals
    pending_bench_commands: str # JSON-encoded list[str] — always parse with json.loads before use

    # Deploy phase
    branch_name: str
    pr_url: str
    pr_number: int| None

    # Conversation and tool output
    messages: list
    intermediate_steps: list # [{"phase": str, "output": str}] — one entry per agent phase

    # Output
    patch_diff: str
    bench_log: str
    files_changed: str

    # Control
    current_stage: str
    error: str  # set by any node on failure; checked by subsequent nodes to short-circuit
    error_log: str  # full traceback, written to DB but not used for graph flow control
    tokens_used: int
    cost_estimate: float  # estimated USD cost based on tokens_used and model pricing
    stage_log: list[dict]  # [{"stage": str, "status": str, "summary": str, "timestamp": str}]
