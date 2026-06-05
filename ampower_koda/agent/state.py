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
    github_token: str
    base_branch: str
    branch_prefix: str
    git_user_name: str
    git_user_email: str

    # Understanding phase
    understanding_summary: str
    explored_paths: list

    # Plan phase
    plan: str
    files_to_modify: list
    files_to_create: list

    # Implementation phase
    implemented_files: list
    edits_made: list  # [{path, summary}, ...]

    # Review phase
    review_passed: bool
    review_notes: str
    review_attempts: int

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
    
