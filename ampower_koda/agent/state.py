# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Agent state schema for LangGraph workflow

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """State passed through the agent graph."""

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

    # Bench commands
    bench_log: str
    pending_bench_commands: str

    # Deploy phase
    branch_name: str
    pr_url: str
    pr_number: int

    # Conversation and tool output
    messages: list
    intermediate_steps: list

    # Output
    patch_diff: str

    # Control
    current_stage: str
    error: str
    error_log: str
    tokens_used: int
    stage_log: list  # [{"stage": str, "status": str, "summary": str, "timestamp": str}]
