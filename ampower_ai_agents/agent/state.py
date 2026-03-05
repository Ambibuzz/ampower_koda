# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Agent state schema for LangGraph workflow

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """State passed through the agent graph."""

    # User input
    user_message: str
    request_type: str
    request_name: str

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

    # Deploy phase (filled by executor after graph)
    branch_name: str
    pr_url: str
    pr_number: int

    # Conversation and tool output
    messages: list
    intermediate_steps: list

    # Model selection
    ai_model: str

    # Control
    current_stage: str
    error: str
    tokens_used: int
    stage_log: list  # [{"stage": str, "status": str, "summary": str, "timestamp": str}]
