# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Git and GitHub operations for the agent

import os
import re
import subprocess

import frappe
import requests


def _get_settings():
    """Return AI Agents Settings singleton."""
    return frappe.get_single("AI Agents Settings")


def get_repo_root() -> str:
    """Return the git repository root for the configured target app."""
    settings = _get_settings()
    app_name = (settings.target_app_name or "").strip()
    if not app_name:
        frappe.throw("Target App Name not set in AI Agents Settings")
    app_path = frappe.get_app_path(app_name)
    return os.path.dirname(app_path)


def run_git(cmd: list[str], cwd: str | None = None) -> tuple[bool, str]:
    """Run a git command. Returns (success, output)."""
    cwd = cwd or get_repo_root()
    try:
        result = subprocess.run(
            ["git"] + cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (result.stdout or "").strip() + (result.stderr or "").strip()
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "Git command timed out"
    except Exception as e:
        return False, str(e)


def get_base_branch() -> str:
    """Get base branch from settings or default to main."""
    try:
        settings = _get_settings()
        return (settings.base_branch or "main").strip()
    except Exception:
        return "main"


def get_branch_prefix() -> str:
    """Get branch prefix from settings."""
    try:
        settings = _get_settings()
        return (settings.agent_branch_prefix or "ai-agent/").strip()
    except Exception:
        return "ai-agent/"


def create_branch(branch_name: str) -> tuple[bool, str]:
    """Create and checkout a new branch from base_branch."""
    base = get_base_branch()
    ok, out = run_git(["fetch", "origin", base])
    if not ok:
        pass
    ok, out = run_git(["checkout", "-b", branch_name, base])
    if not ok:
        ok, out = run_git(["checkout", "-b", branch_name])
    return ok, out


def get_github_token() -> str:
    settings = _get_settings()
    return (settings.get_password("github_token") or "").strip()


def get_github_repo_url() -> str:
    settings = _get_settings()
    return (settings.github_repo_url or "").strip()


def _parse_github_repo(url: str) -> tuple[str, str] | None:
    """Parse GitHub URL to owner and repo. Returns (owner, repo) or None."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        return m.group(1), m.group(2).rstrip("/")
    m = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)
    return None


def _configure_git_identity() -> tuple[bool, str]:
    """Set git user.name and user.email from settings (repo-level config)."""
    try:
        settings = _get_settings()
        name = (settings.git_user_name or "").strip() or "AI Agent"
        email = (settings.git_user_email or "").strip() or "ai-agent@ampower.com"
    except Exception:
        name = "AI Agent"
        email = "ai-agent@ampower.com"

    root = get_repo_root()
    ok1, out1 = run_git(["config", "user.name", name], cwd=root)
    ok2, out2 = run_git(["config", "user.email", email], cwd=root)
    if not ok1 or not ok2:
        return False, f"git config failed: {out1} {out2}"
    return True, f"{name} <{email}>"


def commit_changes(message: str) -> tuple[bool, str]:
    """Stage all changes and commit."""
    root = get_repo_root()

    ok, out = _configure_git_identity()
    if not ok:
        return False, out

    ok, out = run_git(["add", "-A"], cwd=root)
    if not ok:
        return False, out
    ok, out = run_git(["status", "--short"], cwd=root)
    if not out.strip():
        return False, "No changes to commit"
    ok, out = run_git(["commit", "-m", message], cwd=root)
    return ok, out


def push_branch(branch_name: str) -> tuple[bool, str]:
    """Push branch to origin."""
    token = get_github_token()
    url = get_github_repo_url()
    if not url or not token:
        return False, "GitHub URL or token not configured"
    parsed = _parse_github_repo(url)
    if not parsed:
        return False, f"Invalid GitHub URL: {url}"
    owner, repo = parsed
    remote = f"https://{token}@github.com/{owner}/{repo}.git"
    ok, out = run_git(["push", remote, branch_name])
    return ok, out


def create_pull_request(
    title: str,
    body: str,
    head_branch: str,
) -> tuple[bool, str, str | None, int | None]:
    """Create a PR via GitHub API. Returns (success, message, pr_url, pr_number)."""
    token = get_github_token()
    url = get_github_repo_url()
    if not token or not url:
        return False, "GitHub token or repo URL not set", None, None
    parsed = _parse_github_repo(url)
    if not parsed:
        return False, f"Invalid GitHub URL: {url}", None, None
    owner, repo = parsed
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    base_branch = get_base_branch()
    payload = {
        "title": title,
        "body": body,
        "head": head_branch,
        "base": base_branch,
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        if resp.status_code == 201:
            pr_url = data.get("html_url")
            pr_number = data.get("number")
            return True, "PR created", pr_url, pr_number
        msg = data.get("message", resp.text)
        return False, msg, None, None
    except Exception as e:
        return False, str(e), None, None


def generate_branch_name(request_name: str, request_type: str) -> str:
    """Generate a safe branch name from request name and type."""
    prefix = get_branch_prefix()
    safe = re.sub(r"[^a-zA-Z0-9-]", "-", request_name).strip("-")
    return f"{prefix}{safe}"


def get_current_branch() -> str:
    """Return the name of the currently checked-out branch, or empty string."""
    ok, out = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return out.strip() if ok else ""


def cleanup_and_checkout_base() -> tuple[bool, str]:
    """Discard uncommitted changes and checkout the base branch.

    Safe to call in any state. Returns (success, message).
    """
    base = get_base_branch()
    root = get_repo_root()

    run_git(["reset", "--hard", "HEAD"], cwd=root)
    run_git(["clean", "-fd"], cwd=root)

    current = get_current_branch()
    if current == base:
        return True, f"Already on {base}"

    ok, out = run_git(["checkout", base], cwd=root)
    if not ok:
        return False, f"checkout {base} failed: {out}"
    return True, f"Checked out {base}"
