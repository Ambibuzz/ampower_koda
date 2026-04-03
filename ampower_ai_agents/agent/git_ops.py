# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Git and GitHub operations for the agent — all functions accept explicit parameters

import os
import re
import subprocess

import frappe


def get_repo_root(app_name: str) -> str:
    """Return the git repository root for the given app."""
    if not app_name:
        frappe.throw("Target App Name is required")
    app_path = frappe.get_app_path(app_name)
    return os.path.dirname(app_path)


def run_git(cmd: list[str], cwd: str | None = None) -> tuple[bool, str]:
    """Run a git command. Returns (success, output)."""
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


def _parse_github_repo(url: str) -> tuple[str, str] | None:
    """Parse GitHub URL to owner and repo. Returns (owner, repo) or None."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        return m.group(1), m.group(2).rstrip("/")
    m = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)
    return None


def configure_git_identity(
    app_name: str,
    git_user_name: str = "AI Agent",
    git_user_email: str = "ai-agent@ampower.com",
) -> tuple[bool, str]:
    """Set git user.name and user.email at repo-level config."""
    name = (git_user_name or "AI Agent").strip()
    email = (git_user_email or "ai-agent@ampower.com").strip()
    root = get_repo_root(app_name)
    ok1, out1 = run_git(["config", "user.name", name], cwd=root)
    ok2, out2 = run_git(["config", "user.email", email], cwd=root)
    if not ok1 or not ok2:
        return False, f"git config failed: {out1} {out2}"
    return True, f"{name} <{email}>"


def create_branch(app_name: str, branch_name: str, base_branch: str = "main") -> tuple[bool, str]:
    """Create and checkout a new branch from base_branch."""
    root = get_repo_root(app_name)
    run_git(["fetch", "origin", base_branch], cwd=root)
    ok, out = run_git(["checkout", "-b", branch_name, base_branch], cwd=root)
    if not ok:
        ok, out = run_git(["checkout", "-b", branch_name], cwd=root)
    return ok, out


def commit_changes(app_name: str, message: str, git_user_name: str = "AI Agent", git_user_email: str = "ai-agent@ampower.com") -> tuple[bool, str]:
    """Stage all changes and commit."""
    root = get_repo_root(app_name)

    ok, out = configure_git_identity(app_name, git_user_name, git_user_email)
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


def push_branch(app_name: str, branch_name: str, repo_url: str, token: str) -> tuple[bool, str]:
    """Push branch to origin using explicit credentials from the request."""
    if not repo_url:
        return False, "GitHub URL not provided"
    
    clean_token = (token or "").strip()
    
    if not clean_token:
        return False, "GitHub token not provided in the AI Agent Request"

    # Safety check: if it looks like a URL, it's definitely not a token
    if clean_token.startswith("http"):
        # Help the user by pointing out the likely mistake
        return False, (
            f"Invalid GitHub token: The token field seems to contain a URL ('{clean_token[:30]}...'). "
            "Please ensure you enter a valid Personal Access Token (PAT) in the 'GitHub Token' field."
        )

    parsed = _parse_github_repo(repo_url)
    if not parsed:
        return False, f"Invalid GitHub URL: {repo_url}"
    owner, repo = parsed
    
    # Construct authenticated remote URL
    remote = f"https://{clean_token}@github.com/{owner}/{repo}.git"
    root = get_repo_root(app_name)
    ok, out = run_git(["push", remote, branch_name], cwd=root)
    return ok, out


def create_pull_request(
    title: str,
    body: str,
    head_branch: str,
    repo_url: str,
    token: str,
    base_branch: str = "main",
) -> tuple[bool, str, str | None, int | None]:
    """Create a PR via GitHub API. Returns (success, message, pr_url, pr_number)."""
    import requests as http_requests

    if not token or not repo_url:
        return False, "GitHub token or repo URL not provided", None, None
    parsed = _parse_github_repo(repo_url)
    if not parsed:
        return False, f"Invalid GitHub URL: {repo_url}", None, None
    owner, repo = parsed
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    payload = {
        "title": title,
        "body": body,
        "head": head_branch,
        "base": base_branch,
    }
    try:
        resp = http_requests.post(api_url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        if resp.status_code == 201:
            pr_url = data.get("html_url")
            pr_number = data.get("number")
            return True, "PR created", pr_url, pr_number
        msg = data.get("message", resp.text)
        return False, msg, None, None
    except Exception as e:
        return False, str(e), None, None


def generate_branch_name(request_name: str, branch_prefix: str = "ai-agent/", app_name: str = "") -> str:
    """Generate a safe, unique branch name from request name.
    When app_name is provided, checks existing local branches and appends
    _v0, _v1, ... if the base name is already taken."""
    prefix = (branch_prefix or "ai-agent/").strip()
    safe = re.sub(r"[^a-zA-Z0-9-]", "-", request_name).strip("-")
    base_name = f"{prefix}{safe}"

    if not app_name:
        return base_name

    try:
        root = get_repo_root(app_name)
    except Exception:
        return base_name

    ok, branches = run_git(["branch", "--list"], cwd=root)
    if not ok:
        return base_name

    existing = {b.strip().lstrip("* ") for b in branches.splitlines()}
    if base_name not in existing:
        return base_name

    for v in range(100):
        candidate = f"{base_name}_v{v}"
        if candidate not in existing:
            return candidate

    return base_name


def get_current_branch(app_name: str) -> str:
    """Return the name of the currently checked-out branch, or empty string."""
    root = get_repo_root(app_name)
    ok, out = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    return out.strip() if ok else ""


def checkout_base(app_name: str, base_branch: str = "main") -> tuple[bool, str]:
    """Discard uncommitted changes and checkout the base branch.
    This is the manual checkout function — only called explicitly by the user."""
    root = get_repo_root(app_name)

    run_git(["reset", "--hard", "HEAD"], cwd=root)
    run_git(["clean", "-fd"], cwd=root)

    current = get_current_branch(app_name)
    if current == base_branch:
        return True, f"Already on {base_branch}"

    ok, out = run_git(["checkout", base_branch], cwd=root)
    if not ok:
        return False, f"checkout {base_branch} failed: {out}"
    return True, f"Checked out {base_branch}"
