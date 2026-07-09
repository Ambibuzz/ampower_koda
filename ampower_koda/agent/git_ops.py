# Copyright (c) 2026, Ambibuzz Technologies LLP and contributors
# Git and GitHub operations for the agent — all functions accept explicit parameters

import os
import re
import subprocess
import requests as http_requests

import frappe

from ampower_koda.agent.errors import log_agent_error


def get_repo_root(app_name: str) -> str:
    """
    Identifies the root directory of the Git repository for a given app. 
    This is the base path for all Git operations.
    """
    if not app_name:
        frappe.throw("Target App Name is required")
    app_path = frappe.get_app_path(app_name)
    return os.path.dirname(app_path)


def run_git(cmd: list[str], cwd: str | None = None) -> tuple[bool, str]:
    """
    Executes a Git command in the specified directory. 
    Returns a success flag and the combined output of the command.
    """
    try:
        result = subprocess.run(
            ["git"] + cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = ((result.stdout or "").strip() + "\n" + (result.stderr or "").strip()).strip()
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        log_agent_error("Agent Git: command timeout", f"cmd={' '.join(cmd)}\ncwd={cwd}")
        return False, "Git command timed out"
    except Exception as e:
        log_agent_error(
            "Agent Git: command failed",
            f"cmd={' '.join(cmd)}\ncwd={cwd}\n{e}\n{frappe.get_traceback()}",
        )
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
    """
    Configures the local Git identity for the repository. 
    This ensures that all commits made by the agent are clearly attributed.
    """
    name = (git_user_name or "AI Agent").strip()
    email = (git_user_email or "ai-agent@ampower.com").strip()
    root = get_repo_root(app_name)
    ok1, out1 = run_git(["config", "user.name", name], cwd=root)
    ok2, out2 = run_git(["config", "user.email", email], cwd=root)
    if not ok1 or not ok2:
        return False, f"git config failed: {out1} {out2}"
    return True, f"{name} <{email}>"


def create_branch(app_name: str, branch_name: str, base_branch: str = "main") -> tuple[bool, str]:
    """
    Creates and switches to a new working branch starting from the base branch.
    This provides a safe sandbox for the agent's changes.
    """
    root = get_repo_root(app_name)
    run_git(["fetch", "origin", base_branch], cwd=root)
    ok, out = run_git(["checkout", "-b", branch_name, base_branch], cwd=root)
    if not ok:
        return False, f"Failed to create branch '{branch_name}' from '{base_branch}': {out}"
    return ok, out


def commit_changes(app_name: str, message: str, git_user_name: str = "AI Agent", git_user_email: str = "ai-agent@ampower.com") -> tuple[bool, str]:
    """
    Stages all modified and new files and records them in a new commit.
    This finalizes the implementation phase on the local machine.
    """
    root = get_repo_root(app_name)

    ok, out = configure_git_identity(app_name, git_user_name, git_user_email)
    if not ok:
        return False, out

    ok, out = run_git(["add", "-A"], cwd=root)
    if not ok:
        return False, out
    ok, out = run_git(["status", "--short"], cwd=root)
    if not ok:
        return False, f"git status failed: {out}"
    if not out.strip():
        return False, "No changes to commit"
    ok, out = run_git(["commit", "-m", message], cwd=root)
    return ok, out


def push_branch(app_name: str, branch_name: str, repo_url: str, token: str) -> tuple[bool, str]:
    """
    Uploads the local branch to the remote GitHub repository
    using the provided credentials. The token is scrubbed from
    any returned output to prevent it from leaking into logs or
    realtime events.
    """
    if not repo_url:
        return False, "GitHub URL not provided"
    
    clean_token = (token or "").strip()
    
    if not clean_token:
        return False, "GitHub token not provided in the Agent Request"

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
    safe_out = out.replace(clean_token, "***")
    return ok, safe_out


def create_pull_request(
    title: str,
    body: str,
    head_branch: str,
    repo_url: str,
    token: str,
    base_branch: str = "main",
) -> tuple[bool, str, str | None, int | None]:
    """
    Uses the GitHub API to create a new Pull Request for the pushed branch. 
    This is the final step in the agent's collaborative workflow.
    """

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
        log_agent_error(
            "Agent Git: create pull request",
            f"repo={repo_url}\n{e}\n{frappe.get_traceback()}",
        )
        return False, str(e), None, None


def generate_branch_name(request_name: str, branch_prefix: str = "ai-agent/", app_name: str = "") -> str:
    """
    Generates a unique and safe Git branch name based on the request.
    If the name is already taken, it appends a version suffix to avoid conflicts.
    """
    prefix = (branch_prefix or "ai-agent/").strip()
    safe = re.sub(r"[^a-zA-Z0-9-]", "-", request_name).strip("-")
    base_name = f"{prefix}{safe}"

    if not app_name:
        return base_name

    try:
        root = get_repo_root(app_name)
    except Exception as e:
        log_agent_error(
            "Agent Git: generate branch name",
            f"app={app_name}\n{e}\n{frappe.get_traceback()}",
        )
        return base_name

    ok, branches = run_git(["branch", "--list", "--all"], cwd=root)
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
    """
    Identifies the name of the branch that is currently active in the repository.
    """
    root = get_repo_root(app_name)
    ok, out = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    return out.strip() if ok else ""


def run_git_stdout(cmd: list[str], cwd: str | None = None) -> tuple[bool, str]:
    """Run git and return stdout only (stderr excluded — suitable for diff parsing)."""
    try:
        result = subprocess.run(
            ["git"] + cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0, (result.stdout or "")
    except subprocess.TimeoutExpired:
        log_agent_error("Agent Git: stdout command timeout", f"cmd={' '.join(cmd)}\ncwd={cwd}")
        return False, ""
    except Exception as e:
        log_agent_error(
            "Agent Git: stdout command failed",
            f"cmd={' '.join(cmd)}\ncwd={cwd}\n{e}\n{frappe.get_traceback()}",
        )
        return False, ""


def branch_exists(app_name: str, branch_name: str) -> bool:
    """Return True if the given branch or ref exists in the repo."""
    if not branch_name:
        return False
    root = get_repo_root(app_name)
    ok, out = run_git_stdout(["rev-parse", "--verify", branch_name], cwd=root)
    return ok and bool(out.strip())


def list_changed_files(
    app_name: str, base_branch: str, branch_name: str
) -> tuple[bool, list[dict]]:
    """
    List files changed between base_branch and branch_name.
    Returns ([{"status": "M|A|D|R", "path": "relative/path"}, ...]).
    """
    if not branch_name:
        return False, []
    root = get_repo_root(app_name)
    base = (base_branch or "main").strip()
    ok, out = run_git_stdout(
        ["diff", "--name-status", base, branch_name],
        cwd=root,
    )
    if not ok:
        return False, []

    files = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0].upper()
        path = parts[-1].strip()
        if path:
            files.append({"status": status, "path": path})
    return True, files


def diff_file(
    app_name: str, base_branch: str, branch_name: str, file_path: str
) -> tuple[bool, str]:
    """Return unified diff for a single file between base and branch (untruncated)."""
    if not branch_name or not file_path:
        return False, ""
    root = get_repo_root(app_name)
    base = (base_branch or "main").strip()
    ok, out = run_git_stdout(
        ["diff", base, branch_name, "--", file_path],
        cwd=root,
    )
    return ok, out


def checkout_base(app_name: str, base_branch: str = "main") -> tuple[bool, str]:
    """
    Discards all uncommitted changes and returns the repository to its 
    base branch. This is the primary 'reset' tool for the user.
    Also pulls the latest changes from remote after switching branches.
    """
    root = get_repo_root(app_name)

    run_git(["reset", "--hard", "HEAD"], cwd=root)
    run_git(["clean", "-fdx"], cwd=root)

    current = get_current_branch(app_name)
    if current == base_branch:
        run_git(["pull", "origin", base_branch], cwd=root)  
        return True, f"Already on {base_branch}, pulled latest"

    ok, out = run_git(["checkout", base_branch], cwd=root)
    if not ok:
        return False, f"checkout {base_branch} failed: {out}"

    run_git(["pull", "origin", base_branch], cwd=root)
    return True, f"Checked out {base_branch}"
