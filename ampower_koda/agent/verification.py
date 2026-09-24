"""Execute behavioral checks and return evidence to the implementation/review loop.

An app can configure its native integration commands in .koda/verification.json.
Without configuration, portable unittest and Node tests in .koda/tests are run.
Those tests should import the real changed code and replace only external I/O.
Commands are frozen before implementation and are never inferred from model prose.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from .run_control import check_active
from .tools import command_environment

CONFIG_PATH = ".koda/verification.json"
TEST_DIRECTORY = ".koda/tests"
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 300
MAX_OUTPUT = 8000

def test_summary(output: str) -> dict | None:
    """Read a checked runner summary; it never overrides a failing exit code."""
    for line in reversed(output.splitlines()):
        if not line.startswith('KODA_TEST_SUMMARY '):
            continue
        try:
            value = json.loads(line[len('KODA_TEST_SUMMARY '):])
            if not isinstance(value, dict): return None
            for key in ('tests', 'failed', 'skipped'):
                items = value.get(key)
                if not isinstance(items, list) or not all(isinstance(item, str) and item for item in items): return None
                if len(set(items)) != len(items): return None
            total = value.get('total')
            failed, skipped, tests = map(set, (value['failed'], value['skipped'], value['tests']))
            if (type(total) is not int or total <= 0 or total != len(tests)
                    or not (failed | skipped) <= tests or failed & skipped
                    or type(value.get('passed')) is not int
                    or value['passed'] != total - len(failed) - len(skipped)):
                return None
            return {**value, 'tests': sorted(tests), 'failed': sorted(failed), 'skipped': sorted(skipped)}
        except (ValueError, TypeError):
            return None
    return None


def _flat_node_summary(output: str) -> dict | None:
    totals = re.findall(r'^# tests (\d+)\s*$', output, re.M)
    cases = re.findall(r'^(ok|not ok) (\d+) - (.+)$', output, re.M)
    if not totals or int(totals[-1]) != len(cases) or not cases:
        return None  # nested TAP needs a reporter-provided structured summary
    tests, failed, skipped = [], [], []
    for status, number, label in cases:
        pending = bool(re.search(r' # (?:SKIP|TODO)\b', label, re.I))
        name = number + ':' + re.split(r' # (?:SKIP|TODO)\b', label, flags=re.I)[0]
        tests.append(name)
        if pending: skipped.append(name)
        elif status == 'not ok': failed.append(name)
    return {'tests': sorted(tests), 'total': len(tests), 'failed': sorted(failed),
            'skipped': sorted(skipped), 'passed': len(tests) - len(failed) - len(skipped)}


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Verification path escapes the app: {relative}")
    return path


def test_files(root: Path) -> list[Path]:
    directory = _inside(root, TEST_DIRECTORY)
    if not directory.is_dir():
        return []
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and (
        (p.name.startswith("test") and p.suffix == ".py")
        or p.name.endswith((".test.js", ".test.cjs", ".test.mjs"))
    ))
    for path in paths:
        _inside(root, path.relative_to(root).as_posix())
    return paths



def _stop_process(proc) -> None:
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=10)
    proc.wait(timeout=10)


def _output(log) -> str:
    size = log.tell()
    log.seek(0)
    if size <= MAX_OUTPUT:
        data = log.read()
    else:
        head = log.read(2000)
        log.seek(-6000, os.SEEK_END)
        data = head + b"\n... output truncated; failure tail follows ...\n" + log.read()
    return data.decode("utf-8", errors="replace")


def _execute(argv: list[str], cwd: Path, env: dict, timeout: int) -> tuple[int, str, bool]:
    """Bound wall time/output; cancellation also terminates the test process tree."""
    check_active(reserve=timeout + 5)
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=os.name == "posix")
        deadline = time.monotonic() + timeout
        timed_out = False
        try:
            while proc.poll() is None:
                check_active(max_age=1)
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                try:
                    proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Also kill children left behind by a completed runner on POSIX.
            _stop_process(proc)
        return proc.returncode, _output(log), timed_out


def _node_executed_tests(output: str, files: list[Path], root: Path) -> bool:
    """Node counts an empty file as a passing test; require a named test case."""
    if not re.search(r"^# pass [1-9][0-9]*\s*$", output, re.M):
        return False
    normalize = lambda value: re.sub(r"[\\/]+", "/", value).removeprefix("./").casefold()
    filenames = {normalize(name) for path in files
                 for name in (str(path), path.relative_to(root).as_posix())}
    for line in output.splitlines():
        match = re.match(r"\s*ok \d+ - (.+)", line)
        if not match or re.search(r" # (?:SKIP|TODO)\b", line, re.I):
            continue
        name = normalize(match.group(1))
        if name not in filenames:
            return True
    return False


def site_environment() -> dict:
    """The site this worker serves, for runners that connect to it; empty outside a site."""
    try:
        import frappe
        local = getattr(frappe, "local", None)
        site, sites_path = getattr(local, "site", None), getattr(local, "sites_path", None)
    except (ImportError, RuntimeError):
        return {}
    if not isinstance(site, str) or not site or not isinstance(sites_path, str) or not sites_path:
        return {}
    return {"KODA_SITE": site, "KODA_SITES_PATH": os.path.abspath(sites_path)}


def _runner_environment(root: Path, env: dict | None) -> dict:
    command_env = dict(env if env is not None else command_environment())
    for key in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "GH_TOKEN", "GITHUB_TOKEN"):
        command_env.pop(key, None)
    command_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root.parent), str(root),
                                                          command_env.get("PYTHONPATH", "")]))
    command_env["PYTHONDONTWRITEBYTECODE"] = "1"
    command_env["KODA_APP_PARENT"] = str(root.parent)
    command_env["KODA_TEST_PYTHON"] = sys.executable
    for key, value in site_environment().items():
        command_env.setdefault(key, value)
    return command_env


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit // 3] + f"\n... {len(text) - limit} chars omitted ...\n" + text[-(limit - limit // 3):]
