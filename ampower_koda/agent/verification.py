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

# Nothing call_method does may outlive it: writes outside the scratch dir go to an overlay,
# real deletes/renames are refused, jobs and mail are recorded, and an audit hook blocks bypasses.
_CALL_CONTAINMENT = """
import builtins, errno, importlib, io, pathlib, shutil, smtplib, stat
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_real = {name: getattr(os, name) for name in ('open', 'stat', 'lstat', 'mkdir', 'rename', 'replace', 'remove',
         'unlink', 'rmdir', 'listdir', 'scandir', 'access', 'chmod', 'utime', 'truncate') if hasattr(os, name)}
_real_open, _real_rmtree = io.open, shutil.rmtree
_shadowed, _written, _jobs, _mails, _undo = set(), {}, [], [], []
_active, _SCRATCH, _OVERLAY, _DEVNULL = False, None, None, None

def _key(value):
    # absolute, case-folded path; None for a descriptor or a non-path
    if isinstance(value, int):
        return None
    try:
        return os.path.normcase(os.path.abspath(os.fsdecode(os.fspath(value))))
    except (TypeError, ValueError):
        return None

def _outside(key):
    return key is not None and key != _DEVNULL and key != _SCRATCH and not key.startswith(_SCRATCH + os.sep)

def _shadow(key):
    drive, rest = os.path.splitdrive(key)
    return os.path.join(_OVERLAY, drive.replace(':', '').strip(os.sep).replace(os.sep, '_'), rest.lstrip(os.sep))

def _there(path):
    try:
        _real['lstat'](path)
        return True
    except (OSError, ValueError):
        return False

def _is_dir(path):
    try:
        return stat.S_ISDIR(_real['stat'](path).st_mode)
    except (OSError, ValueError):
        return False

def _lives(key):
    # written by this call: the path or a directory above it exists only in the overlay
    while _shadowed and key not in _shadowed:
        parent = os.path.dirname(key)
        if parent == key:
            return False
        key = parent
    return bool(_shadowed)

def _read(value):
    # a path this call wrote is read from its overlay copy
    key = _key(value) if _shadowed else None
    return _shadow(key) if _outside(key) and _lives(key) else value

def _makedirs(path):
    if not _is_dir(path):
        _makedirs(os.path.dirname(path))
        try:
            _real['mkdir'](path)
        except FileExistsError:
            pass

def _write(value, keep=False, exclusive=False):
    # the overlay path that takes a write to a real path; None for the call's own paths
    key = _key(value)
    if not _outside(key):
        return None, key
    target = _shadow(key)
    if not _lives(key):
        name = os.fsdecode(os.fspath(value))
        if exclusive and _there(key):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), name)
        parent = os.path.dirname(key)
        if not (_is_dir(parent) or _is_dir(_shadow(parent))):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name)
        _makedirs(os.path.dirname(target))
        if keep and _there(key):
            with _real_open(key, 'rb') as source, _real_open(target, 'wb') as copy:
                shutil.copyfileobj(source, copy)
    return target, key

def _wrote(key, value):
    _shadowed.add(key)
    _written.setdefault(key, ('directory: ' if _is_dir(_shadow(key)) else 'file: ')
                        + os.path.abspath(os.fsdecode(os.fspath(value))))

def _forget(key):
    _shadowed.difference_update([known for known in _shadowed if known == key or known.startswith(key + os.sep)])

def _open(file, mode='r', *args, **kwargs):
    if _active and isinstance(mode, str) and not isinstance(file, int):
        if any(flag in mode for flag in 'wax+'):
            # append needs no copy of the original; read-write modes do
            target, key = _write(file, '+' in mode and 'w' not in mode, 'x' in mode)
            if target is not None:
                handle = _real_open(target, mode, *args, **kwargs)
                if _there(target) and not _is_dir(target):  # an opener (tempfile's) may open another path
                    _wrote(key, file)
                return handle
        else:
            file = _read(file)
    return _real_open(file, mode, *args, **kwargs)

def _os_open(path, flags, mode=0o777, *, dir_fd=None):
    if _active and dir_fd is None:
        if flags & _WRITE_FLAGS:
            keep = not flags & os.O_TRUNC and bool(flags & os.O_RDWR or not flags & os.O_APPEND)
            target, key = _write(path, keep, bool(flags & os.O_CREAT and flags & os.O_EXCL))
            if target is not None:
                descriptor = _real['open'](target, flags, mode)
                _wrote(key, path)
                return descriptor
        else:
            path = _read(path)
    return _real['open'](path, flags, mode, dir_fd=dir_fd)

def _mkdir(path, mode=0o777, *, dir_fd=None):
    key = _key(path) if _active and dir_fd is None else None
    if not _outside(key):
        return _real['mkdir'](path, mode, dir_fd=dir_fd)
    if _there(_shadow(key) if _lives(key) else key):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), os.fsdecode(os.fspath(path)))
    target, key = _write(path)
    _real['mkdir'](target, mode)
    _wrote(key, path)

def _routed(function):
    def call(*args, **kwargs):
        if _active and args and kwargs.get('dir_fd') is None:
            args = (_read(args[0]),) + args[1:]
        return function(*args, **kwargs)
    return call

def _deleting(function):
    # only what this call wrote can go; a real file reaches the audit hook
    def call(path, *, dir_fd=None):
        key = _key(path) if _active and dir_fd is None else None
        if _outside(key) and _lives(key) and not _there(key):
            function(_shadow(key))
            _forget(key)
            return None
        return function(path, dir_fd=dir_fd)
    return call

def _renaming(function):
    def call(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        key = _key(src) if _active and src_dir_fd is None and dst_dir_fd is None else None
        moved = _outside(key) and _lives(key) and not _there(key)
        if key is None or not moved and _outside(key):
            return function(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        target, written = _write(dst)
        function(_shadow(key) if moved else src, dst if target is None else target)
        if moved:
            _forget(key)
        if target is not None:
            _wrote(written, dst)
    return call

def _rmtree(path, *args, **kwargs):
    key = _key(path) if _active and kwargs.get('dir_fd') is None else None
    if _outside(key) and _lives(key) and not _there(key):
        _real_rmtree(_shadow(key), *args, **kwargs)
        _forget(key)
        return None
    return _real_rmtree(path, *args, **kwargs)
_rmtree.avoids_symlink_attacks = _real_rmtree.avoids_symlink_attacks

def _copying(function):
    # a native file copy (shutil.copy2 on Windows) writes like open()
    def call(src, dst, *args, **kwargs):
        target, key = _write(dst) if _active else (None, None)
        result = function(_read(src) if _active else src, dst if target is None else target, *args, **kwargs)
        if target is not None:
            _wrote(key, dst)
        return result
    return call

"""


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
