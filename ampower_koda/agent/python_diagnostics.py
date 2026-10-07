"""Type-checker diagnostics for edited Python, reported in the edit receipt.

``validate_code`` only parses Python, so an undefined name, a misspelled helper
or a call with the wrong arguments reached tests and review, each a full
repair pass. pyright finds them in about two seconds per file. Frappe's API is
dynamic (``frappe.db``, ``doc.fieldname``), so library types are not inferred
and only rules that do not depend on them are reported.

Optional: without a ``pyright`` executable on PATH nothing is reported
(install it with ``npm install -g pyright``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

TIMEOUT_SECONDS = 60
MAX_REPORTED = 10
# Rules that hold without knowing Frappe's types.
KEPT_RULES = frozenset({
    "reportUndefinedVariable",   # a name that is never defined
    "reportPossiblyUnbound",     # a name defined on only some paths
    "reportRedeclaration",       # a second def/class hiding the first
    "reportCallIssue",           # wrong arguments to a function pyright can see
    "reportSelfClsParameterName",
})
CONFIG = {
    "typeCheckingMode": "basic",
    "useLibraryCodeForTypes": False,
    "reportMissingModuleSource": False,
}

#: Keyed by file and app-source fingerprint, since pyright also reads imported modules.
_cache: dict[tuple[str, str, str], list[str]] = {}
CACHE_ENTRIES = 256  # a long-lived worker adds one per edit; drop the oldest beyond this
SKIPPED_DIRECTORIES = frozenset({"__pycache__", ".git", "node_modules"})


def executable(env: dict | None = None) -> str | None:
    return shutil.which("pyright", path=(env or os.environ).get("PATH"))


def _kept(diagnostic: dict, package: str) -> bool:
    if diagnostic.get("severity") not in ("error", "warning"):
        return False
    rule, message = diagnostic.get("rule") or "", diagnostic.get("message") or ""
    if rule in KEPT_RULES:
        return True
    own = f'"{package}' if package else None
    if rule == "reportMissingImports":
        # Frappe and ERPNext may be absent from the checker's path; the app's own modules are not.
        return bool(own) and own in message
    if rule == "reportAttributeAccessIssue":
        # A helper that does not exist in one of the app's modules; not dynamic document fields.
        return bool(own) and f"of module {own}" in message
    return False


def _sources_fingerprint(root: str) -> str:
    """Path, size and mtime of every Python source under ``root``; any edit changes it."""
    digest = hashlib.sha256()
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = sorted(name for name in subdirectories if name not in SKIPPED_DIRECTORIES)
        for name in sorted(files):
            if not name.endswith((".py", ".pyi")):
                continue
            path = os.path.join(directory, name)
            try:
                info = os.stat(path)
            except OSError:
                continue
            digest.update(f"{path}\0{info.st_size}\0{info.st_mtime_ns}\n".encode("utf-8", "surrogateescape"))
    return digest.hexdigest()


def diagnostics(app_root: str, full_path: str, package: str, *, env: dict | None = None,
                python: str = sys.executable) -> list[str] | None:
    """Problems in ``full_path`` as ``line N: message``; None when pyright is unavailable or fails."""
    command = executable(env)
    if not command:
        return None
    try:
        with open(full_path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None
    key = (full_path, digest, _sources_fingerprint(app_root))
    if key in _cache:
        return _cache[key]
    with tempfile.TemporaryDirectory() as directory:
        config = os.path.join(directory, "pyrightconfig.json")
        with open(config, "w", encoding="utf-8") as handle:
            json.dump({**CONFIG, "extraPaths": [app_root]}, handle)
        try:
            done = subprocess.run([command, "--outputjson", "--project", config, "--pythonpath", python, full_path],
                                  cwd=app_root, env=env, capture_output=True, text=True,
                                  timeout=TIMEOUT_SECONDS, check=False)
            report = json.loads(done.stdout or "{}")
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    target = os.path.normcase(os.path.abspath(full_path))
    found = set()
    for d in report.get("generalDiagnostics", []):
        line = ((d.get("range") or {}).get("start") or {}).get("line")
        if (isinstance(line, int) and os.path.normcase(os.path.abspath(d.get("file", ""))) == target
                and _kept(d, package)):
            found.add((line + 1, str(d.get("message", "")).replace("\n", " ")))
    lines = [f"line {line}: {message}" for line, message in sorted(found)]
    while len(_cache) >= CACHE_ENTRIES:
        _cache.pop(next(iter(_cache)))
    _cache[key] = lines
    return lines


def receipt(app_root: str, full_path: str, package: str, *, env: dict | None = None) -> str:
    """The block appended to an edit receipt; empty when clean or unavailable."""
    problems = diagnostics(app_root, full_path, package, env=env)
    if not problems:
        return ""
    shown = problems[:MAX_REPORTED]
    more = f"\n... {len(problems) - len(shown)} more" if len(problems) > len(shown) else ""
    return (f"\nType checker (pyright) found {len(problems)} problem(s); fix them before building on this "
            "file:\n" + "\n".join(shown) + more)
