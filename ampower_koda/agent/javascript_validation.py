"""Fast, scope-aware JavaScript diagnostics between edits and runtime tests."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess


def globals_configs(full: str, app_root: str, framework_root: str) -> list[str]:
    """Framework globals, then app data configs from outermost to nearest."""
    app = Path(app_root).resolve()
    directories = [Path(framework_root).resolve().parent]
    current = Path(full).resolve().parent
    project = app.parent if (app.parent / "pyproject.toml").is_file() or (app.parent / ".git").exists() else app
    ancestors = []
    while current == project or project in current.parents:
        ancestors.append(current)
        if current == project:
            break
        current = current.parent
    directories.extend(reversed(ancestors))
    return list(dict.fromkeys(str(directory / name) for directory in directories
                              for name in ("package.json", ".eslintrc", ".eslintrc.json")))


def validate_javascript_names(path: str, source: str, *, configs: list[str], env: dict) -> str:
    try:
        result = subprocess.run(
            ["node", str(Path(__file__).with_name("javascript_check.cjs"))],
            input=json.dumps({"source": source, "configs": configs}), capture_output=True,
            text=True, encoding="utf-8", timeout=10, env=env,
        )
        data = json.loads(result.stdout)
        if result.returncode or data.get("checked") is not True:
            reason = data.get("error") or result.stderr[-500:] or "checker did not complete"
            return f"VALIDATION_UNAVAILABLE: JavaScript scope check: {reason}"
        if data["count"]:
            lines = [f"JAVASCRIPT_ERROR in {path}: {data['count']} scope/parser error(s)."]
            lines.extend(f"{path}:{item['line']}:{item['column']}: {item['message']} ({item['ruleId'] or 'parser'})"
                         for item in data["diagnostics"])
            if data["count"] > len(data["diagnostics"]):
                lines.append("Showing the first 20 errors; repair and revalidate for the rest.")
            lines.append("Restore missing local declarations or imports. For an actual browser global, use its explicit window property.")
            return "\n".join(lines)
        return f"VALID: {path} passed JavaScript syntax and undefined-name checks. Runtime behavior still needs tests."
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        return f"VALIDATION_UNAVAILABLE: JavaScript scope check did not complete: {type(exc).__name__}: {exc}"
