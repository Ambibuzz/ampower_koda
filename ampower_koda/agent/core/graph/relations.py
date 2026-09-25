"""Explicit framework and test relationships absent from ordinary call tags."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath

from ampower_koda.agent.frappe_rpc import call_options

from ..contracts.repository import RepositoryIndex

_FEATURES = {"doctype", "page", "report", "print_format"}
_EXTENSIONS = {".py", ".js", ".ts", ".json", ".html", ".css"}
_FORM = re.compile(r'''frappe\.ui\.form\.on\s*\(\s*["']([^"']+)["']''')
_ROUTE = re.compile(r'''frappe\.set_route\s*\(\s*["']([^"']+)["']''')
_HOOK_SCRIPT = re.compile(r'''["']([^"']+)["']\s*:\s*["']([^"']+\.(?:js|css))["']''')


@dataclass(frozen=True, slots=True)
class Relation:
    source: str
    target: str
    symbol: str
    kind: str


def _name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _test_stem(stem: str) -> str:
    for prefix in ("test_", "spec_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    for suffix in ("_test", "_spec", ".test", ".spec"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return ""


def framework_relations(index: RepositoryIndex) -> tuple[Relation, ...]:
    """Resolve only existing, unambiguous targets; never invent a file path."""
    groups: dict[str, list[str]] = defaultdict(list)
    features: dict[tuple[str, str], list[str]] = defaultdict(list)
    modules: dict[str, list[str]] = defaultdict(list)
    implementations: dict[tuple[str, str], list[str]] = defaultdict(list)
    scripts: dict[str, list[str]] = defaultdict(list)
    paths = index.paths
    found: set[Relation] = set()

    def connect(source: str, target: str, symbol: str, kind: str) -> None:
        if source != target:
            found.add(Relation(source, target, symbol, kind))

    for path in paths:
        file = PurePosixPath(path)
        if file.suffix in {".py", ".js", ".ts"} and not _test_stem(file.stem):
            implementations[(file.stem, file.suffix)].append(path)
        if file.suffix in {".js", ".css"}:
            scripts[_name(file.stem)].append(path)
        if file.suffix == ".py":
            module = str(file.with_suffix("")).replace("/", ".")
            modules[module].append(path)
        parts = file.parts
        if len(parts) < 3 or parts[-3] not in _FEATURES or file.suffix not in _EXTENSIONS:
            continue
        stem = _test_stem(file.stem) or file.stem
        if stem != file.parent.name:
            continue
        groups[str(file.parent)].append(path)
        features[(parts[-3], _name(file.parent.name))].append(path)

    for group, members in groups.items():
        for i, source in enumerate(members):
            for target in members[i + 1:]:
                connect(source, target, PurePosixPath(group).name, "feature")

    for path in paths:
        file = PurePosixPath(path)
        test_stem = _test_stem(file.stem)
        if test_stem:
            candidates = implementations.get((test_stem, file.suffix), [])
            local = [p for p in candidates if PurePosixPath(p).parent == file.parent]
            candidates = local or candidates
            if len(candidates) == 1:
                connect(path, candidates[0], test_stem, "test")

        if file.suffix not in {".js", ".ts", ".py"}:
            continue
        # Chunks cover the indexed source, including uncovered registrations.
        text = "\n".join(chunk.body for chunk in index.files[path].chunks)
        methods = {options.get("method") for options in call_options(text) if options.get("method")}
        for name in methods:
            module, _, symbol = name.rpartition(".")
            targets = modules.get(module, [])
            if not targets:
                targets = [p for m, ps in modules.items()
                           if module.endswith("." + m) or m.endswith("." + module)
                           for p in ps]
            if len(targets) == 1:
                connect(path, targets[0], symbol, "rpc")

        for label in _FORM.findall(text):
            for target in features.get(("doctype", _name(label)), ()):
                connect(path, target, label, "feature")
        for label in _ROUTE.findall(text):
            for target in features.get(("page", _name(label)), ()):
                connect(path, target, label, "route")

        if file.name == "hooks.py":
            for label, registered in _HOOK_SCRIPT.findall(text):
                targets = [p for p in paths if p == registered or p.endswith("/" + registered)]
                # A broken filename registration is exactly when the hook and
                # the existing script need to be retrieved together.
                if not targets:
                    candidates = scripts.get(_name(PurePosixPath(registered).stem), [])
                    parent = str(PurePosixPath(registered).parent)
                    local = [p for p in candidates if str(PurePosixPath(p).parent) == parent
                             or str(PurePosixPath(p).parent).endswith("/" + parent)]
                    targets = local or scripts.get(_name(label), [])
                if len(targets) == 1:
                    connect(path, targets[0], label, "registration")

    return tuple(sorted(found, key=lambda r: (r.source, r.target, r.kind, r.symbol)))
