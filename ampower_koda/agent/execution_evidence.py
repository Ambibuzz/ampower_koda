"""Bounded current-source evidence for implementation and review."""
import difflib
import json
import re
from .execution_contract import revision


def source_context(paths, read_current, before=None, limit=12000):
    """Share changed regions fairly; explicitly mark all omitted source."""
    paths = list(dict.fromkeys(paths))
    blocks = []
    remaining = limit
    for index, path in enumerate(paths):
        allowance = remaining // (len(paths) - index)
        current = read_current(path)
        header = f"\n### {path} SHA256 {revision(current)}\n"
        if current is None:
            body = "[File missing]"
        else:
            lines = current.splitlines()
            original = (before or {}).get(path)
            selected = set()
            if original is not None and original != current:
                for tag, _, _, start, end in difflib.SequenceMatcher(None, original.splitlines(), lines, autojunk=False).get_opcodes():
                    if tag != "equal":
                        selected.update(range(max(0, start - 4), min(len(lines), max(end, start + 1) + 4)))
            else:
                selected.update(range(len(lines)))
            body = ""
            available = max(0, allowance - len(header) - 100)
            included = 0
            for line in sorted(selected):
                snippet = f"{line + 1:5} | {lines[line]}\n"
                if len(body) + len(snippet) > available:
                    break
                body += snippet
                included += 1
            if included < len(lines):
                body += "[Source excerpt; omitted lines must be read with tools when relevant.]\n"
        block = header + body
        if len(block) > remaining:
            blocks.append("\n[Further source omitted; use tools.]"[:remaining])
            break
        blocks.append(block)
        remaining -= len(block)
    return "".join(blocks)


class SourceMemory:
    """Keep bounded successful source reads after tool rounds are compacted.

    Entries are revision checked before reuse and invalidated on every attempted
    write, including failed writes that might have partially changed a file.
    """
    def __init__(self, read_current, limit=12000):
        self.read_current = read_current
        self.limit = limit
        self.entries = {}

    def record(self, arguments, result, round_number):
        path = arguments.get("path", "")
        if not path or not result.startswith(f"[{path}]"):
            return
        try:
            current = self.read_current(path)
            lines = (current or "").splitlines()
            header = result.splitlines()[0]
            full = re.fullmatch(re.escape(f"[{path}]") + r" (\d+) lines(?: total)?", header)
            part = re.fullmatch(re.escape(f"[{path}]") + r" lines (\d+)-(\d+) of (\d+)", header)
            if full:
                start, end, total = 1, int(full[1]), int(full[1])
            elif part:
                start, end, total = map(int, part.groups())
            else:
                return
            if total != len(lines):
                return
            numbered = re.findall(r"^\s*(\d+) \| (.*)$", result, re.MULTILINE)
            # The tool output and digest must describe the same snapshot. A
            # concurrent edit between the tool read and this read invalidates it.
            if [int(n) for n, _ in numbered] != list(range(start, end + 1)):
                return
            if not numbered or any(not 1 <= int(n) <= len(lines) or lines[int(n) - 1].rstrip() != text
                                   for n, text in numbered):
                return
            digest = revision(current)
        except (OSError, ValueError):
            return
        key = json.dumps(arguments, sort_keys=True)
        self.entries.pop(key, None)
        self.entries[key] = (path, digest, result, round_number)
        while sum(len(v[2]) + len(v[0]) + 100 for v in self.entries.values()) > self.limit:
            self.entries.pop(next(iter(self.entries)))

    def invalidate(self):
        self.entries.clear()

    def render(self, retained_rounds, max_chars=None):
        blocks = []
        for key, (path, digest, result, number) in list(self.entries.items()):
            if number in retained_rounds:
                continue
            try:
                current = revision(self.read_current(path))
            except (OSError, ValueError):
                current = None
            if current != digest:
                del self.entries[key]
            else:
                blocks.append(f"{path} SHA256 {digest}\n{result}")
        if max_chars is None:
            return "\n".join(blocks)
        kept = []
        used = 0
        for block in reversed(blocks):
            if used + len(block) + 1 <= max_chars:
                kept.append(block)
                used += len(block) + 1
        return "\n".join(reversed(kept))
