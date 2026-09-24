"""Extract literal Frappe RPC calls from JavaScript without executing it."""

from __future__ import annotations

import re

_JS_TOKEN = re.compile(
    r'''//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|[A-Za-z_$][\w$]*|[^\s]'''
)


def _literal(token: str, next_token: str) -> str | None:
    # Only a complete literal expression is a static method path. A template
    # interpolation, concatenation or conditional needs semantic/runtime review.
    if next_token not in {",", "}", ")"} or not token.startswith(('"', "'", '`')):
        return None
    if token.startswith('`') and re.search(r'(?<!\\)(?:\\\\)*\$\{', token):
        return None
    return token[1:-1]


def call_options(source: str):
    """Yield literal options for ``frappe.call`` and ``frappe.xcall`` calls.

    Top-level object fields are returned without allowing nested ``args`` keys
    to impersonate call options. Direct string calls are represented as a
    ``method`` option. Dynamic values remain ``None`` for semantic review.
    """
    tokens = [match.group() for match in _JS_TOKEN.finditer(source)
              if not match.group().startswith(("//", "/*"))]
    for index in range(len(tokens) - 4):
        if tokens[index:index + 2] != ["frappe", "."]:
            continue
        if tokens[index + 2] not in {"call", "xcall"} or tokens[index + 3] != "(":
            continue
        first = tokens[index + 4]
        if first.startswith(("\"", "'", "`")):
            yield {"method": _literal(first, tokens[index + 5] if index + 5 < len(tokens) else '')}
            continue
        if first != "{":
            yield {"method": None}
            continue
        depth, options, cursor = 1, {}, index + 5
        while cursor < len(tokens) and depth:
            token = tokens[cursor]
            if depth == 1 and cursor + 2 < len(tokens) and tokens[cursor + 1] == ":":
                key = token.strip("\"'`")
                value = tokens[cursor + 2]
                options[key] = _literal(value, tokens[cursor + 3] if cursor + 3 < len(tokens) else '')
            if token in ("{", "[", "("):
                depth += 1
            elif token in ("}", "]", ")"):
                depth -= 1
            cursor += 1
        if not depth:
            yield options
