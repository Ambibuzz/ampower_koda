"""Literal Frappe Query Builder field references, without importing application code."""
from __future__ import annotations

import ast

DOCTYPE_FACTORIES = {"frappe.qb.DocType", "frappe.query_builder.DocType"}


def query_fields(source: str) -> list[tuple[str, str, int]]:
    """``(doctype, field, line)`` for each ``qb.DocType("X").field`` reference, sorted by line.

    Each function is its own scope: a name counts as a DocType only when it is bound to
    exactly one DocType there (parameters shadow module-level bindings). Method calls on a
    table (``.as_()``, ``.isin()``) are not fields; an unparsable source has no references.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    def nodes(body):
        # Every node of this scope; nested functions, classes and lambdas are their own scopes.
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            yield node
            yield from nodes(ast.iter_child_nodes(node))

    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return dotted(node.value) + "." + node.attr
        return ""

    def aliases(items, inherited=()):
        # Local names for DocType imported from frappe.query_builder.
        result = set(inherited)
        for node in items:
            if isinstance(node, ast.ImportFrom) and node.module == "frappe.query_builder":
                result.update(alias.asname or alias.name for alias in node.names if alias.name == "DocType")
        return result

    def table(node, env, factories):
        # The DocType an expression refers to, or None when it is not a single known DocType.
        if isinstance(node, ast.Name):
            choices = env.get(node.id, set())
            return next(iter(choices)) if len(choices) == 1 else None
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            if name in factories or name in DOCTYPE_FACTORIES:
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    return node.args[0].value
            if isinstance(node.func, ast.Attribute) and node.func.attr == "as_":
                return table(node.func.value, env, factories)
        return None

    def bindings(items, inherited, factories):
        # Name -> every DocType it is assigned in this scope (a loop target binds None).
        env = {key: set(value) for key, value in inherited.items()}
        local = set()

        def assign(target, value):
            if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
                for left, right in zip(target.elts, value.elts):
                    assign(left, right)
            elif isinstance(target, ast.Name):
                kind = table(value, env, factories)  # before rebinding: `t = t.as_("x")` keeps t's DocType
                if target.id not in local:
                    env[target.id] = set()
                    local.add(target.id)
                env[target.id].add(kind)

        for node in items:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    assign(target, node.value)
            elif isinstance(node, ast.AnnAssign):
                assign(node.target, node.value)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                assign(node.target, None)
        return env

    top = list(nodes(tree.body))
    factories = aliases(top)
    global_bindings = bindings(top, {}, factories)
    scopes = [(top, global_bindings, factories)]
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        items = list(nodes(function.body))
        local_factories = aliases(items, factories)
        inherited = dict(global_bindings)
        for argument in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]:
            inherited.pop(argument.arg, None)
            local_factories.discard(argument.arg)
        scopes.append((items, bindings(items, inherited, local_factories), local_factories))

    found = set()
    for items, env, constructors in scopes:
        called = {id(node.func) for node in items if isinstance(node, ast.Call)}
        for node in items:
            if not isinstance(node, ast.Attribute) or id(node) in called:
                continue
            if node.attr.startswith("_") or node.attr == "star":
                continue
            doctype = table(node.value, env, constructors)
            if doctype:
                found.add((doctype, node.attr, node.lineno))
    return sorted(found, key=lambda item: (item[2], item[0], item[1]))
