"""Turning ``total`` into ``Cart.total``."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..contracts.symbols import Definition, DefinitionSite


def resolve_definitions(sites: Iterable[DefinitionSite]) -> tuple[Definition, ...]:
    """Attach container chains to flat definition sites."""
    ordered = sorted(sites, key=_nesting_key)
    stack: list[DefinitionSite] = []
    resolved: list[Definition] = []

    for site in ordered:
        while stack and not _encloses(stack[-1], site):
            stack.pop()

        resolved.append(
            Definition(
                name=site.name,
                role=site.role,
                extent=site.extent,
                name_line=site.name_line,
                container=tuple(open_site.name for open_site in stack),
            )
        )
        stack.append(site)

    return tuple(resolved)


def _encloses(outer: DefinitionSite, inner: DefinitionSite) -> bool:
    """True when ``outer`` *strictly* contains ``inner``."""
    return outer.extent.contains(inner.extent) and outer.extent != inner.extent


def _nesting_key(site: DefinitionSite) -> tuple[int, int, str]:
    """Outermost first; ties broken by name so the order is total."""
    return (site.extent.start, -site.extent.end, site.name)


def qualified_names(definitions: Sequence[Definition]) -> tuple[str, ...]:
    """Return every definition's qualified name, in order. A convenience for tests."""
    return tuple(definition.qualified_name for definition in definitions)
