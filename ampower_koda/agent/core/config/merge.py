"""Config precedence, as pure functions over sparse mappings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, TypeVar

from ..errors import ConfigError
from .schema import CoreConfig, config_defaults

T = TypeVar("T")


def merge_config(
    *overrides: Mapping[str, Any] | None,
    base: CoreConfig | None = None,
) -> CoreConfig:
    """Apply sparse ``overrides`` over ``base``, lowest precedence first."""
    config = base if base is not None else config_defaults()
    for override in overrides:
        if override:
            config = _apply(config, override, prefix="")
    return CoreConfig(
        indexing=config.indexing,
        context=config.context,
        history=config.history,
        models=config.models,
        retrieval=config.retrieval,
        security=config.security,
    )


def _apply(node: T, override: Mapping[str, Any], *, prefix: str) -> T:
    """Return ``node`` with ``override`` applied, recursing into config groups."""
    if not is_dataclass(node):  # pragma: no cover
        raise ConfigError(prefix or "<root>", "is not a config group")

    known = {spec.name: spec for spec in fields(node)}
    updates: dict[str, Any] = {}

    for key, value in override.items():
        path = f"{prefix}{key}"
        spec = known.get(key)
        if spec is None:
            raise ConfigError(path, f"unknown key; expected one of {sorted(known)}")

        current = getattr(node, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ConfigError(path, "expected a table of keys")
            updates[key] = _apply(current, value, prefix=f"{path}.")
        else:
            updates[key] = _coerce(path, current, value)

    return replace(node, **updates) if updates else node


def _coerce(path: str, current: Any, value: Any) -> Any:
    """Coerce ``value`` to the shape of ``current``, or explain why it cannot be."""
    if isinstance(current, tuple):
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ConfigError(path, "expected a list")
        return tuple(str(item) for item in value)

    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(path, f"expected a boolean, got {type(value).__name__}")
        return value

    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(path, f"expected an integer, got {type(value).__name__}")
        return value

    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(path, f"expected a number, got {type(value).__name__}")
        return float(value)

    if isinstance(current, str):
        if not isinstance(value, str):
            raise ConfigError(path, f"expected a string, got {type(value).__name__}")
        return value

    raise ConfigError(path, f"unsupported config type {type(current).__name__}")


def parse_toml(text: str) -> dict[str, Any]:
    """Parse a config file's text into a sparse override mapping."""
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ConfigError(
                "<file>",
                "no TOML parser available; install tomli on Python 3.10",
            ) from exc

    try:
        return tomllib.loads(text)
    except Exception as exc:
        raise ConfigError("<file>", f"is not valid TOML: {exc}") from exc
