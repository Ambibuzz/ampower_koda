"""Reading a workspace's ``.koda/config.toml`` into a resolved config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import CoreError
from .merge import merge_config, merge_config_by_key, parse_toml
from .schema import CoreConfig, config_defaults

if TYPE_CHECKING:
    from ..workspace.ports import Workspace

CONFIG_PATH = ".koda/config.toml"


def resolve_config(
    workspace: Workspace,
    overrides: dict | None,
    notes: list[str],
) -> CoreConfig:
    """Resolve ``defaults < .koda/config.toml < overrides``.

    An invalid key in the file is skipped on its own, with a note naming it and
    the real error; the file's other keys still apply. An invalid override is
    the caller's mistake and raises as itself, never blamed on the file.
    """
    config = config_defaults()
    if workspace.stat(CONFIG_PATH) is not None:
        try:
            from_file = parse_toml(workspace.read_bytes(CONFIG_PATH).decode("utf-8"))
        except (CoreError, UnicodeDecodeError) as exc:
            notes.append(f"{CONFIG_PATH} ignored: {exc}")
        else:
            config, errors = merge_config_by_key(from_file, base=config)
            notes.extend(f"{CONFIG_PATH}: key {error.key!r} ignored: {error.reason}" for error in errors)

    return merge_config(overrides, base=config)
