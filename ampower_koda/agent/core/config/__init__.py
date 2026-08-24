"""Configuration: one defaulted schema, one precedence rule."""

from __future__ import annotations

from .merge import merge_config, parse_toml
from .schema import (
    ContextConfig,
    CoreConfig,
    EscalationConfig,
    HistoryConfig,
    IndexingConfig,
    ModelsConfig,
    RetrievalConfig,
    SecurityConfig,
    as_mapping,
    config_defaults,
)

__all__ = [
    "ContextConfig",
    "CoreConfig",
    "EscalationConfig",
    "HistoryConfig",
    "IndexingConfig",
    "ModelsConfig",
    "RetrievalConfig",
    "SecurityConfig",
    "as_mapping",
    "config_defaults",
    "merge_config",
    "parse_toml",
]
