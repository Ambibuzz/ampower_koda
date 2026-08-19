"""The config surface for the cold-start stages."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from ..constants import (
    COCHANGE_HALF_LIFE_DAYS,
    COCHANGE_MAX_COMMITS,
    COCHANGE_MAX_FILES_PER_COMMIT,
    COCHANGE_MAX_NEIGHBOURS,
    DEFAULT_ARCHITECT_MODEL,
    DEFAULT_REDACT_GLOBS,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_WINDOW_TOKENS,
    ESCALATION_CONFIDENT,
    ESCALATION_MID_MARGIN,
    ESCALATION_WEAK,
    FANOUT_MAX,
    MAP_MAX_TOKENS,
    MAX_INDEX_FILE_BYTES,
    MAX_SEARCH_LIMIT,
    MEMORY_MAX_TOKENS,
)
from ..errors import ConfigError


@dataclass(frozen=True, slots=True)
class IndexingConfig:
    """What gets read off disk and turned into chunks."""

    max_file_bytes: int = MAX_INDEX_FILE_BYTES
    use_cache: bool = True
    extra_excluded_directories: tuple[str, ...] = ()
    """Appended to the built-in exclusions, never replacing them. A user who
    wants to index ``node_modules`` has a different problem than a config key
    can solve."""

    def validate(self) -> None:
        if self.max_file_bytes <= 0:
            raise ConfigError("indexing.max_file_bytes", "must be positive")


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Budgets for the blocks cold start renders."""

    window_tokens: int = DEFAULT_WINDOW_TOKENS
    """Every context budget derives from this one number. It was a private
    constant once, "which meant a 32k model and a 200k model were handed
    byte-identical budgets"."""

    memory_tokens: int = MEMORY_MAX_TOKENS
    """Shared across all repository memory files, not per file."""

    map_tokens: int = MAP_MAX_TOKENS

    ledger_soft_tokens: int = 0
    """0 lets the allocator decide. A non-zero value is an explicit override and
    wins outright — the key predates the allocator, and a number a developer
    turned up on purpose must not be quietly re-derived."""

    def validate(self) -> None:
        if self.window_tokens <= 0:
            raise ConfigError("context.window_tokens", "must be positive")
        if self.memory_tokens < 0:
            raise ConfigError("context.memory_tokens", "cannot be negative")
        if self.map_tokens < 0:
            raise ConfigError("context.map_tokens", "cannot be negative")
        if self.ledger_soft_tokens < 0:
            raise ConfigError("context.ledger_soft_tokens", "cannot be negative")


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    """Which model the prompt is assembled for."""

    architect: str = DEFAULT_ARCHITECT_MODEL

    def validate(self) -> None:
        if not self.architect.strip():
            raise ConfigError("models.architect", "cannot be empty")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """How wide a search reaches."""

    limit: int = DEFAULT_SEARCH_LIMIT
    """Hits returned to the caller. The candidate pool behind it is 40-60 wide
    and costs nothing, because candidates never enter the prompt."""

    expand: bool = True
    """Run the structural, graph and history legs when the query is anchored.
    Turning it off leaves a purely lexical retriever, which is a useful thing to
    be able to measure against."""

    def validate(self) -> None:
        if not 1 <= self.limit <= MAX_SEARCH_LIMIT:
            raise ConfigError("retrieval.limit", f"must be between 1 and {MAX_SEARCH_LIMIT}")


@dataclass(frozen=True, slots=True)
class EscalationConfig:
    """When a weak search may spend a model call on *vocabulary*."""

    enabled: bool = True
    """The master switch. Off means no escalation and no model call, ever, and
    the ladder becomes one comparison and an early return."""

    fan_out: bool = False
    """The ``expand`` rung: up to four rewrites along four angles. See above."""

    confident: float = ESCALATION_CONFIDENT
    weak: float = ESCALATION_WEAK
    mid_margin: float = ESCALATION_MID_MARGIN

    max_rewrites: int = FANOUT_MAX
    """Rewrites per fan-out call, and therefore *extra searches* per fan-out
    call. Turning it down is supported; turning it up past
    :data:`~…constants.FANOUT_MAX` is not, because that constant is the measured
    point where merge cost stops being negligible beside one more search."""

    def validate(self) -> None:
        if not 0.0 <= self.weak <= self.confident <= 1.0:
            raise ConfigError(
                "escalation.weak",
                "must satisfy 0 ≤ weak ≤ confident ≤ 1 — the bands cannot cross",
            )
        if not 0.0 <= self.mid_margin <= 1.0:
            raise ConfigError("escalation.mid_margin", "must be between 0 and 1")
        if not 1 <= self.max_rewrites <= FANOUT_MAX:
            raise ConfigError("escalation.max_rewrites", f"must be between 1 and {FANOUT_MAX}")


@dataclass(frozen=True, slots=True)
class HistoryConfig:
    """How much git history feeds co-change memory."""

    enabled: bool = True
    max_commits: int = COCHANGE_MAX_COMMITS
    half_life_days: float = COCHANGE_HALF_LIFE_DAYS
    max_files_per_commit: int = COCHANGE_MAX_FILES_PER_COMMIT
    max_neighbours: int = COCHANGE_MAX_NEIGHBOURS

    def validate(self) -> None:
        if self.max_commits < 0:
            raise ConfigError("history.max_commits", "cannot be negative")
        if self.half_life_days <= 0:
            raise ConfigError("history.half_life_days", "must be positive")
        if self.max_files_per_commit < 2:
            raise ConfigError(
                "history.max_files_per_commit",
                "must be at least 2 — a commit touching one file couples nothing",
            )
        if self.max_neighbours < 0:
            raise ConfigError("history.max_neighbours", "cannot be negative")


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    """What never leaves the machine."""

    redact_globs: tuple[str, ...] = DEFAULT_REDACT_GLOBS
    """Matched before a file is read for parsing. A redacted file is refused,
    not reported absent — see
    :class:`~ampower_koda.agent.core.errors.RedactedFileError`."""

    def validate(self) -> None:
        if any(not glob.strip() for glob in self.redact_globs):
            raise ConfigError("security.redact_globs", "contains an empty pattern")


@dataclass(frozen=True, slots=True)
class CoreConfig:
    """The whole config surface, one group per pipeline concern."""

    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __post_init__(self) -> None:
        """Validate every group, found by reflection rather than by a list."""
        for group in fields(self):
            validate = getattr(getattr(self, group.name), "validate", None)
            if callable(validate):
                validate()


def config_defaults() -> CoreConfig:
    """Return a fresh, fully defaulted config."""
    return CoreConfig()


def as_mapping(config: Any) -> dict[str, Any]:
    """Render a config (or group) as nested plain dicts."""
    if not is_dataclass(config):
        raise ConfigError("<root>", f"not a config dataclass: {type(config).__name__}")

    result: dict[str, Any] = {}
    for spec in fields(config):
        value = getattr(config, spec.name)
        if is_dataclass(value):
            result[spec.name] = as_mapping(value)
        elif isinstance(value, tuple):
            result[spec.name] = list(value)
        else:
            result[spec.name] = value
    return result
