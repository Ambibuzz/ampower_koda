"""The config surface for the cold-start stages."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import isfinite

from ..constants import (
    COCHANGE_HALF_LIFE_DAYS,
    COCHANGE_MAX_COMMITS,
    COCHANGE_MAX_FILES_PER_COMMIT,
    COCHANGE_MAX_NEIGHBOURS,
    DEFAULT_ARCHITECT_MODEL,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_WINDOW_TOKENS,
    MAX_INDEX_FILE_BYTES,
    MAX_SEARCH_LIMIT,
    MEMORY_MAX_TOKENS,
)
from ..errors import ConfigError
from ..budget.request import DEFAULT_INPUT_TOKENS


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

    input_tokens: int = DEFAULT_INPUT_TOKENS
    """Full-input cleanup threshold, capped by model capacity and reply space.
    History stays intact below it; pressure cleanup targets two thirds of it."""

    ledger_soft_tokens: int = 0
    """0 lets the allocator decide. A non-zero value is an explicit override and
    wins outright — the key predates the allocator, and a number a developer
    turned up on purpose must not be quietly re-derived."""

    def validate(self) -> None:
        if self.window_tokens <= 0:
            raise ConfigError("context.window_tokens", "must be positive")
        if self.input_tokens <= 0:
            raise ConfigError("context.input_tokens", "must be positive")
        if self.memory_tokens < 0:
            raise ConfigError("context.memory_tokens", "cannot be negative")
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
    """Hits returned to the caller. Candidate generation is local; dedicated
    reranking has its own bounded request and never enters the chat prompt."""

    expand: bool = True
    """Run the structural, graph and history legs from retrieved candidates.
    Turning it off leaves a purely lexical retriever, which is a useful thing to
    be able to measure against."""

    def validate(self) -> None:
        if not 1 <= self.limit <= MAX_SEARCH_LIMIT:
            raise ConfigError("retrieval.limit", f"must be between 1 and {MAX_SEARCH_LIMIT}")


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """Bounded dedicated reranking, independent of the conversational model."""

    enabled: bool = True
    model: str = "cohere/rerank-v3.5"
    candidates: int = 60
    per_file: int = 3
    query_chars: int = 4000
    document_chars: int = 3000
    timeout_seconds: float = 5.0
    min_score: float = 0.1
    """A model-specific relevance floor, not a probability of correctness."""

    brief_model: str = "cohere/rerank-4-pro"
    """The starting-points brief: one call over file cards, one over their definitions."""

    def validate(self) -> None:
        if not self.model.strip():
            raise ConfigError("rerank.model", "cannot be empty")
        if not self.brief_model.strip():
            raise ConfigError("rerank.brief_model", "cannot be empty")
        for name, low, high in (("candidates", 1, 100), ("per_file", 1, 10),
                                ("query_chars", 256, 8000), ("document_chars", 256, 8000)):
            if not low <= getattr(self, name) <= high:
                raise ConfigError(f"rerank.{name}", f"must be between {low} and {high}")
        if not isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 30:
            raise ConfigError("rerank.timeout_seconds", "must be greater than 0 and at most 30")
        if not isfinite(self.min_score) or not 0 <= self.min_score <= 1:
            raise ConfigError("rerank.min_score", "must be between 0 and 1")


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

    redact_globs: tuple[str, ...] = ()
    """Appended to the built-in ``DEFAULT_REDACT_GLOBS``, never replacing them:
    ``["*.pem"]`` once replaced them and made ``.env`` an ordinary indexed file.
    Matched before a file is read for parsing. A redacted file is skipped at
    discovery and reported with reason ``redacted``, not silently absent, and
    the read tools refuse it."""

    def validate(self) -> None:
        if any(not glob.strip() for glob in self.redact_globs):
            raise ConfigError("security.redact_globs", "contains an empty pattern")


@dataclass(frozen=True, slots=True)
class CoreConfig:
    """The whole config surface, one group per pipeline concern."""

    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
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
