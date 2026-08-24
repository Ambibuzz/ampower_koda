"""Every tunable number the cold-start stages depend on."""

from __future__ import annotations

from typing import Final


CHARS_PER_TOKEN: Final[float] = 3.6

MEMORY_MAX_TOKENS: Final[int] = 800


CHUNK_LINES: Final[int] = 80

CHUNK_CHARS: Final[int] = 6_000

CHUNK_LONG_LINE_STRIDE: Final[int] = 5_600

CHUNK_OVERLAP_LINES: Final[int] = 8

COMMENT_MAX_LINES: Final[int] = 24

CHUNKLESS_ROLES: Final[tuple[str, ...]] = ("constant", "type", "enum")

NON_INDEXABLE_ROLES: Final[tuple[str, ...]] = ("field",)


ANCHOR_DIGEST: Final[int] = 8


MAX_INDEX_FILE_BYTES: Final[int] = 1_500_000

EXCLUDED_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".koda",
        ".venv",
        "venv",
        "env",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "__pycache__",
        "node_modules",
        "bower_components",
        ".pnpm-store",
        ".yarn",
        ".next",
        ".turbo",
        "dist",
        "build",
        "coverage",
        "_to_delete",
    }
)

DEFAULT_REDACT_GLOBS: Final[tuple[str, ...]] = (".env*", "**/secrets*")

MEMORY_FILENAMES: Final[tuple[str, ...]] = ("KODA.md", "CLAUDE.md", "AGENTS.md")


CACHE_ENTRY_VERSION: Final[int] = 5

CACHE_DIRECTORY: Final[str] = ".koda/cache/tags"


COCHANGE_MAX_COMMITS: Final[int] = 1_000

COCHANGE_HALF_LIFE_DAYS: Final[float] = 180.0

COCHANGE_MAX_FILES_PER_COMMIT: Final[int] = 25

COCHANGE_MAX_NEIGHBOURS: Final[int] = 12


EDGE_WEIGHTS: Final[dict[str, float]] = {
    "calls": 1.0,
    "instantiates": 0.6,
    "contains": 0.5,
    "references": 0.35,
}

PAGERANK_ALPHA: Final[float] = 0.85

PAGERANK_ITERATIONS: Final[int] = 40

MAP_BOOST_ESTABLISHED: Final[float] = 4.0
MAP_BOOST_BASENAME: Final[float] = 3.0
MAP_BOOST_PARTIAL: Final[float] = 2.0
MAP_BOOST_DIRECTORY: Final[float] = 1.5

MIRROR_RANK_FACTOR: Final[float] = 0.1

MIRROR_MIN_SHARED_FILES: Final[int] = 5
MIRROR_MIN_SHARED_FRACTION: Final[float] = 0.6

VENDOR_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {
        "reference",
        "vendor",
        "vendored",
        "third_party",
        "thirdparty",
        "external",
        "node_modules",
        "site-packages",
        "bundled",
        "apps",
        "copy",
        "copies",
    }
)

MAP_MAX_TOKENS: Final[int] = 2_000


BUDGET_SHARES: Final[dict[str, float]] = {
    "ledger": 0.03,
    "working_set": 0.08,
    "hot_results": 0.12,
    "fold": 0.015,
}

BUDGET_FLOORS: Final[dict[str, int]] = {"ledger": 4_000, "hot_results": 12_000, "hot_count": 20}

BUDGET_FLOOR_CEILING: Final[float] = 0.25

BUDGET_TOTAL_CEILING: Final[float] = 0.35

BUDGET_CEILED_REGIONS: Final[tuple[str, ...]] = (
    "ledger",
    "working_set",
    "fold",
    "repo_map",
    "memory",
)

BUDGET_TOKENS_PER_HOT_RESULT: Final[int] = 6_000

COMPACTION_TRIGGER_FRACTION: Final[float] = 0.75

REPLY_HEADROOM_TOKENS: Final[int] = 8_000
MAX_TURN_TOKENS_MARGINAL: Final[int] = 80_000
MAX_TURN_TOKENS_OBSERVED: Final[int] = 160_000
OBSERVED_WINDOW_MULTIPLE: Final[float] = 1.25

MAX_ROUNDS: Final[int] = 60
MAX_OUTPUT_TOKENS: Final[int] = 8_000

CALIBRATOR_CLAMP: Final[tuple[float, float]] = (0.4, 3.0)
CALIBRATOR_ALPHA: Final[float] = 0.3


MAX_PREFIX_BREAKPOINTS: Final[int] = 3
MAX_TOTAL_BREAKPOINTS: Final[int] = 4

MIN_CACHEABLE_BY_FAMILY: Final[dict[str, int]] = {
    "opus-5": 512,
    "fable-5": 512,
    "sonnet": 1_024,
    "opus": 4_096,
    "haiku": 4_096,
}
MIN_CACHEABLE_DEFAULT: Final[int] = 1_024

SESSION_ID_MAX_CHARS: Final[int] = 256

CACHE_WRITE_TO_READ: Final[int] = 20


BM25_K1: Final[float] = 1.2
BM25_B: Final[float] = 0.75

BM25_PATH_REPEAT: Final[int] = 2
BM25_SYMBOL_REPEAT: Final[int] = 3

BM25_TEXT_BONUS: Final[float] = 2.5
BM25_PATH_BONUS: Final[float] = 1.5
BM25_SYMBOL_BONUS: Final[float] = 4.0

SAME_FILE_DECAY: Final[float] = 0.85
MAX_HITS_PER_FILE: Final[int] = 6

PROSE_COMMENT_SHARE: Final[float] = 0.85
PROSE_RESULT_PENALTY: Final[float] = 0.55

BRIDGE_SCORE_PER_TERM: Final[float] = 2.5
BRIDGE_MAX_SYMBOLS: Final[int] = 8
SYMBOL_EXPANSION_MAX: Final[int] = 12
SYMBOL_EXPANSION_WEIGHT: Final[float] = 0.3
SYMBOL_EXPANSION_DEFINITION_BONUS: Final[float] = 3.5
BRIDGE_DEFINITION_BONUS: Final[float] = 4.0
BRIDGE_TERM_WEIGHT: Final[float] = 0.35

ISSUE_QUERY_MIN_CHARS: Final[int] = 320
VIEW_RANK_DECAY: Final[float] = 0.35
VIEW_WEIGHTS: Final[dict[str, float]] = {
    "original": 1.0,
    "title": 1.2,
    "identifiers": 1.3,
    "anchor": 1.55,
    "path": 1.8,
}
IDENTIFIER_MAX_SITES: Final[int] = 100

STRUCTURAL_MAX_SEEDS: Final[int] = 8
STRUCTURAL_PER_SYMBOL_CAP: Final[int] = 4
STRUCTURAL_LIMIT: Final[int] = 12
STRUCTURAL_DEFINITION_WEIGHT: Final[float] = 1.0
STRUCTURAL_REFERENCE_WEIGHT: Final[float] = 0.4

GRAPH_DEPTH: Final[int] = 2
GRAPH_EXPAND_LIMIT: Final[int] = 24
GRAPH_KEEP: Final[int] = 12
GRAPH_HOP_DECAY: Final[float] = 0.45

HISTORY_NEIGHBOURS: Final[int] = 6

FUSION_RANK_DECAY: Final[float] = 0.35
RRF_K: Final[int] = 60
SEED_LIMIT: Final[int] = 8
SOURCE_LIMIT: Final[int] = 40
UNION_LIMIT: Final[int] = 60
DEFAULT_SEARCH_LIMIT: Final[int] = 20
MAX_SEARCH_LIMIT: Final[int] = 50

RERANK_WEIGHTS: Final[dict[str, float]] = {
    "prior": 5.00,
    "centrality": 0.75,
    "leg_trust": 0.50,
    "term_coverage": 0.45,
    "symbol_match": 0.40,
    "leg_agreement": 0.20,
    "rarity": 0.15,
    "definitionness": 0.00,
    "prose_penalty": -0.30,
    "vendored_copy": -0.30,
}

DENSE_CONFIDENCE_WEIGHT: Final[float] = 0.4
AGREEMENT_LIFT: Final[float] = 0.2


ESCALATION_CONFIDENT: Final[float] = 0.90

ESCALATION_WEAK: Final[float] = 0.75

ESCALATION_MID_MARGIN: Final[float] = 0.25

FANOUT_MAX: Final[int] = 4

FANOUT_RESERVED_SLOTS: Final[int] = 2

TRANSLATE_MAX_OUTPUT_TOKENS: Final[int] = 300

TRANSLATE_MAX_NAMES: Final[int] = 8


DISTILL_SCAN_LINES: Final[int] = 60

DISTILL_MAX_ANCHORS: Final[int] = 4

LEDGER_MERGE_MAX_CHARS: Final[int] = 160
LEDGER_MERGE_MAX_PART_CHARS: Final[int] = 72
LEDGER_MERGE_MAX_REFS: Final[int] = 4

LEDGER_RECENT_WINDOW: Final[int] = 12


WORKING_SET_SEARCH_LIMIT: Final[int] = 8

WORKING_SET_MAX_SPANS: Final[int] = 12

WORKING_SET_MAX_EDITED: Final[int] = 8

WORKING_SET_EXCERPT_CHARS: Final[int] = 120

WORKING_SET_WEAK_COVERAGE: Final[float] = ESCALATION_WEAK


HOTCOLD_LOW_WATER: float = 0.5

AMORTISATION_RATIO: Final[int] = CACHE_WRITE_TO_READ

HOTCOLD_HARD_PRESSURE: Final[int] = 2

ROADMAP_MAX_COORDINATES: Final[int] = 24

KEEP_RAW_TURNS: Final[int] = 3

COMPACTION_CLIFF_FRACTION: float = 0.75

COMPACTION_MAX_OUTPUT_TOKENS: Final[int] = 700

COMPACTION_KEEP_PROSE: Final[int] = 2

COMPACTION_THRASH_WINDOW: Final[int] = 10

FOLD_MAX_OUTPUT_TOKENS: Final[int] = 900

FOLD_ARGS_CLIP_CHARS: Final[int] = 200
FOLD_RESULT_CLIP_CHARS: Final[int] = 400


DEFAULT_WINDOW_TOKENS: Final[int] = 180_000

DEFAULT_ARCHITECT_MODEL: Final[str] = "anthropic/claude-sonnet-5"
