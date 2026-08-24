"""Retrieval: six legs, one fuser, ten features, all of it off-prompt."""

from __future__ import annotations

from .bm25 import LexicalIndex, build_lexical_index, is_prose, score
from .confidence import compute_confidence, margin_of
from .engine import Retriever, build_retriever, search
from .fusion import FusedHit, fuse, leg_trust, reciprocal_rank_fusion
from .legs import Seed, graph_leg, history_leg, seeds_from, structural_leg
from .query import QueryPlan, QueryView, plan_query
from .rerank import RerankContext, features, rerank
from .select import diversify, is_test_path
from .tokenize import CONCEPT_GROUPS, STOP_WORDS, is_code_shaped, stem, tokenize

__all__ = [
    "CONCEPT_GROUPS",
    "STOP_WORDS",
    "FusedHit",
    "LexicalIndex",
    "QueryPlan",
    "QueryView",
    "RerankContext",
    "Retriever",
    "Seed",
    "build_lexical_index",
    "build_retriever",
    "compute_confidence",
    "diversify",
    "features",
    "fuse",
    "graph_leg",
    "history_leg",
    "is_code_shaped",
    "is_prose",
    "is_test_path",
    "leg_trust",
    "margin_of",
    "plan_query",
    "reciprocal_rank_fusion",
    "rerank",
    "score",
    "search",
    "seeds_from",
    "stem",
    "structural_leg",
    "tokenize",
]
