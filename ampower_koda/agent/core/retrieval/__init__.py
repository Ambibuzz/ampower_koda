"""Candidate retrieval and dedicated relevance ranking, all off-prompt."""

from __future__ import annotations

from .bm25 import LexicalIndex, build_lexical_index, is_prose, score
from .confidence import margin_of
from .engine import Retriever, brief, build_retriever, search
from .fusion import FusedHit, fuse, leg_trust
from .legs import Seed, graph_leg, history_leg, seeds_from, structural_leg
from .query import QueryPlan, QueryView, plan_query
from .rerank import Ranking, rerank
from .select import diversify
from .tokenize import CONCEPT_GROUPS, STOP_WORDS, is_code_shaped, stem, tokenize

__all__ = [
    "CONCEPT_GROUPS",
    "STOP_WORDS",
    "FusedHit",
    "LexicalIndex",
    "QueryPlan",
    "QueryView",
    "Ranking",
    "Retriever",
    "Seed",
    "build_lexical_index",
    "build_retriever",
    "diversify",
    "fuse",
    "graph_leg",
    "history_leg",
    "is_code_shaped",
    "is_prose",
    "leg_trust",
    "margin_of",
    "plan_query",
    "rerank",
    "score",
    "brief",
    "search",
    "seeds_from",
    "stem",
    "structural_leg",
    "tokenize",
]
