"""The reference graph: edges and vendored-copy detection."""

from __future__ import annotations

from .edges import CodeGraph, Edge, ambiguity_discount, build_graph
from .mirrors import detect_mirrors

__all__ = ["CodeGraph", "Edge", "ambiguity_discount", "build_graph", "detect_mirrors"]
