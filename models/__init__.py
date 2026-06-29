"""
models — graph neural network models for lymphoma subtype classification.

Consumes the graphs produced by ``util`` / ``scripts/build_graphs.py``.
"""

from .gat import GATGraphClassifier

__all__ = ["GATGraphClassifier"]
