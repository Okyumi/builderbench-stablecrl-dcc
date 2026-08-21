"""Continual-learning utilities for the StableCRL BuilderBench port."""

from continual.semantic_layout import SemanticLayout
from continual.task_manifest import TaskRecord, build_manifest, canonical_goal_hash

__all__ = [
    "SemanticLayout",
    "TaskRecord",
    "build_manifest",
    "canonical_goal_hash",
]
