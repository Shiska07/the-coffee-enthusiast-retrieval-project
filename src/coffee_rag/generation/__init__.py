"""Generation package for the coffee RAG pipeline.

This module is intentionally lightweight at startup so the project can
import cleanly before the final generation pipeline is implemented.
"""

from __future__ import annotations

__all__ = ["generate_answer"]


def generate_answer(*args, **kwargs):
    """Placeholder generation entry point.

    Replace this with the actual LLM or prompt orchestration logic.
    """
    raise NotImplementedError("Generation logic has not been implemented yet.")
