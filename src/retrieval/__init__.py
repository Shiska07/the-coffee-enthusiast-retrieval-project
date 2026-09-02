"""Retrieval package for the coffee RAG pipeline.

This package is the home for retrieval logic such as vector search,
reranking, and filtering before the LLM answer is generated.
"""

from __future__ import annotations

__all__ = ["retrieve_documents"]


def retrieve_documents(*args, **kwargs):
    """Placeholder retrieval entry point.

    Return the candidate documents or passages relevant to a user query.
    """
    raise NotImplementedError("Retrieval logic has not been implemented yet.")
