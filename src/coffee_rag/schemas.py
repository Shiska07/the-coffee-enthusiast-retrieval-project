"""Shared data structures passed between pipeline stages.
"""

from __future__ import annotations
from datetime import date

from langchain_core.documents import Document
from pydantic import BaseModel, Field, HttpUrl

# ----  Embedding stage -------------------------------------------------------------


class ReviewMetadata(BaseModel):
    """Metadata for a single coffee review."""

    review_uid: str
    slug: HttpUrl | None = None
    rating: int | None = None
    roaster: str | None = None
    name: str | None = None
    location: str | None = None
    origin: str | None = None
    roast: str | None = None
    est_price: str | None = None
    review_date: date | None = None
    agtron: str | None = None
    aroma: float | None = None
    acid: float | None = None
    body: float | None = None
    flavor: float | None = None
    aftertaste: float | None = None
    
    
class AnswerResult(BaseModel):
    """A single answer returned by the LLM generation stage."""

    question: str
    answer: str
    contexts: list[Document] = Field(default_factory=list)  # which retrieved documents were used to generate the answer
    attempts: int = 1
    used_web_search: bool = False  # whether the answer was generated using web search fallback
    low_confidence: bool = False  # whether the answer was flagged as low-confidence by the hallucination detection stage
    query_strategy: str = "raw"  # which query strategy was used to retrieve documents for this answer
    