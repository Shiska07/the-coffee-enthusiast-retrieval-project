"""Shared data structures passed between pipeline stages.
"""

from __future__ import annotations
from datetime import date

from langchain_core.documents import Document
from pydantic import BaseModel, HttpUrl

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
    
    
