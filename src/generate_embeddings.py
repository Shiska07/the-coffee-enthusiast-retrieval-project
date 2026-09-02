"""Phase 1 entry point — reads reviews from SQLite, builds one combined
passage per review, embeds them, and upserts into the Chroma vector store.

Run from the repo root:
    python -m src.generate_embeddings
    python -m src.generate_embeddings --rebuild --limit 50
"""

from __future__ import annotations

import argparse
import sqlite3

from src.config import settings
from src.retrieval.vector_store import VectorStore, to_document
from src.schemas import ReviewMetadata

QUERY = """
SELECT
    m.review_uid, m.slug, m.rating, m.roaster, m.name, m.location, m.origin,
    m.roast, m.est_price, m.review_date, m.agtron,
    m.aroma, m.acid, m.body, m.flavor, m.aftertaste,
    t.desc_1, t.desc_2_clean, t.desc_3
FROM coffee_reviews m
JOIN coffee_reviews_text t ON m.review_uid = t.review_uid
"""

def _clean(value):
    """Normalize empty/whitespace-only strings to None."""
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return value


def load_rows(limit: int | None = None) -> list[sqlite3.Row]:
    query = QUERY + (f" LIMIT {limit}" if limit else "")
    with sqlite3.connect(settings.SQLITE_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(query).fetchall()
    
    
def build_document(row: sqlite3.Row):
    metadata = ReviewMetadata(
        review_uid=row["review_uid"],
        slug=_clean(row["slug"]),
        rating=row["rating"],
        roaster=_clean(row["roaster"]),
        name=_clean(row["name"]),
        location=_clean(row["location"]),
        origin=_clean(row["origin"]),
        roast=_clean(row["roast"]),
        est_price=_clean(row["est_price"]),
        review_date=row["review_date"],
        agtron=_clean(row["agtron"]),
        aroma=row["aroma"],
        acid=row["acid"],
        body=row["body"],
        flavor=row["flavor"],
        aftertaste=row["aftertaste"],
    )
    # combine per the 03_text_analysis.ipynb decision — check the separator
    # you actually used there and match it if this differs
    text = " ".join(
        part for part in (row["desc_1"], row["desc_3"], row["desc_2_clean"]) if part
    )
    return to_document(text, metadata)


def main(rebuild: bool = False, limit: int | None = None) -> None:
    store = VectorStore()
    if rebuild:
        print(f"Rebuilding collection '{store.collection_name}'...")
        store.reset()

    rows = load_rows(limit=limit)
    print(f"Loaded {len(rows)} reviews from {settings.SQLITE_PATH}.")

    documents = [build_document(row) for row in rows]
    store.upsert(documents)

    print(f"Vector store now has {store.count()} chunks in '{store.collection_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed coffee reviews into Chroma.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the collection first.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N reviews (for testing).")
    args = parser.parse_args()
    main(rebuild=args.rebuild, limit=args.limit)
    