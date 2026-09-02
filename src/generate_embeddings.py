"""Build the vector store from SQLite.

Reads ``coffee_reviews_text`` (joined to ``coffee_reviews`` for metadata),
builds the combined passage per review (chunking decision from
``notebooks/03_text_analysis.ipynb``), embeds it, and upserts into the local
Chroma store under ``vector_store/chroma/``.

Idempotent: re-running upserts by ``review_uid``. Use ``--rebuild`` to drop the
collection first.

Usage:
    python -m src.generate_embeddings
    python -m src.generate_embeddings --rebuild --limit 50   # quick smoke test
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

from src.config import settings, combined_text



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="drop the collection first")
    parser.add_argument("--limit", type=int, default=None, help="only process N reviews")
    parser.add_argument("--no-smoke-test", action="store_true")
    args = parser.parse_args()

    if not settings.OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY is not set (put it in .env).")

    store = build(rebuild=args.rebuild, limit=args.limit)
    if not args.no_smoke_test:
        _smoke_test(store)


if __name__ == "__main__":
    main()