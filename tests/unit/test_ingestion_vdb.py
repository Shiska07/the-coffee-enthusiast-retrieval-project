"""Unit tests for the ingestion prep layer -- everything that converts the raw
SQLite row into a LangChain ``Document``, *before* any embedding or Chroma
call happens.

No network, no OpenAI, no Chroma. The functions under test are pure:

  * ``_clean``          (src/generate_embeddings.py)   -- string normalisation
  * ``build_document``  (src/generate_embeddings.py)   -- row -> Document glue
                                                          (metadata + text join)
  * ``_batched``        (src/retrieval/vector_store.py) -- upsert batching

``build_document`` only ever does ``row["col"]`` access, so a plain dict
stands in for a real ``sqlite3.Row`` -- no database needed.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.generate_embeddings import _clean, build_document
from src.retrieval.vector_store import _batched


# ---------------------------------------------------------------------------
# _clean -- empty/whitespace strings become None, everything else passes through
# ---------------------------------------------------------------------------


class TestClean:
    def test_whitespace_only_string_becomes_none(self):
        """Blank DB cells often come through as "" or "   "; those must become
        None so ``exclude_none`` can drop them from Chroma metadata."""
        assert _clean("   ") is None
        assert _clean("") is None

    def test_surrounding_whitespace_is_stripped(self):
        assert _clean("  Acme Roasters  ") == "Acme Roasters"

    def test_non_string_values_pass_through_unchanged(self):
        """Numeric columns (rating, aroma, ...) must not be touched."""
        assert _clean(8.5) == 8.5
        assert _clean(0) == 0
        assert _clean(None) is None


# ---------------------------------------------------------------------------
# build_document -- raw row (dict / sqlite3.Row)  ->  Document
#
# Covers both jobs the function now does inline (since to_document was removed):
#   1. metadata: ReviewMetadata -> Chroma-safe dict (no None, rich types -> str)
#   2. page_content: the desc_* fields joined into the passage that gets embedded
# ---------------------------------------------------------------------------


def _row(**overrides) -> dict:
    """A complete review row with sane defaults; override individual columns."""
    base = {
        "review_uid": "rev-200",
        "slug": "https://www.coffeereview.com/review/example/",
        "rating": 92,
        "roaster": "Acme Roasters",
        "name": "Yirgacheffe",
        "location": "Portland, OR",
        "origin": "Ethiopia",
        "roast": "Light",
        "est_price": "$18/12 oz",
        "review_date": "2024-02-10",
        "agtron": "58/72",
        "aroma": 9.0,
        "acid": 8.0,
        "body": 8.0,
        "flavor": 9.0,
        "aftertaste": 8.0,
        "desc_1": "Tasting notes here.",
        "desc_2_clean": "Sourcing context.",
        "desc_3": "Takeaway line.",
    }
    base.update(overrides)
    return base


class TestBuildDocumentMetadata:
    def test_returns_document_keyed_by_review_uid(self):
        """The Document id is the upsert/dedupe key -- must be the review_uid."""
        doc = build_document(_row(review_uid="rev-201"))

        assert isinstance(doc, Document)
        assert doc.id == "rev-201"
        assert doc.metadata["review_uid"] == "rev-201"

    def test_text_fields_are_cleaned_before_landing_in_metadata(self):
        """String columns pass through ``_clean``: padded -> trimmed,
        blank -> dropped from metadata entirely."""
        doc = build_document(_row(roaster="  Acme Roasters  ", origin="   "))

        assert doc.metadata["roaster"] == "Acme Roasters"
        assert "origin" not in doc.metadata

    def test_metadata_is_chroma_safe_even_for_a_sparse_row(self):
        """Chroma rejects None metadata values and non-primitive types.
        ``model_dump(mode="json", exclude_none=True)`` must guarantee:
          - no None values (unset optionals dropped)
          - every value is str / int / float / bool
          - rich types (date) are stringified
        """
        sparse = _row(
            slug=None, roaster=None, name=None, location=None, origin=None,
            roast=None, est_price=None, agtron=None,
            aroma=None, acid=None, body=None, flavor=None, aftertaste=None,
        )

        doc = build_document(sparse)

        assert doc.metadata["review_date"] == "2024-02-10"  # date -> str
        assert all(v is not None for v in doc.metadata.values())
        assert all(
            isinstance(v, (str, int, float, bool)) for v in doc.metadata.values()
        )


class TestBuildDocumentText:
    def test_page_content_joins_parts_in_fixed_order_with_single_space(self):
        """Order is desc_1, then desc_3, then desc_2_clean, joined by a single
        space. (This is the join that lives inside build_document now that
        combined_text / to_document were removed.)"""
        doc = build_document(
            _row(desc_1="A", desc_3="B", desc_2_clean="C")
        )

        assert doc.page_content == "A B C"

    def test_missing_description_parts_are_skipped_no_stray_separators(self):
        """A review missing desc_3 should not leave a doubled space."""
        doc = build_document(_row(desc_1="notes", desc_3=None, desc_2_clean="sourcing"))

        assert doc.page_content == "notes sourcing"

    def test_all_descriptions_missing_yields_empty_page_content(self):
        doc = build_document(_row(desc_1=None, desc_2_clean=None, desc_3=None))

        assert doc.page_content == ""


# ---------------------------------------------------------------------------
# _batched -- how upsert() chunks documents for the embedding API
# ---------------------------------------------------------------------------


class TestBatched:
    def test_exact_multiple_splits_evenly(self):
        assert list(_batched([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]

    def test_remainder_goes_in_a_short_final_batch(self):
        assert list(_batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    def test_empty_sequence_yields_nothing(self):
        assert list(_batched([], 3)) == []

    def test_batch_size_larger_than_sequence_yields_one_batch(self):
        assert list(_batched([1, 2], 10)) == [[1, 2]]
