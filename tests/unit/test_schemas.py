"""Unit tests for the pipeline's shared data structures (``src/schemas.py``).

These are pure Pydantic model tests: no Chroma, no LLM, no network. They lock
down the *contract* every pipeline stage relies on when it passes data to the
next stage:

  * ``ReviewMetadata`` -- the per-review metadata attached to every embedded
    passage. If this silently accepts bad data (or rejects good data), every
    downstream stage inherits the problem, so we test coercion and validation
    explicitly.
  * ``AnswerResult`` -- the object the generation stage returns. Its default
    values encode real behaviour (attempt counting, the web-search flag, etc.),
    so we pin those defaults here.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.schemas import AnswerResult, ReviewMetadata


# ---------------------------------------------------------------------------
# ReviewMetadata
# ---------------------------------------------------------------------------


class TestReviewMetadata:
    def test_minimal_valid_row_only_needs_review_uid(self):
        """``review_uid`` is the one required field; everything else is optional.

        A real ingestion row is often sparse (missing rating, agtron, sensory
        scores), so the model must accept a bare record without complaint.
        """
        m = ReviewMetadata(review_uid="rev-001")

        assert m.review_uid == "rev-001"
        assert m.rating is None
        assert m.aroma is None

    def test_missing_review_uid_is_rejected(self):
        """A record with no stable id can't be upserted/deduped -- must raise."""
        with pytest.raises(ValidationError):
            ReviewMetadata()  # no review-id passed

    def test_review_date_string_is_coerced_to_date(self):
        """CSV/JSON rows carry dates as strings; the model must parse them to
        ``datetime.date`` so downstream date filtering/sorting works."""
        m = ReviewMetadata(review_uid="rev-002", review_date="2024-03-15")

        assert m.review_date == date(2024, 3, 15)

    def test_numeric_sensory_scores_accept_int_and_str_forms(self):
        """Sensory columns are floats but can arrive as ints ("8") or
        numeric strings ("8.5"); Pydantic should coerce them to float."""
        m = ReviewMetadata(review_uid="rev-003", aroma=8, body="8.5")

        assert m.aroma == 8.0
        assert m.body == 8.5

    def test_non_numeric_sensory_score_is_rejected(self):
        """Garbage in a numeric field should fail loudly at ingestion time,
        not surface as a weird value three stages later."""
        with pytest.raises(ValidationError):
            ReviewMetadata(review_uid="rev-004", flavor="delicious")

    def test_invalid_slug_url_is_rejected(self):
        """``slug`` is typed ``HttpUrl`` -- a bare non-URL string must not pass."""
        with pytest.raises(ValidationError):
            ReviewMetadata(review_uid="rev-005", slug="not-a-url")

    def test_valid_slug_url_is_accepted(self):
        m = ReviewMetadata(
            review_uid="rev-006",
            slug="https://www.coffeereview.com/review/example/",
        )

        assert str(m.slug).startswith("https://www.coffeereview.com/")

    def test_model_dump_json_mode_is_chroma_safe(self):
        """Chroma metadata values must be primitives (str/int/float/bool) and
        must not be ``None``. ``to_document`` dumps with ``mode="json"`` and
        ``exclude_none=True``; this test pins the two properties that keeps
        the upsert from blowing up:

          1. unset optional fields are dropped entirely (no ``None`` values)
          2. rich types (``date``, ``HttpUrl``) become strings
        """
        m = ReviewMetadata(
            review_uid="rev-007",
            roaster="Acme Roasters",
            review_date=date(2024, 1, 2),
            slug="https://www.coffeereview.com/review/example/",
        )

        dumped = m.model_dump(mode="json", exclude_none=True)

        assert "rating" not in dumped  # unset -> absent, not None
        assert dumped["review_date"] == "2024-01-02"  # date -> str
        assert isinstance(dumped["slug"], str)  # HttpUrl -> str
        assert all(v is not None for v in dumped.values())
        assert all(isinstance(v, (str, int, float, bool)) for v in dumped.values())


# ---------------------------------------------------------------------------
# AnswerResult
# ---------------------------------------------------------------------------
class TestAnswerResult:
    def test_defaults_encode_first_attempt_no_fallback(self):
        """A freshly built answer (before any grading/retry/fallback) should
        report: 1 attempt, no web search, not low-confidence, raw query."""
        r = AnswerResult(question="What is a washed process?", answer="...")

        assert r.attempts == 1
        assert r.used_web_search is False
        assert r.low_confidence is False
        assert r.query_strategy == "raw"
        assert r.contexts == []


    def test_question_and_answer_are_required(self):
        with pytest.raises(ValidationError):
            AnswerResult(answer="orphan answer")  # type: ignore[call-arg]
