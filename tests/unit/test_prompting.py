"""Unit tests for prompt construction (``src/coffee_rag/generation/prompting.py``).

String-building and token budgeting -- no LLM call.
These functions decide exactly what text reaches the generator, so the tests
target the ways that can go wrong:

  * a retrieved doc with no usable metadata (web-search results, random_sample,
    schema drift) still has to format without raising
  * partial metadata must not produce a dangling half-sentence
  * the natural-language join must not drop or duplicate an item
  * context packing keeps only whole documents that fit the token budget, and
    subtracts the fixed prompt overhead plus the question before packing any
  * PROMPT_OVERHEAD_TOKENS, computed once from the template, stays a sane value
    -- positive, larger than the system prompt alone, and small enough that a
    misconfigured CONTEXT_MAX_TOKENS still leaves room for documents

"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from coffee_rag.config import settings
from coffee_rag.generation import prompting
from coffee_rag.generation.prompting import (
    PROMPT_OVERHEAD_TOKENS,
    _join_natural,
    _metadata_sentence,
    build_context,
    count_tokens,
    format_context_block,
)
from coffee_rag.schemas import ReviewMetadata
from coffee_rag.templates import GENERATOR_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# _join_natural -- ", ".join with an Oxford "and" before the last item
# ---------------------------------------------------------------------------


class TestJoinNatural:
    def test_zero_items_is_empty_string(self):
        assert _join_natural([]) == ""

    def test_one_item_returned_as_is(self):
        assert _join_natural(["solo"]) == "solo"

    def test_two_items_joined_with_and_no_comma(self):
        assert _join_natural(["a", "b"]) == "a, and b"

    def test_three_items_get_oxford_comma(self):
        assert _join_natural(["a", "b", "c"]) == "a, b, and c"


# ---------------------------------------------------------------------------
# format_context_block -- Document -> string stuffed into the prompt
# ---------------------------------------------------------------------------


class TestFormatContextBlock:
    def test_doc_without_metadata_returns_page_content_unchanged(self):
        """Web-search docs and random_sample results can arrive with {} metadata.
        The function must fall back to bare page_content, not raise."""
        doc = Document(page_content="A washed Ethiopian with bright acidity.", metadata={})

        assert format_context_block(doc) == "A washed Ethiopian with bright acidity."

    def test_doc_with_unknown_metadata_keys_does_not_crash(self):
        """Chroma / schema drift can add keys ReviewMetadata doesn't declare.
        Extra keys should be ignored, not blow up construction."""
        doc = Document(
            page_content="passage",
            metadata={"review_uid": "rev-1", "roaster": "Acme", "some_new_field": 123},
        )

        out = format_context_block(doc)

        assert out.startswith("passage\n")
        assert "Acme" in out


# ---------------------------------------------------------------------------
# _metadata_sentence -- structured review fields -> one English sentence
# ---------------------------------------------------------------------------


class TestMetadataSentence:
    def test_none_metadata_yields_empty_string(self):
        assert _metadata_sentence(None) == ""

    def test_partial_profile_no_scores_has_no_ratings_clause(self):
        out = _metadata_sentence(
            ReviewMetadata(review_uid="rev-4", name="Yirgacheffe", origin="Ethiopia")
        )

        assert "Yirgacheffe" in out and "Ethiopia" in out
        assert "rating" not in out  # no rating/sub-scores -> no "It received ..."

    def test_sub_scores_are_formatted_to_one_decimal(self):
        """Scores may arrive as ints after a JSON round-trip; the '.1f' format
        must still apply (9 -> '9.0')."""
        out = _metadata_sentence(
            ReviewMetadata(review_uid="rev-5", aroma=9, acid=8.5)
        )

        assert "aroma 9.0/10" in out
        assert "acidity 8.5/10" in out


# ---------------------------------------------------------------------------
# build_context -- pack relevance-ordered docs into the token budget
#
# The reranker hands generate() a relevance-ordered list; some queries retrieve
# more text than the model's window holds. build_context must pack only WHOLE
# documents that fit -- never half a review.
#
# The budget is derived internally (CONTEXT_MAX_TOKENS - PROMPT_OVERHEAD_TOKENS
# - QUESTION_RESERVE_TOKENS). The fixture below stubs the counter to 1 token ==
# 1 char and zeroes the two fixed reservations, so a test can pin the effective
# document budget directly via CONTEXT_MAX_TOKENS.
# ---------------------------------------------------------------------------


def _doc(uid: str, text: str) -> Document:
    # metadata={} -> format_context_block returns page_content unchanged, so the
    # formatted block length == len(text). Keeps the budget math obvious.
    return Document(id=uid, page_content=text, metadata={})


@pytest.fixture
def set_budget(monkeypatch):
    monkeypatch.setattr(prompting, "count_tokens", len)
    monkeypatch.setattr(prompting, "PROMPT_OVERHEAD_TOKENS", 0)
    monkeypatch.setattr(settings, "QUESTION_RESERVE_TOKENS", 0)

    def _set(n_tokens: int) -> None:
        monkeypatch.setattr(settings, "CONTEXT_MAX_TOKENS", n_tokens)

    return _set


class TestBuildContext:
    def test_all_documents_fit_and_are_joined_with_blank_line(self, set_budget):
        set_budget(100)
        docs = [_doc("a", "AAAA"), _doc("b", "BBBB")]

        assert build_context(docs) == "AAAA\n\nBBBB"

    def test_lowest_ranked_documents_are_dropped_when_over_budget(self, set_budget):
        """Order is relevance-descending, so the tail is what gets cut."""
        set_budget(25)
        docs = [_doc("a", "A" * 10), _doc("b", "B" * 10), _doc("c", "C" * 10)]

        # "A"*10 + "\n\n"(2) + "B"*10 = 22 ok; +2+10 = 34 > 25 -> drop c
        context = build_context(docs)

        assert context == "A" * 10 + "\n\n" + "B" * 10
        assert "C" not in context

    def test_whole_documents_only_no_partial_text(self, set_budget):
        set_budget(14)
        docs = [_doc("a", "A" * 10), _doc("b", "B" * 10)]

        # "a" (10) fits; adding "\n\n"+"b" would hit 22 > 14 -> stop.
        # "b" must be entirely absent, not clipped.
        assert build_context(docs) == "A" * 10

    def test_top_document_is_kept_even_if_it_alone_exceeds_budget(self, set_budget):
        """Sending no context is worse than sending an over-long top hit."""
        set_budget(10)
        docs = [_doc("a", "A" * 100), _doc("b", "B" * 5)]

        assert build_context(docs) == "A" * 100

    def test_question_reserve_shrinks_the_effective_budget(self, set_budget, monkeypatch):
        set_budget(25)
        monkeypatch.setattr(settings, "QUESTION_RESERVE_TOKENS", 10)
        docs = [_doc("a", "A" * 10), _doc("b", "B" * 10)]

        # raw budget 25 fits both (22); reserving 10 for the question drops the
        # effective budget to 15 -> only "a" fits.
        assert build_context(docs) == "A" * 10

    def test_empty_document_list_yields_empty_context(self, set_budget):
        set_budget(100)

        assert build_context([]) == ""


# ---------------------------------------------------------------------------
# PROMPT_OVERHEAD_TOKENS -- fixed cost reserved before any document is packed
# ---------------------------------------------------------------------------


class TestPromptOverhead:
    """Derived from rendering GENERATION_PROMPT with empty context/question.
    A bad value silently starves the context budget, so pin that it's sane."""

    def test_overhead_is_positive(self):
        assert PROMPT_OVERHEAD_TOKENS > 0

    def test_overhead_exceeds_the_system_prompt_alone(self):
        """It's the system prompt PLUS the 'Context:/Question:' scaffolding, so
        it must be strictly larger than the system prompt by itself."""
        assert PROMPT_OVERHEAD_TOKENS > count_tokens(GENERATOR_SYSTEM_PROMPT)

    def test_overhead_leaves_a_usable_document_budget(self):
        """Guardrail against a misconfigured CONTEXT_MAX_TOKENS: the fixed
        overhead must not eat the whole window -- require room for a few
        reviews' worth of context (~1000 tokens) after subtracting it."""
        assert settings.CONTEXT_MAX_TOKENS - PROMPT_OVERHEAD_TOKENS >= 1000
