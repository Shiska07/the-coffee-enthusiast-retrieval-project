"""
Prompt construction for the generation stage. Givena retreived document, we need to format it into a prompt that can be stuffed into the LLM's context window. This module also contains the system prompt and the chat 
prompt template used for generation. This part uses additional info from the metadata (review table with additonal structured info)
to provide more context to the generator.

"""

import tiktoken
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from coffee_rag.config import settings
from coffee_rag.schemas import ReviewMetadata

# Mistral tokenises with SentencePiece, not cl100k, so this is an approximation
# (typically within ~10-15% for English prose) used only for budgeting.
_ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))

from coffee_rag.templates import GENERATION_PROMPT

# Fixed token cost of the prompt itself -- SYSTEM_PROMPT plus the
PROMPT_OVERHEAD_TOKENS = count_tokens(GENERATION_PROMPT.format(context="", question=""))


def format_context_block(doc: Document) -> str:
    """Render a retrieved Document for stuffing into the generation prompt."""
    meta = ReviewMetadata(**doc.metadata) if doc.metadata else None
    sentence = _metadata_sentence(meta)
    return f"{doc.page_content}\n{sentence}" if sentence else doc.page_content


def _metadata_sentence(m: ReviewMetadata | None) -> str:
    """E.g. 'Ethiopia Yirgacheffe is roasted by Blue Bottle, a light roast,
    sourced from Ethiopia. It received an overall rating of 92/100, and
    tasting notes scored aroma 9.0/10, acidity 8.5/10, body 8.0/10,
    flavor 9.0/10, and aftertaste 8.5/10.'
    """
    if not m:
        return ""

    subject = m.name or "This coffee"
    profile = []
    if m.roaster:
        profile.append(f"roasted by {m.roaster}")
    if m.roast:
        profile.append(f"a {m.roast.lower()} roast")
    if m.origin:
        profile.append(f"sourced from {m.origin}")

    sentence = subject
    if profile:
        sentence += " is " + _join_natural(profile)
    sentence += "."
        
    if m.slug:
        sentence += f" Url: {m.slug}."
    
    if m.est_price:
        sentence += f" Estimated price: {m.est_price}."

    ratings = []
    if m.rating is not None:
        ratings.append(f"an overall rating of {m.rating}/100")

    sub_scores = [
        f"{label} {value:.1f}/10"
        for label, value in (
            ("aroma", m.aroma),
            ("acidity", m.acid),
            ("body", m.body),
            ("flavor", m.flavor),
            ("aftertaste", m.aftertaste),
        )
        if value is not None
    ]
    
    if sub_scores:
        ratings.append("tasting notes scored " + _join_natural(sub_scores))

    if ratings:
        sentence += " It received " + _join_natural(ratings) + "."
        
    return sentence


def _join_natural(items: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b, and c'; handles 1 and 2 items too."""
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + ", and " + items[-1]


def build_context(documents: list[Document]) -> str:
    """Format `documents` into the prompt's context block, keeping only WHOLE
    documents that fit the token budget.

    `documents` is assumed to be pre-sorted by relevance (reranker output), so
    we fill greedily from the top and stop before the first document that would
    push us over budget. A document is never split, and the top-ranked document
    is always kept even if it alone exceeds the budget (no context is worse).

    Budget = `CONTEXT_MAX_TOKENS`, minus the fixed prompt overhead (system
    prompt + template scaffolding) and `QUESTION_RESERVE_TOKENS` held back for
    the question -- both known ahead of time, so callers pass only the docs.
    """
    budget = (
        settings.CONTEXT_MAX_TOKENS
        - PROMPT_OVERHEAD_TOKENS
        - settings.QUESTION_RESERVE_TOKENS
    )

    blocks: list[str] = []
    used = 0
    sep_cost = 0  # no separator before the first block
    for doc in documents:
        block = format_context_block(doc)
        cost = sep_cost + count_tokens(block)
        if blocks and used + cost > budget:
            break
        blocks.append(block)
        used += cost
        sep_cost = count_tokens("\n\n")

    return "\n\n".join(blocks)
