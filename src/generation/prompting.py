"""
Prompt construction for the generation stage. Givena retreived document, we need to format it into a prompt that can be stuffed into the LLM's context window. This module also contains the system prompt and the chat 
prompt template used for generation. This part uses additional info from the metadata (review table with additonal structured info)
to provide more context to the generator.

"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.schemas import ReviewMetadata

SYSTEM_PROMPT = """

You are a knowledgeable coffee expert answering questions \
using ONLY the coffee review excerpts provided as context.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so explicitly \
rather than guessing.
- When relevant, mention which roaster(s) or coffee(s) support your answer.
- Keep your answer concise and directly responsive to the question.

"""

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)


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