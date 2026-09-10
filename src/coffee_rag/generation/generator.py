"""
LLM Generation stage given the Question and the retrieved context documents.

Wraps ChatOllama (configured via GENERATION_* settings) into a single
Generator.generate(question, documents) -> AnswerResult call. Mirrors
VectorStore's pattern: the client is built once in __init__, not per call.    
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from coffee_rag.config import settings
from coffee_rag.generation.prompting import (
    GENERATION_PROMPT,
    PROMPT_OVERHEAD_TOKENS,
    build_context,
    count_tokens,
)
from coffee_rag.schemas import AnswerResult

class Generator:
    def __init__(self, llm: ChatOllama | None = None) -> None:
        self.llm = llm or ChatOllama(
        model=settings.GENERATION_MODEL,
            base_url=settings.GENERATION_BASE_URL,
            temperature=settings.GENERATION_TEMPERATURE,
            max_tokens=settings.GENERATION_MAX_TOKENS,
        )
        self.chain = GENERATION_PROMPT | self.llm | StrOutputParser()
        
    def generate(self, question: str, documents: list[Document]) -> AnswerResult:
        
        """
        Documents carry metadata that can give the LLM additional context beyond the raw
        review text. 'build_context' formats each document (page_content + a
        natural-language sentence built from its metadata: roast level, acidity /
        flavor scores, etc.) and packs as many WHOLE documents as fit the token
        budget (CONTEXT_MAX_TOKENS), dropping the lowest-ranked ones rather than truncating any.

        The budget is reduced up front by the fixed prompt overhead (system
        prompt + template scaffolding) and the question, so documents only
        compete for the space that actually remains.
        """
        context, used = build_context(
            documents,
            reserve_tokens=PROMPT_OVERHEAD_TOKENS + count_tokens(question),
        )
        result = self.chain.invoke({"question": question, "context": context})
        return AnswerResult(
            answer=result,
            question=question,
            contexts=used,
        )
    
    