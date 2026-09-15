"""
LLM Generation stage: given the question and an already-built context string,
run it through the prompt + model.

Wraps ChatOllama (configured via GENERATION_* settings) into a single
Generator.generate(question, context) -> AnswerResult call. Mirrors
VectorStore's pattern: the client is built once in __init__, not per call.
Context assembly (retrieval -> rerank -> build_context) happens upstream.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from coffee_rag.config import settings
from coffee_rag.templates import GENERATION_PROMPT

class Generator:
    def __init__(self, llm: ChatOllama | None = None) -> None:
        self.llm = llm or ChatOllama(
        model=settings.GENERATION_MODEL,
            base_url=settings.GENERATION_BASE_URL,
            temperature=settings.GENERATION_TEMPERATURE,
            max_tokens=settings.GENERATION_MAX_TOKENS,
        )
        self.chain = GENERATION_PROMPT | self.llm | StrOutputParser()
        
    def generate(self, question: str, context: str) -> str:
        """Run `question` + the pre-built `context` block through the prompt and
        model, returning the raw answer text. `context` is produced upstream by
        `build_context` (retrieval -> rerank -> pack to the token budget)."""
        return self.chain.invoke({"question": question, "context": context})
    