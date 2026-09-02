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

from src.config import settings
from src.generation.prompting import GENERATION_PROMPT, format_context_block
from src.schemas import AnswerResult

class Generator:
    def _init__(self, llm: ChatOllama | None = None) -> None:
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
        review text. 'format_context_block' builds a string combining the document's
        page_content with a natural-language sentence generated from its metadata.
        That combination is the full context fed per document to the LLM. For this case this is the coffee
        roast, level, scoring on acidity, flavor, etc. and other relevant information that can help the LLM answer the
        question more accurately.
        """
        context = "\n\n".join(format_context_block(doc) for doc in documents)
        result = self.chain.invoke({"question": question, "context": context})
        return AnswerResult(
            answer=result, 
            question=question, 
            contexts=documents
        )
    
    