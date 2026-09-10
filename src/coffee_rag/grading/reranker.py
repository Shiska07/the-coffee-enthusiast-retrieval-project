""" Reranking logic for filtering and ordering retrieved documents. """

from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from coffee_rag.config import settings

class Reranker:
    def __init__(
        self,
        model: CrossEncoder | None = None,
    ):
        self.model = model or CrossEncoder(
            settings.RERANKER_MODEL,
            device=settings.RERANKER_DEVICE,
        )
        
    def rerank(self, query: str, docs: list[Document], k: int = settings.RETRIEVAL_TOP_K) -> list[Document]:
        """Rerank the retrieved documents based on their relevance to the query.

        Args:
            query (str): The user query.
            docs (list[Document]): The list of retrieved documents.
            k (int): The number of top documents to return.

        Returns:
            list[Document]: The reranked list of documents.
        """
        if not docs:
            return []

        # Prepare pairs of (query, document content) for scoring
        pairs = [(query, doc.page_content) for doc in docs]
        
        # Get relevance scores from the CrossEncoder model
        scores = self.model.predict(pairs)
        
        # Sort documents based on scores in descending order
        reranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        
        return reranked_docs[:k]  # Return top-k documents
    
    def rerank_and_return_scores(self, query: str, docs: list[Document], k: int = settings.RETRIEVAL_TOP_K) -> list[tuple[Document, float]]:
        """Rerank the retrieved documents based on their relevance to the query and return scores.

        Args:
            query (str): The user query.
            docs (list[Document]): The list of retrieved documents.
            k (int): The number of top documents to return.

        Returns:
            list[tuple[Document, float]]: The reranked list of documents with their scores.
        """
        if not docs:
            return []

        # Prepare pairs of (query, document content) for scoring
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        # (doc, rerank_score, initial_rank)  -- initial_rank 0 = top retrieval hit
        triples = zip(docs, scores, range(len(docs)))
        reranked = sorted(triples, key=lambda t: t[1], reverse=True)

        return [(doc, float(score), rank) for doc, score, rank in reranked[:k]]
    