"""Phase 1 — persistent Chroma vector store + similarity-search interface.

Built on `langchain_chroma.Chroma` so retrieval speaks LangChain's `Document`
type end-to-end, interoperating directly with the rest of the LangChain/
LangGraph ecosystem this project uses elsewhere.

`generate_embeddings.py` writes into this store via `VectorStore.upsert`;
every later phase reads from it through `VectorStore.similarity_search`,
which returns `list[tuple[Document, float]]` — score travels alongside each
Document rather than inside it (see schemas.py for why).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.schemas import ReviewMetadata


def to_document(text: str, metadata: ReviewMetadata) -> Document:
    """Build a LangChain Document from a review passage + already-validated review metadata."""
    return Document(
        id=metadata.review_uid,
        page_content=text,
        metadata=metadata.model_dump(mode="json", exclude_none=True),
    )

def _batched(seq: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


class VectorStore:
    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str | None = None,
        embeddings: OpenAIEmbeddings | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir or settings.VECTOR_STORE_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
        self._store = self._build_store()

    def _build_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate the collection — backs the --rebuild flag."""
        self._store.delete_collection()
        self._store = self._build_store()

    def count(self) -> int:
        # Chroma exposes no public count(); .get(include=[]) fetches ids only
        # (cheap) and avoids reaching into the private ._collection attribute.
        return len(self._store.get(include=[])["ids"])

    def upsert(self, documents: Sequence[Document], batch_size: int | None = None) -> None:
        """Embed and upsert documents. Idempotent on each Document's `id`."""
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        for batch in _batched(documents, batch_size):
            self._store.add_documents(documents=list(batch), ids=[d.id for d in batch])

    def similarity_search(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        k = k or settings.RETRIEVAL_CANDIDATE_k
        return self._store.similarity_search_with_relevance_scores(query, k=k)
    