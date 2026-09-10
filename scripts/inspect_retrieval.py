""" Inspect the retrieval system and scores produced by the vector store. """

from argparse import ArgumentParser

from coffee_rag.config import settings
from coffee_rag.retrieval.vector_store import VectorStore
from coffee_rag.schemas import ReviewMetadata
from coffee_rag.generation.prompting import _metadata_sentence
from coffee_rag.grading.reranker import Reranker


if __name__ == "__main__":
    parser = ArgumentParser(description="Inspect the retrieval system and scores produced by the vector store.")
    parser.add_argument("--query", help="The query to search for.")
    parser.add_argument("--k", type=int, default=5, help="The number of top-k candidate documents to retrieve.")
    args = parser.parse_args()

    # inspect topk retrieval candidates and their scores from the vector store
    vector_store = VectorStore()
    results = vector_store.get_topk_docs_with_scores(query=args.query, k=args.k)

    for doc, score in results:
        metadata_sentence = _metadata_sentence(ReviewMetadata(**doc.metadata))
        print(f"Score: {score}, Document: {doc.id} \n\nContent: {doc.page_content} \n\nMetadata Sentence: {metadata_sentence}\n")
        print("-" * 80)

    print("=" * 80)
    
    # inspect reranker results for the same query and retrieved documents
    reranker = Reranker()
    reranker_results = reranker.rerank_and_return_scores(
        query=args.query,
        docs=[doc for doc, _ in results],
        k=settings.RETRIEVAL_TOP_K
    )
    for doc, score, rank in reranker_results:
            metadata_sentence = _metadata_sentence(ReviewMetadata(**doc.metadata))
            print(f"Reranked Score: {score}, Document: {doc.id}, Initial Rank: {rank} \n\nContent: {doc.page_content} \n\nMetadata Sentence: {metadata_sentence}\n")
            print("-" * 80)
            