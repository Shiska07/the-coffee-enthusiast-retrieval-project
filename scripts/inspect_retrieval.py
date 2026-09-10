""" Inspect the retrieval system and scores produced by the vector store. """

from argparse import ArgumentParser
from coffee_rag.retrieval.vector_store import VectorStore
from coffee_rag.schemas import ReviewMetadata
from coffee_rag.generation.prompting import _metadata_sentence

if __name__ == "__main__":
    parser = ArgumentParser(description="Inspect the retrieval system and scores produced by the vector store.")
    parser.add_argument("--query", help="The query to search for.")
    parser.add_argument("--k", type=int, default=5, help="The number of top-k documents to retrieve.")
    args = parser.parse_args()

    vector_store = VectorStore()
    results = vector_store.get_topk_docs_with_scores(query=args.query, k=args.k)

    for doc, score in results:
        metadata_sentence = _metadata_sentence(ReviewMetadata(**doc.metadata))
        print(f"Score: {score}, Document: {doc.id} \n\nContent: {doc.page_content} \n\nMetadata Sentence: {metadata_sentence}\n")
        print("-" * 80)

