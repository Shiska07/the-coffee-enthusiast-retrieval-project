""" Visually Inspect the generator system and outputs produced by the language model. """

from argparse import ArgumentParser

from coffee_rag.config import settings
from coffee_rag.generation.prompting import build_context
from coffee_rag.retrieval.vector_store import VectorStore
from coffee_rag.generation.generator import Generator

if __name__ == "__main__":
    parser = ArgumentParser(description="Inspect the retrieval system and scores produced by the vector store.")
    parser.add_argument("--query", help="The query to search for.")
    parser.add_argument("--k", type=int, default=5, help="The number of top-k documents to retrieve.")
    args = parser.parse_args()

    # retreive relevant documents from the vector store
    vector_store = VectorStore()
    relevant_docs = vector_store.get_topk_docs(query=args.query, k=settings.RETRIEVAL_CANDIDATE_k)
    context = build_context(relevant_docs)

    # generate an answer using the retrieved context and the question
    generator = Generator()
    output = generator.generate(question=args.query, context=context)
    print(f"Generated Output: {output}")
    