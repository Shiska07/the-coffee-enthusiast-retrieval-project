# Why Coffee + RAG?
I drink an unreasonable amount of coffee, and somewhere between my third and fourth cup of the day, being a "genuinely curious hobbyist" slowly turned into "I want to open a coffee shop someday."

This project is where that interest collided with something else I wanted to do: implement a production-focused sophisticated RAG (retrieval-augmented generation) by building something substantial, rather than following another toy tutorial on a dataset I don't care about. So instead of pointing a retrieval pipeline at Wikipedia or a random set of PDFs, I pointed it at a dataset of coffee reviews — something I already enjoy and want to learn more about.

Best case, I come out of this knowing a lot more about both good coffee and how retrieval pipelines actually work. Worst case, I become even more opinionated about coffee.

I'm building the project incrementally, one stage at a time, rather than trying to build the entire system at once. This README reflects what's actually implemented so far, followed by what I plan to add next.


# TheCoffeeEnthusiast

A retrieval-augmented question-answering system built over a dataset of coffee reviews.
Given a question about coffee — origin, roast, tasting notes, recommendations — the
system retrieves the most relevant reviews and generates an answer grounded in that
retrieved text, using a locally-hosted LLM.

This project is being built incrementally, one stage at a time.
This README reflects what's implemented so far, followed by what's planned next.

## Data Preparation for Retrieval

1. `notebooks/01_data_cleaning.ipynb` : cleans and normalizes the raw coffee review text
   (handling missing values, removing formatting noise, and concatenating fragmented
   descriptions into description text), because embedding quality — and therefore
   retrieval quality — directly depends on the quality of the text fed into it.
2. `notebooks/02_create_database.ipynb` : splits the cleaned data into a two-table
   relational schema (structured metadata such as rating, roaster, origin, and sensory
   scores in one table, narrative text in another, linked by a shared review ID) and
   loads both into a SQLite database, because separating structured fields from free
   text lets each be queried, filtered, or embedded independently without duplicating
   data.
3. `notebooks/03_text_analysis.ipynb` : reads the review text directly from the SQL
   database and analyzes token length distributions across the description fields to
   determine the chunking strategy for embedding, because chunk size directly affects
   retrieval quality (too short risks weak, low-signal embeddings; too long risks
   diluting the semantic signal of a single passage).

## Embeddings & Vector Store

4. `src/generate_embeddings.py` : builds the combined per-review text (per the chunking
   strategy decided above), embeds it with OpenAI's `text-embedding-3-small`, and writes
   it into a persistent Chroma vector store — each vector is keyed by `review_uid` so
   results can be joined back to the structured SQLite metadata at query time.

## Baseline Retrieval + Generation Pipeline

5. `src/retrieval/vector_store.py` : wraps the Chroma store with a similarity-search
   interface used at query time.
6. `src/generation/prompting.py` : builds the prompt sent to the LLM, combining each
   retrieved review's text with a natural-language sentence generated from its metadata
   (roaster, origin, roast, ratings), so the model reasons over more than a bare passage.
7. `src/generation/generator.py` : runs the prompt through a locally-hosted Mistral model
   (via Ollama) and returns a structured answer.

Together, steps 5–7 are the current end-to-end pipeline: a question comes in, the most
similar reviews are retrieved, and an answer is generated grounded only in that
retrieved context — no reranking, relevance grading, or retry logic yet. This
intentionally "boring" baseline exists as a working reference point that every later
improvement gets measured against.

## What's Next

Roughly in this order:

- an evaluation harness (RAGAS metrics against a curated question set) to measure
  retrieval and answer quality before adding more complexity
- alternate query construction strategies (multi-query, step-back, HyDE) to try when
  the first retrieval attempt comes up short
- a cross-encoder reranker and an LLM relevance grader, with a retry loop that switches
  strategies when retrieved context isn't good enough
- a web search fallback, domain-restricted to coffee-related sources, for questions the
  review dataset can't answer
- a hallucination/groundedness check on generated answers using a local classifier
  (Vectara HHEM), surfacing a low-confidence flag rather than retrying silently
- a simple chat interface wrapping the finished pipeline

## Built with
LangChain, LangGraph, Chroma, Ollama (Mistral), OpenAI embeddings.
