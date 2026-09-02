"""Central configuration for the coffee RAG pipeline.

Values are read from environment variables / the
project `.env` file (see `.env.example`), with defaults so the pipeline
runs without any `.env` for everything except the API keys.

Import the singleton:

    from src.config import settings
    settings.SQLITE_PATH
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory that contains `src/`, `db/`, `data/`, ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):    
    model_config = SettingsConfigDict(
        env_file= PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # --- paths ---
    PROJECT_ROOT: Path = PROJECT_ROOT
    SQLITE_PATH: Path = PROJECT_ROOT / "db" / "coffee_reviews.db"  # path to the SQLite database file
    VECTOR_STORE_DIR: Path = PROJECT_ROOT / "vector_store"  # path to the vector store directory
    '''
    Add evals and test directories to the project root.
    '''
    
    # --- Embeddings ---------------------------------------------------------
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # 1536-dim, ~$0.02/1M tokens
    EMBEDDING_BATCH_SIZE: int = 256
    CHROMA_COLLECTION: str = "coffee_reviews"
    
    
    # --- Retrieval ---------------------------------------------------------
    RETRIEVAL_CANDIDATE_k: int = 20  # number of candidate documents to retrieve for each query (gets re-ranked by cross-encoder)
    RETRIEVAL_SCORE_THRESHOLD: float = 0.75 # pre-filter cutoff on raw similarity score (0-1) for candidate documents before re-ranking
    RETRIEVAL_TOP_K: int = 5  # number of top documents to retrieve for each query post re-ranking
    MAX_RETREVAL_ATTEMPTS: int = 3  # number of attempts to retrieve documents before giving up
    
    # --- Relevance Scoring/Grading ---------------------------------------------------------
    # Stage 1 (primary signal): cross-encoder reranks RETRIEVAL_CANDIDATE_K candidates down
    # to RETRIEVAL_TOP_K. English-only, purpose-trained on MS MARCO, ~22.7M params — cheap enough to run on CPU for every query.
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    RERANKER_DEVICE: str = "cpu"
    
    # Stage 2 (optional corrective gate): LLM binary pass/fail grade on the reranked top-k, run on
    # GENERATION_MODEL with structured output (relevant / not relevant per chunk).
    RELEVANCE_GRADE_PASS_RATIO: float = 0.5  # >= half of top-k graded relevant = pass
    
    # --- Web search fallback ---------------------------------------------------------
    TAVILY_API_KEY: str | None = Field(default=None)
    WEB_SEARCH_INCLUDE_DOMAINS: list[str] = [
        "coffeereview.com",
        "sca.coffee",
        "perfectdailygrind.com",
        "sprudge.com",
        "jameshoffmann.co.uk",
    ]
    WEB_SEARCH_MAX_RESULTS: int = 3
    
    # --- Generation ---------------------------------------------------------
    GENERATION_PROVIDER: str = "ollama"
    GENERATION_MODEL: str = "mistral:7b-instruct-q4_K_M"
    GENERATION_BASE_URL: str = "http://localhost:11434"
    GENERATION_TEMPERATURE: float = 0.1
    GENERATION_MAX_TOKENS: int = 512
    
    # --- Hallucination detection -----------------------------------------------------
    # Primary: local download, free, prupose-built factual-consistency classifier (Vectara HHEM-2.1).
    HALLUCINATION_MODEL: str = "vectara/hallucination_evaluation_model"
    HALLUCINATION_SCORE_THRESHOLD: float = 0.65  # below this -> flag answer low-confidence
    
    # Optional tie-breaker: only invoked when HHEM's score falls in the ambiguous band
    # below, not on every request. Requires ANTHROPIC_API_KEY.
    HALLUCINATION_TIEBREAKER_ENABLED: bool = False
    HALLUCINATION_TIEBREAKER_MODEL: str = "claude-haiku-4-5"
    HALLUCINATION_AMBIGUOUS_LOW: float = 0.3
    HALLUCINATION_AMBIGUOUS_HIGH: float = 0.7
    ANTHROPIC_API_KEY: str | None = Field(default=None)
    
    # --- Logging & Tracing ---------------------------------------------------------
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = Field(default=None)
    LANGSMITH_PROJECT: str = "coffee-rag"
    
 
settings = Settings()
   
def combined_text(desc_1: str | None, desc_3: str | None, desc_2_clean: str | None) -> str:
    """Build the single passage embedded per review.

    Chunking strategy decided in ``notebooks/03_text_analysis.ipynb``: concatenate
    all three description fields into one passage. Order matches the notebook
    (``desc_1`` tasting notes, ``desc_3`` takeaway, ``desc_2_clean`` sourcing/
    business context).
    """
    parts = [(desc_1 or "").strip(), (desc_3 or "").strip(), (desc_2_clean or "").strip()]
    return "\n\n".join(p for p in parts if p)

    