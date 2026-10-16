"""Central configuration.

Everything tunable lives here and can be overridden with ``RAG_``-prefixed
environment variables (or a local ``.env``). The defaults are chosen so that
``pip install -r requirements.txt && uvicorn app.main:app`` works on a laptop
with no GPU, no model download and no cloud account.

The heavier components (Sentence Transformers, FAISS, the cross-encoder) are
opt-in at runtime rather than hard requirements, because a reviewer should be
able to clone this repo and run the test suite in seconds. Each backend has a
deterministic pure-Python/NumPy fallback, and every fallback is reported in
``GET /stats`` so a result is never silently produced by a weaker code path.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Backend = Literal["auto", "native", "fallback"]


class Settings(BaseSettings):
    """Runtime settings for ingestion, retrieval, reranking and generation."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- storage -----------------------------------------------------------
    index_dir: Path = Field(
        default=Path("data/index"),
        description="Directory holding the persisted index (chunks, vectors, stats).",
    )

    # ---- chunking ----------------------------------------------------------
    chunk_words: int = Field(default=180, ge=20)
    chunk_overlap_words: int = Field(default=40, ge=0)
    min_chunk_words: int = Field(
        default=25,
        ge=1,
        description="Chunks shorter than this are merged into the previous one.",
    )

    # ---- embeddings --------------------------------------------------------
    embedder_backend: Backend = Field(
        default="auto",
        description="auto = Sentence Transformers if importable, else hashing fallback.",
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hashing_dim: int = Field(
        default=512,
        ge=64,
        description="Vector width of the dependency-free hashing embedder.",
    )
    embedding_batch_size: int = Field(default=32, ge=1)

    # ---- vector index ------------------------------------------------------
    use_faiss: bool = Field(
        default=True,
        description="Use FAISS when importable; NumPy brute force otherwise.",
    )

    # ---- BM25 --------------------------------------------------------------
    bm25_k1: float = Field(default=1.5, gt=0.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    # ---- hybrid fusion -----------------------------------------------------
    fusion: Literal["rrf", "linear"] = "rrf"
    rrf_k: int = Field(default=60, gt=0, description="RRF damping constant.")
    dense_weight: float = Field(default=1.0, ge=0.0)
    sparse_weight: float = Field(default=1.0, ge=0.0)
    lexical_dense_weight: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Ceiling on the dense weight when the embedder is the dependency-free hashing "
            "proxy. The proxy is a lexical model, so its ranking is strongly correlated with "
            "BM25's, and reciprocal rank fusion assumes the inputs contribute independent "
            "evidence: at equal weights two correlated rankings can outvote the better "
            "calibrated sparse one. Measured on eval/testset.jsonl, capping the proxy at 0.5 "
            "takes recall@1 from 0.786 to 0.810 and MRR from 0.925 to 0.937. A real "
            "bi-encoder is not correlated with BM25 and keeps the full weight, so this only "
            "ever applies to the fallback. It is a ceiling, not a multiplier: a lower "
            "RAG_DENSE_WEIGHT always wins."
        ),
    )

    # ---- retrieval / reranking --------------------------------------------
    candidate_k: int = Field(default=50, ge=1, description="Candidates per retriever.")
    top_k: int = Field(default=5, ge=1, description="Passages returned to the caller.")
    rerank: bool = True
    reranker_backend: Backend = "auto"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- answer generation -------------------------------------------------
    generator: Literal["extractive", "openai"] = "extractive"
    max_context_chunks: int = Field(default=5, ge=1)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    request_timeout: float = Field(default=30.0, gt=0.0)

    # ---- lifecycle ---------------------------------------------------------
    persist_on_shutdown: bool = Field(
        default=True,
        description="Save the index when the server shuts down.",
    )

    @property
    def use_sentence_transformers(self) -> bool:
        return self.embedder_backend in {"auto", "native"}

    @property
    def use_cross_encoder(self) -> bool:
        return self.reranker_backend in {"auto", "native"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor so API handlers and scripts agree on one object."""
    return Settings()
