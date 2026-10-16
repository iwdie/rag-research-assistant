"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Passage(BaseModel):
    """A retrieved passage as returned to a client."""

    chunk_id: str
    doc_id: str
    title: str = ""
    page: int | None = None
    score: float
    ranks: dict[str, int] = Field(default_factory=dict)
    retriever_scores: dict[str, float] = Field(default_factory=dict)
    rerank_score: float | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    marker: int
    chunk_id: str
    doc_id: str
    title: str
    page: int | None = None
    snippet: str
    score: float


class QueryRequest(BaseModel):
    """A question plus optional narrowing and retrieval overrides."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | list[str] | None = Field(
        default=None,
        description=(
            "Metadata filter. Either {'field': value} for equality, or a list of "
            "clauses such as \"year >= 2020\" / \"doc_id in [a,b]\"."
        ),
    )
    rerank: bool | None = Field(
        default=None,
        description="Override cross-encoder reranking for this request.",
    )
    debug: bool = Field(default=False, description="Include per-stage retrieval counts.")


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    generator: str
    used_chunks: list[str] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    filters: list[dict[str, Any]] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    """Retrieval only - no answer synthesis, no LLM cost."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    candidate_k: int | None = Field(default=None, ge=1, le=500)
    filters: dict[str, Any] | list[str] | None = None
    rerank: bool | None = None
    debug: bool = False


class SearchResponse(BaseModel):
    query: str
    passages: list[Passage] = Field(default_factory=list)
    filters: list[dict[str, Any]] = Field(default_factory=list)
    debug: dict[str, Any] | None = None


class TextIndexRequest(BaseModel):
    """Inline document ingestion, for corpora that are not PDFs on disk."""

    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1)
    doc_id: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    indexed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    chunks_added: int
    n_documents: int
    n_chunks: int
    persisted: bool


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    source: str = ""
    n_pages: int = 0
    n_words: int = 0
    chunk_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    embedder: str
    embedding_dim: int
    vector_backend: Literal["faiss", "numpy"] | str
    reranker: str | None = None
    generator: str
    fusion: str
    fusion_weights: dict[str, float] = Field(default_factory=dict)
    bm25: dict[str, Any] = Field(default_factory=dict)
    total_words: int = 0
    chunks_per_document: dict[str, int] = Field(default_factory=dict)


class EvaluateRequest(BaseModel):
    """Run the labelled test set against the live index."""

    testset: str | None = Field(
        default=None,
        description="Path to a JSONL test set. Defaults to eval/testset.jsonl.",
    )
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    rerank: bool = True
    compare: bool = Field(
        default=True,
        description="Also report sparse-only and dense-only rankings for the same queries.",
    )


class EvaluationResponse(BaseModel):
    configuration: str
    n_queries: int
    metrics: dict[str, float]
    comparison: dict[str, dict[str, float]] = Field(default_factory=dict)
    coverage: dict[str, Any] = Field(default_factory=dict)
    failures: list[dict[str, Any]] = Field(default_factory=list)
