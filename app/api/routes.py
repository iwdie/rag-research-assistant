"""HTTP routes.

Every handler is deliberately thin: resolve the assistant, translate the
request model into a call, shape the response. Anything that needs a decision
(fusion, filtering, ranking) lives in ``app.core`` so the API, the scripts and
the tests all exercise the same code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from app.api.schemas import (
    Citation,
    DocumentSummary,
    EvaluateRequest,
    EvaluationResponse,
    IndexResponse,
    Passage,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
    TextIndexRequest,
)
from app.core.filters import parse_filters
from app.core.hybrid import RetrievedChunk
from app.core.ingest import Document, ingest_paths, normalise_text
from app.evaluation.metrics import evaluate_ranking, index_coverage, load_testset, resolve_anchors
from app.services.assistant import RAGAssistant

logger = logging.getLogger(__name__)
router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTSET = REPO_ROOT / "eval" / "testset.jsonl"

# Uploads are size-capped in application code as well as at the proxy: a 500 MB
# "PDF" should be rejected before it is written to disk.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


def get_assistant(request: Request) -> RAGAssistant:
    assistant = getattr(request.app.state, "assistant", None)
    if assistant is None:  # pragma: no cover - only if startup failed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the index is still starting up",
        )
    return assistant


Assistant = Annotated[RAGAssistant, Depends(get_assistant)]


def _passage(chunk: RetrievedChunk) -> Passage:
    data = chunk.as_dict()
    return Passage(**data)


def _filter_echo(filters: Any) -> list[dict[str, Any]]:
    return [{"field": f.field, "op": f.op, "value": f.value} for f in parse_filters(filters)]


@router.get("/health", tags=["ops"])
def health(assistant: Assistant) -> dict[str, Any]:
    """Liveness plus the current corpus size and the backends actually in use."""
    return assistant.health()


@router.get("/stats", tags=["ops"], response_model=StatsResponse)
def stats(assistant: Assistant) -> StatsResponse:
    """Index-level statistics, including which vector backend is live."""
    payload = assistant.store.stats()
    return StatsResponse(
        documents=payload["n_documents"],
        chunks=payload["n_chunks"],
        embedder=payload["embedder"],
        embedding_dim=payload["embedding_dim"],
        vector_backend=payload["vector_backend"],
        reranker=assistant.reranker.name if assistant.reranker else None,
        generator=assistant.generator.name,
        fusion=assistant.settings.fusion,
        fusion_weights={"sparse": assistant.settings.sparse_weight, "dense": assistant.dense_weight},
        bm25=payload["bm25"],
        total_words=payload["total_words"],
        chunks_per_document=payload["chunks_per_document"],
    )


@router.get("/documents", tags=["index"], response_model=list[DocumentSummary])
def list_documents(assistant: Assistant) -> list[DocumentSummary]:
    return [DocumentSummary(**record) for record in assistant.store.list_documents()]


@router.post("/query", tags=["rag"], response_model=QueryResponse)
def query(payload: QueryRequest, assistant: Assistant) -> QueryResponse:
    """Answer a question from the indexed corpus, with citations."""
    try:
        result = assistant.query(
            payload.question,
            top_k=payload.top_k,
            filters=payload.filters,
            use_rerank=payload.rerank,
            debug=payload.debug,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return QueryResponse(**result.as_dict())


@router.post("/search", tags=["rag"], response_model=SearchResponse)
def search(payload: SearchRequest, assistant: Assistant) -> SearchResponse:
    """Retrieve passages without generating an answer."""
    try:
        passages = assistant.search(
            payload.query,
            top_k=payload.top_k,
            candidate_k=payload.candidate_k,
            filters=payload.filters,
            rerank_results=payload.rerank,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    debug = None
    if payload.debug and assistant.retriever.last_debug:
        debug = assistant.retriever.last_debug.as_dict()
    return SearchResponse(
        query=payload.query,
        passages=[_passage(chunk) for chunk in passages],
        filters=_filter_echo(payload.filters),
        debug=debug,
    )


@router.post("/index/text", tags=["index"], response_model=IndexResponse)
def index_text(payload: TextIndexRequest, assistant: Assistant) -> IndexResponse:
    """Index an inline document.

    Useful for smoke tests and for corpora that arrive as strings rather than
    files (notes, tickets, scraped sections).
    """
    import hashlib

    text = normalise_text(payload.text)
    doc_id = payload.doc_id or hashlib.blake2b(
        f"{payload.title}\n{text}".encode(), digest_size=8
    ).hexdigest()[:12]
    document = Document(
        doc_id=doc_id,
        title=payload.title,
        source=f"inline:{payload.title}",
        pages=[text],
        metadata={**payload.metadata, "source_name": f"{payload.title}.txt", "pages": 1},
    )
    added = assistant.index_documents([document])
    assistant.save()
    return IndexResponse(
        indexed=[doc_id] if added else [],
        failed=[] if added else [doc_id],
        chunks_added=added,
        n_documents=len(assistant.store.documents),
        n_chunks=assistant.store.n_chunks,
        persisted=True,
    )


@router.post("/index/documents", tags=["index"], response_model=IndexResponse)
async def index_documents(
    assistant: Assistant,
    files: Annotated[list[UploadFile], File(description="PDF, .txt or .md files")] = ...,
    persist: Annotated[bool, Query(description="Save the index to disk afterwards")] = True,
) -> IndexResponse:
    """Upload and index documents.

    Files land in ``<index_dir>/uploads`` so a re-upload after a restart is a
    hash hit and does not duplicate chunks.
    """
    upload_dir = Path(assistant.settings.index_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    failed: list[str] = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in {".pdf", ".txt", ".md", ".markdown"}:
            failed.append(f"{upload.filename}: unsupported type {suffix or '(none)'}")
            continue
        blob = await upload.read()
        if len(blob) > MAX_UPLOAD_BYTES:
            failed.append(f"{upload.filename}: exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
            continue
        target = upload_dir / Path(upload.filename or "upload").name
        target.write_bytes(blob)
        saved.append(target)

    documents = ingest_paths(saved)
    added = assistant.index_documents(documents)
    if added and persist:
        assistant.save()

    # Report per-file outcomes: a scanned PDF with no text layer is "failed"
    # from the caller's point of view even though the upload succeeded.
    ingested_names = {Path(document.source).name for document in documents}
    for upload in files:
        name = Path(upload.filename or "upload").name
        if name not in ingested_names and not any(entry.startswith(name) for entry in failed):
            failed.append(f"{name}: no extractable text")

    return IndexResponse(
        indexed=[document.doc_id for document in documents],
        failed=failed,
        chunks_added=added,
        n_documents=len(assistant.store.documents),
        n_chunks=assistant.store.n_chunks,
        persisted=bool(added and persist),
    )


@router.delete("/documents/{doc_id}", tags=["index"])
def delete_document(doc_id: str, assistant: Assistant, persist: bool = True) -> dict[str, Any]:
    """Remove a document and rebuild the indexes."""
    removed = assistant.remove_document(doc_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown document {doc_id}")
    if persist:
        assistant.save()
    return {
        "doc_id": doc_id,
        "chunks_removed": removed,
        "n_documents": len(assistant.store.documents),
        "n_chunks": assistant.store.n_chunks,
    }


@router.post("/evaluate", tags=["evaluation"], response_model=EvaluationResponse)
def evaluate(payload: EvaluateRequest, assistant: Assistant) -> EvaluationResponse:
    """Score the retrieval pipeline against a labelled test set.

    Runs the full hybrid pipeline, and by default the two single-retriever
    baselines over the same queries, so the delta from hybrid fusion and
    reranking is visible in one response.
    """
    path = Path(payload.testset) if payload.testset else DEFAULT_TESTSET
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"test set not found at {path}",
        )
    if assistant.store.n_chunks == 0:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="the index is empty")

    records = resolve_anchors(load_testset(path), assistant.store.chunks)
    coverage = index_coverage(assistant.store.chunks, records)
    scored = [record for record in records if record.get("relevant")]
    if not scored:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="no test record resolved to a gold chunk against this index",
        )

    k_values = sorted({k for k in payload.k_values if k > 0})
    max_k = max(k_values)
    queries = [record["query"] for record in scored]
    relevant_sets = [set(record["relevant"]) for record in scored]

    def ranking(record: dict[str, Any]) -> list[str]:
        filters = parse_filters(record.get("filters")) if record.get("filters") else None
        chunks = assistant.retriever.retrieve(
            record["query"], top_k=max_k, filters=filters, rerank_results=payload.rerank
        )
        return [chunk.chunk_id for chunk in chunks]

    hybrid = evaluate_ranking(
        "hybrid" + ("+rerank" if payload.rerank else ""),
        queries,
        relevant_sets,
        [ranking(record) for record in scored],
        k_values=k_values,
    )

    comparison: dict[str, dict[str, float]] = {}
    if payload.compare:
        sparse_lists: list[list[str]] = []
        dense_lists: list[list[str]] = []
        for record in scored:
            filters = parse_filters(record.get("filters")) if record.get("filters") else None
            allowed = assistant.retriever.allowed_rows(filters or [])
            sparse_lists.append(
                [chunk_id for chunk_id, _ in assistant.store.bm25.search(record["query"], k=max_k, allowed_indices=allowed)]
            )
            dense_lists.append(
                [chunk_id for chunk_id, _ in assistant.store.search_dense(record["query"], k=max_k, allowed_indices=allowed)]
            )
        comparison["sparse_bm25"] = evaluate_ranking(
            "sparse_bm25", queries, relevant_sets, sparse_lists, k_values=k_values
        ).metrics
        comparison["dense_only"] = evaluate_ranking(
            "dense_only", queries, relevant_sets, dense_lists, k_values=k_values
        ).metrics
        comparison["hybrid"] = hybrid.metrics

    return EvaluationResponse(
        configuration=hybrid.name,
        n_queries=hybrid.n_queries,
        metrics=hybrid.metrics,
        comparison=comparison,
        coverage=coverage,
        failures=hybrid.as_dict()["failures"][:20],
    )


@router.get("/citations/{chunk_id}", tags=["rag"], response_model=Citation)
def citation(chunk_id: str, assistant: Assistant, marker: int = 1) -> Citation:
    """Resolve a chunk id back to a citable passage."""
    chunk = assistant.store.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown chunk {chunk_id}")
    from app.core.generator import build_citations
    from app.core.hybrid import RetrievedChunk

    synthesized = RetrievedChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        score=1.0,
        page=chunk.page,
        title=chunk.title,
        metadata=chunk.metadata,
    )
    built = build_citations([synthesized], limit=1)[0]
    built.marker = marker
    return Citation(**built.as_dict())
