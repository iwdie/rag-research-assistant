"""The HTTP surface: health, indexing, querying, filtering, evaluation, deletes."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.api.routes import DEFAULT_TESTSET
from app.config import Settings
from app.main import create_app
from fastapi.testclient import TestClient

SAMPLESET_EXISTS = DEFAULT_TESTSET.exists()


@pytest.fixture
def client(settings: Settings):
    """An isolated app: temporary index directory, no bundled sample corpus."""
    with TestClient(create_app(settings, sample_dir=None)) as test_client:
        yield test_client


@pytest.fixture
def sample_client(settings: Settings):
    """An app bootstrapped from the bundled sample corpus."""
    if not SAMPLESET_EXISTS:
        pytest.skip("eval/testset.jsonl has not been generated")
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def index_inline(client: TestClient, title: str, text: str, **metadata) -> dict:
    response = client.post(
        "/index/text",
        json={"title": title, "text": text, "metadata": metadata},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_root_serves_the_demo_ui(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Research Assistant" in response.text


def test_openapi_schema_is_available(client: TestClient):
    schema = client.get("/openapi.json").json()
    for path in ("/query", "/search", "/index/text", "/index/documents", "/evaluate", "/stats"):
        assert path in schema["paths"]


def test_health_reports_the_backends_in_use(client: TestClient):
    payload = client.get("/health").json()
    assert payload["status"] in {"ok", "empty"}
    assert payload["vector_backend"] in {"faiss", "numpy"}
    assert payload["fusion"] == "rrf"
    assert "embedder" in payload


def test_stats_match_the_health_endpoint(client: TestClient):
    health = client.get("/health").json()
    stats = client.get("/stats").json()
    assert stats["documents"] == health["documents"]
    assert stats["chunks"] == health["chunks"]
    assert stats["embedding_dim"] > 0
    assert stats["bm25"]["k1"] > 0


def test_index_text_then_query_it(client: TestClient):
    payload = index_inline(
        client,
        "Quantum Error Correction Notes",
        "Surface codes detect errors by measuring stabiliser operators on a lattice of "
        "physical qubits. A logical qubit is encoded in the topology so that local errors "
        "are correctable without disturbing the encoded information.",
        category="quantum",
        year=2023,
    )
    assert payload["chunks_added"] >= 1

    response = client.post("/query", json={"question": "How do surface codes detect errors?", "debug": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["citations"], body
    assert body["generator"] == "extractive"
    assert any("surface" in passage["text"].lower() for passage in body["passages"])
    assert body["debug"]["fused_candidates"] >= 1


def test_query_with_an_empty_corpus_explains_itself(settings: Settings, tmp_path: Path):
    # No sample corpus is loaded, so the index is empty and the answer must say so.
    empty = settings.model_copy(update={"index_dir": tmp_path / "nothing"})
    with TestClient(create_app(empty, sample_dir=None)) as isolated:
        assert isolated.get("/health").json()["status"] == "empty"
        response = isolated.post("/query", json={"question": "anything at all"})
        assert response.status_code == 200
        assert "No passage" in response.json()["answer"]


def test_query_rejects_an_empty_question(client: TestClient):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_validates_top_k(client: TestClient):
    assert client.post("/query", json={"question": "x", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"question": "x", "top_k": 500}).status_code == 422


def test_query_rejects_a_malformed_filter(client: TestClient):
    response = client.post("/query", json={"question": "x", "filters": ["nonsense clause"]})
    assert response.status_code == 400
    assert "cannot parse filter" in response.json()["detail"]


def test_metadata_filter_narrows_the_results(client: TestClient):
    index_inline(client, "Retrieval Alpha", "Hybrid retrieval fuses sparse and dense rankings.", category="retrieval")
    index_inline(client, "Vision Beta", "Hybrid training of vision transformers needs large datasets.", category="vision")

    filtered = client.post(
        "/search",
        json={"query": "hybrid", "top_k": 10, "filters": ["category = retrieval"]},
    ).json()
    assert filtered["filters"] == [{"field": "category", "op": "eq", "value": "retrieval"}]
    assert filtered["passages"]
    assert all(p["metadata"]["category"] == "retrieval" for p in filtered["passages"])


def test_search_debug_is_opt_in(client: TestClient):
    index_inline(client, "Debug Doc", "Debugging retrieval counts requires a debug flag.")
    without = client.post("/search", json={"query": "debug"}).json()
    withdebug = client.post("/search", json={"query": "debug", "debug": True}).json()
    assert without["debug"] is None
    assert withdebug["debug"]["top_k"] >= 1


def test_search_passages_carry_rank_provenance(client: TestClient):
    index_inline(client, "Ranks Doc", "Rank provenance shows where each passage came from.")
    passages = client.post("/search", json={"query": "rank provenance", "top_k": 3}).json()["passages"]
    assert passages
    assert set(passages[0]["ranks"]) == {"sparse", "dense", "fused"}


def test_reranking_can_be_disabled_per_request(client: TestClient):
    index_inline(client, "Rerank Doc", "Disabling reranking leaves the fused order in place.")
    mostly = client.post("/search", json={"query": "reranking", "rerank": False}).json()["passages"]
    assert all(p["rerank_score"] is None for p in mostly)


def test_documents_listing_and_deletion(client: TestClient):
    created = index_inline(client, "Deletable Doc", "This document exists to be deleted again.")
    doc_id = created["indexed"][0]

    listed = client.get("/documents").json()
    assert any(record["doc_id"] == doc_id for record in listed)

    removed = client.delete(f"/documents/{doc_id}")
    assert removed.status_code == 200
    assert removed.json()["chunks_removed"] >= 1

    assert all(record["doc_id"] != doc_id for record in client.get("/documents").json())
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_citation_endpoint_resolves_a_chunk(client: TestClient):
    index_inline(client, "Citation Doc", "Citation resolution returns the passage for a chunk id.")
    passages = client.post("/search", json={"query": "citation resolution"}).json()["passages"]
    chunk_id = passages[0]["chunk_id"]

    citation = client.get(f"/citations/{chunk_id}", params={"marker": 2}).json()
    assert citation["chunk_id"] == chunk_id
    assert citation["marker"] == 2
    assert citation["snippet"]


def test_citation_endpoint_404s_for_an_unknown_chunk(client: TestClient):
    assert client.get("/citations/nope::0").status_code == 404


def test_upload_indexes_supported_files(client: TestClient):
    response = client.post(
        "/index/documents",
        files={"files": ("methods.md", b"# Methods\n\nWe fine-tune with low rank adapters on frozen weights.", "text/markdown")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunks_added"] >= 1
    assert body["persisted"] is True


def test_upload_rejects_unsupported_types(client: TestClient):
    response = client.post(
        "/index/documents",
        files={"files": ("data.csv", b"a,b\n1,2", "text/csv")},
    )
    assert response.status_code == 200
    assert any("unsupported type" in entry for entry in response.json()["failed"])


def test_upload_rejects_an_empty_pdf(client: TestClient):
    response = client.post(
        "/index/documents",
        files={"files": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 200
    assert any("no extractable text" in entry for entry in response.json()["failed"])


@pytest.mark.skipif(not SAMPLESET_EXISTS, reason="eval/testset.jsonl has not been generated")
def test_evaluate_endpoint_scores_the_index(sample_client: TestClient):
    response = sample_client.post("/evaluate", json={"k_values": [1, 5], "compare": True})
    if response.status_code == 422:
        pytest.skip("resolved test set is empty against this index")
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["metrics"]["recall@5"] <= 1.0
    assert body["metrics"]["mrr"] > 0.0
    assert "sparse_bm25" in body["comparison"]
    assert body["coverage"]["n_queries"] >= 20


def test_evaluate_conflicts_on_an_empty_index(settings: Settings, tmp_path: Path):
    empty = settings.model_copy(update={"index_dir": tmp_path / "nothing"})
    with TestClient(create_app(empty, sample_dir=None)) as isolated:
        assert isolated.post("/evaluate", json={}).status_code == 409


def test_evaluate_missing_testset_is_404(client: TestClient):
    response = client.post("/evaluate", json={"testset": "definitely/not/here.jsonl"})
    assert response.status_code == 404


def test_unknown_route_is_404(client: TestClient):
    assert client.get("/nope").status_code == 404
