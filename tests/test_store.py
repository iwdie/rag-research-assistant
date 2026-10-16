"""Index persistence, idempotent ingestion and document removal."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.config import Settings
from app.services.store import IndexStore


def rebuild(settings: Settings) -> IndexStore:
    return IndexStore.from_settings(settings)


def test_add_documents_is_idempotent(settings: Settings, documents):
    store = rebuild(settings)
    first = store.add_documents(
        documents,
        chunk_words=settings.chunk_words,
        overlap_words=settings.chunk_overlap_words,
        min_words=settings.min_chunk_words,
    )
    second = store.add_documents(
        documents,
        chunk_words=settings.chunk_words,
        overlap_words=settings.chunk_overlap_words,
        min_words=settings.min_chunk_words,
    )
    assert first > 0
    assert second == 0
    assert len(store.documents) == len(documents)


def test_stats_describe_the_index(store: IndexStore):
    stats = store.stats()
    assert stats["n_documents"] == 4
    assert stats["n_chunks"] > 0
    assert stats["embedding_dim"] == store.embedder.dim
    assert stats["vector_backend"] == "numpy"
    assert sum(stats["chunks_per_document"].values()) == stats["n_chunks"]
    assert stats["bm25"]["vocabulary"] > 0


def test_filter_view_exposes_structural_and_document_fields(store: IndexStore):
    chunk = store.chunks[0]
    view = store.filter_view(0, chunk)
    for key in ("chunk_id", "doc_id", "page", "index", "category", "year"):
        assert key in view


def test_save_and_load_round_trip_preserves_search(settings: Settings, documents):
    store = rebuild(settings)
    store.add_documents(
        documents,
        chunk_words=settings.chunk_words,
        overlap_words=settings.chunk_overlap_words,
        min_words=settings.min_chunk_words,
    )
    store.save(settings.index_dir)

    assert (settings.index_dir / "meta.json").exists()
    assert (settings.index_dir / "chunks.jsonl").exists()

    reloaded = IndexStore.load(settings.index_dir, embedder=rebuild(settings).embedder)
    assert reloaded.n_chunks == store.n_chunks
    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in store.chunks]
    assert [c.text for c in reloaded.chunks] == [c.text for c in store.chunks]
    assert set(reloaded.documents) == set(store.documents)
    assert reloaded.documents["bm25"].title == "Okapi BM25 Probabilistic Ranking"

    query = "saturated term frequency factor"
    assert reloaded.bm25.search(query, k=3)[0][0] == store.bm25.search(query, k=3)[0][0]
    assert reloaded.search_dense(query, k=3)[0][0] == store.search_dense(query, k=3)[0][0]


def test_load_missing_index_raises(settings: Settings, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        IndexStore.load(tmp_path / "absent", embedder=rebuild(settings).embedder)


def test_load_rejects_a_different_embedder(settings: Settings, documents):
    store = rebuild(settings)
    store.add_documents(
        documents,
        chunk_words=settings.chunk_words,
        overlap_words=settings.chunk_overlap_words,
        min_words=settings.min_chunk_words,
    )
    store.save(settings.index_dir)

    wider = IndexStore.from_settings(Settings(index_dir=settings.index_dir, embedder_backend="fallback", hashing_dim=512))
    with pytest.raises(ValueError, match="dim"):
        IndexStore.load(settings.index_dir, embedder=wider.embedder)


def test_remove_document_rebuilds_both_indexes(store: IndexStore):
    before = store.n_chunks
    removed = store.remove_document("bm25")
    assert removed > 0
    assert store.n_chunks == before - removed
    assert "bm25" not in store.documents
    assert all(chunk.doc_id != "bm25" for chunk in store.chunks)
    assert store.get_chunk(store.chunks[0].chunk_id) is not None
    # BM25 IDF depends on corpus size, so the postings must have been rebuilt.
    assert store.bm25.n_docs == store.n_chunks
    assert store.vectors.size == store.n_chunks


def test_remove_unknown_document_is_a_no_op(store: IndexStore):
    assert store.remove_document("nope") == 0


def test_removing_every_document_leaves_an_empty_index(store: IndexStore):
    for doc_id in list(store.documents):
        store.remove_document(doc_id)
    assert store.n_chunks == 0
    assert store.vectors.size == 0
    assert store.bm25.search("anything", k=3) == []
    assert store.search_dense("anything", k=3) == []


def test_rebuild_reproduces_the_index(store: IndexStore):
    expected = store.search_dense("self attention heads", k=3)
    store.rebuild()
    assert store.search_dense("self attention heads", k=3) == expected


def test_get_chunk_unknown_id_returns_none(store: IndexStore):
    assert store.get_chunk("missing::0") is None


def test_list_documents_is_sorted_and_counted(store: IndexStore):
    records = store.list_documents()
    assert [record["title"] for record in records] == sorted(record["title"] for record in records)
    assert sum(record["chunk_count"] for record in records) == store.n_chunks


def test_documents_with_no_extractable_text_still_register(settings: Settings):
    from app.core.ingest import Document

    store = rebuild(settings)
    empty = Document(doc_id="empty", title="Empty", source="empty.txt", pages=[""])
    added = store.add_documents([empty], chunk_words=50, overlap_words=10, min_words=5)
    assert added == 0
    assert "empty" in store.documents
