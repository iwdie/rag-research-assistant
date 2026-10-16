"""Vector store: exactness, filtering, persistence and FAISS/NumPy parity.

All vectors are unit-norm, so the FAISS ``IndexFlatIP`` path and the NumPy
dot-product path must return identical scores. That parity is asserted rather
than assumed - it is the property that lets the fallback be trusted in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from app.core.embeddings import HashingEmbedder, _l2_normalise
from app.core.vector_store import VectorStore


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


def unit(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _l2_normalise(rng.random((1, dim)))[0]


def test_nearest_neighbour_is_found():
    store = VectorStore(dim=8, use_faiss=False)
    vectors = np.stack([unit(seed) for seed in range(5)])
    store.add([f"c{i}" for i in range(5)], vectors)
    results = store.search(vectors[3], k=1)
    assert results[0][0] == "c3"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_scores_are_descending_and_capped_at_k():
    store = VectorStore(dim=8, use_faiss=False)
    store.add([f"c{i}" for i in range(6)], np.stack([unit(seed) for seed in range(6)]))
    results = store.search(unit(0), k=3)
    assert len(results) == 3
    assert [score for _, score in results] == sorted((s for _, s in results), reverse=True)


def test_allowed_indices_restrict_the_search():
    store = VectorStore(dim=8, use_faiss=False)
    store.add([f"c{i}" for i in range(5)], np.stack([unit(seed) for seed in range(5)]))
    results = store.search(unit(0), k=5, allowed_indices=[1, 2])
    assert len(results) == 2
    assert {chunk_id for chunk_id, _ in results} == {"c1", "c2"}


def test_empty_allow_list_returns_nothing():
    store = VectorStore(dim=8, use_faiss=False)
    store.add(["a"], np.stack([unit(1)]))
    assert store.search(unit(1), k=3, allowed_indices=[]) == []


def test_search_on_empty_store_is_safe():
    store = VectorStore(dim=8, use_faiss=False)
    assert store.search(unit(1), k=3) == []
    assert store.size == 0


def test_query_dimension_mismatch_raises():
    store = VectorStore(dim=8, use_faiss=False)
    store.add(["a"], np.stack([unit(1, 8)]))
    with pytest.raises(ValueError, match="expects 8"):
        store.search(unit(1, 4), k=1)


def test_add_rejects_mismatched_shapes():
    store = VectorStore(dim=8, use_faiss=False)
    with pytest.raises(ValueError, match="same length"):
        store.add(["a", "b", "c"], np.stack([unit(1), unit(2)]))
    with pytest.raises(ValueError, match="expected"):
        store.add(["a"], np.stack([unit(1, 4)]))
    with pytest.raises(ValueError, match="expected"):
        store.add(["a"], unit(1))


def test_save_and_load_round_trip(tmp_path: Path):
    store = VectorStore(dim=8, use_faiss=False)
    store.add([f"c{i}" for i in range(4)], np.stack([unit(seed) for seed in range(4)]))
    store.save(tmp_path)
    assert (tmp_path / "vectors.npy").exists()
    assert (tmp_path / "vector_ids.json").exists()

    reloaded = VectorStore(dim=8, use_faiss=False).load(tmp_path)
    assert reloaded.ids == store.ids
    assert reloaded.search(unit(2), k=2) == store.search(unit(2), k=2)


def test_load_rejects_embedder_dimension_change(tmp_path: Path):
    store = VectorStore(dim=8, use_faiss=False)
    store.add(["a"], np.stack([unit(1, 8)]))
    store.save(tmp_path)
    with pytest.raises(ValueError, match="dim"):
        VectorStore(dim=16, use_faiss=False).load(tmp_path)


def test_clear_and_rebuild():
    store = VectorStore(dim=8, use_faiss=False)
    store.add(["a"], np.stack([unit(1)]))
    store.clear()
    assert store.size == 0
    store.add(["b"], np.stack([unit(2)]))
    assert store.ids == ["b"]


@pytest.mark.parametrize("query_seed", [0, 1, 2])
def test_faiss_and_numpy_agree(query_seed: int):
    pytest.importorskip("faiss", reason="faiss-cpu is not installed")

    vectors = np.stack([unit(seed) for seed in range(8)])
    ids = [f"c{i}" for i in range(8)]
    query = unit(query_seed)

    numpy_store = VectorStore(dim=8, use_faiss=False)
    numpy_store.add(ids, vectors)
    faiss_store = VectorStore(dim=8, use_faiss=True)
    assert faiss_store.backend == "faiss"
    faiss_store.add(ids, vectors)

    numpy_results = numpy_store.search(query, k=4)
    faiss_results = faiss_store.search(query, k=4)
    assert [chunk_id for chunk_id, _ in numpy_results] == [chunk_id for chunk_id, _ in faiss_results]
    for (_, left), (_, right) in zip(numpy_results, faiss_results, strict=False):
        assert left == pytest.approx(right, abs=1e-5)


def test_faiss_fallback_path_matches_numpy():
    pytest.importorskip("faiss", reason="faiss-cpu is not installed")
    vectors = np.stack([unit(seed) for seed in range(6)])
    store = VectorStore(dim=8, use_faiss=True)
    store.add([f"c{i}" for i in range(6)], vectors)
    results = store.search(unit(3), k=2, allowed_indices=[0, 1, 3])
    assert all(chunk_id in {"c0", "c1", "c3"} for chunk_id, _ in results)
    assert results[0][0] == "c3"


def test_hashing_embedder_is_normalised_and_deterministic():
    embedder = HashingEmbedder(dim=128)
    first = embedder.encode(["hybrid retrieval combines bm25 and dense vectors", "unrelated text about gardening"])
    second = embedder.encode(["hybrid retrieval combines bm25 and dense vectors", "unrelated text about gardening"])
    assert first.shape == (2, 128)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(first, second)


def test_hashing_embedder_prefers_overlapping_text():
    embedder = HashingEmbedder(dim=512)
    vectors = embedder.encode(
        [
            "cross encoder reranking scores query passage pairs jointly",
            "cross encoder reranking scores a query and a passage together",
            "unrelated content about marine biology field work",
        ]
    )
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])
