"""Shared fixtures.

Tests run with the *fallback* backends by default (hashing embeddings, lexical
reranker, NumPy vectors). That keeps the suite offline, deterministic and fast,
and it means a failure is a bug in this repository rather than a flaky model
download. Tests that specifically cover the FAISS and Sentence Transformers code
paths opt in explicitly and skip when the dependency is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings, get_settings  # noqa: E402
from app.core.ingest import Document  # noqa: E402
from app.services.assistant import RAGAssistant  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """The settings object is cached for the process; tests must not inherit it."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline, deterministic settings pointed at a temporary index directory."""
    return Settings(
        index_dir=tmp_path / "index",
        embedder_backend="fallback",
        hashing_dim=256,
        reranker_backend="fallback",
        use_faiss=False,
        generator="extractive",
        chunk_words=90,
        chunk_overlap_words=20,
        candidate_k=20,
        top_k=5,
    )


@pytest.fixture
def documents() -> list[Document]:
    """A tiny corpus with unambiguous topics for retrieval assertions."""
    specs = [
        (
            "transformer",
            "The Transformer: Self-Attention",
            "ranking",
            2017,
            [
                "Self-attention lets every position attend to every other position in a single step, "
                "which removes the sequential dependency of recurrent networks.",
                "Multi-head attention projects queries, keys and values into several subspaces so that "
                "different heads can attend to different kinds of relationships.",
            ],
        ),
        (
            "bm25",
            "Okapi BM25 Probabilistic Ranking",
            "retrieval",
            1994,
            [
                "Okapi BM25 scores a document by summing an inverse document frequency weight multiplied "
                "by a saturated term frequency factor over the query terms.",
                "The parameter k1 controls how quickly term frequency saturates, and the parameter b "
                "controls how strongly long documents are penalised.",
            ],
        ),
        (
            "diffusion",
            "Denoising Diffusion Probabilistic Models",
            "generative vision",
            2020,
            [
                "The reverse diffusion process is trained to predict the noise that was added at each "
                "timestep, which makes the objective a weighted mean squared error.",
                "Sampling starts from pure Gaussian noise and repeatedly applies the learned denoising "
                "step to produce an image.",
            ],
        ),
        (
            "federated",
            "Communication-Efficient Federated Learning",
            "distributed systems",
            2017,
            [
                "Each round samples a subset of clients and every selected client performs several epochs "
                "of local gradient descent on its own private data.",
                "The server averages the resulting client models, weighted by the number of local "
                "examples, to form the next global model.",
            ],
        ),
    ]
    return [
        Document(
            doc_id=key,
            title=title,
            source=f"{key}.txt",
            pages=[" ".join(paragraphs)],
            metadata={"category": category, "year": year, "key": key},
        )
        for key, title, category, year, paragraphs in specs
    ]


@pytest.fixture
def store(settings: Settings, documents: list[Document]):
    """An in-memory index built from the fixture corpus."""
    from app.services.store import IndexStore

    index = IndexStore.from_settings(settings)
    index.add_documents(
        documents,
        chunk_words=settings.chunk_words,
        overlap_words=settings.chunk_overlap_words,
        min_words=settings.min_chunk_words,
    )
    return index


@pytest.fixture
def assistant(settings: Settings, documents: list[Document]) -> RAGAssistant:
    """A full assistant over the fixture corpus, with nothing persisted."""
    instance = RAGAssistant.build(settings)
    instance.index_documents(documents)
    return instance


@pytest.fixture
def sample_settings(tmp_path: Path) -> Settings:
    """The shipped production defaults, pointed at a temporary index directory.

    The sample-corpus evaluation tests must run the *documented* configuration,
    otherwise the numbers asserted here would not be the numbers in the README.
    """
    return Settings(
        index_dir=tmp_path / "sample-index",
        embedder_backend="fallback",
        reranker_backend="fallback",
        use_faiss=False,
        chunk_words=180,
        chunk_overlap_words=40,
        candidate_k=50,
        top_k=5,
    )


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """Generate the bundled synthetic corpus into a temporary directory."""
    import importlib

    module = importlib.import_module("scripts.make_sample_data")
    corpus_dir = tmp_path / "sample"
    testset = tmp_path / "testset.jsonl"
    exit_code = module.main(["--corpus-dir", str(corpus_dir), "--testset", str(testset)])
    assert exit_code == 0
    return corpus_dir


@pytest.fixture
def sample_testset(tmp_path: Path, sample_corpus: Path) -> Path:
    return tmp_path / "testset.jsonl"
