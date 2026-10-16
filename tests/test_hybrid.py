"""Fusion rules and the rerankers."""

from __future__ import annotations

import pytest
from app.core.hybrid import RankedList, RetrievedChunk, fuse, linear_fusion, reciprocal_rank_fusion
from app.core.reranker import LexicalReranker, get_reranker, rerank


def chunk(chunk_id: str, text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id="d", text=text, score=score)


# -- fusion ------------------------------------------------------------------
def test_rrf_rewards_documents_that_appear_in_both_lists():
    sparse = RankedList("sparse", [("a", 9.0), ("b", 5.0)])
    dense = RankedList("dense", [("b", 0.9), ("c", 0.8)])
    fused = dict(reciprocal_rank_fusion([sparse, dense]))
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def test_rrf_ignores_raw_score_scale():
    # BM25 scores in the tens and cosine scores in [0, 1] must fuse by rank.
    sparse = RankedList("sparse", [("a", 42.0), ("b", 41.0)])
    dense = RankedList("dense", [("b", 0.99), ("a", 0.98)])
    fused = reciprocal_rank_fusion([sparse, dense])
    assert {chunk_id for chunk_id, _ in fused} == {"a", "b"}
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_rrf_weights_can_switch_off_a_retriever():
    sparse = RankedList("sparse", [("a", 1.0)], weight=1.0)
    dense = RankedList("dense", [("b", 1.0)], weight=0.0)
    assert [chunk_id for chunk_id, _ in reciprocal_rank_fusion([sparse, dense])] == ["a"]


def test_rrf_damping_constant_flattens_the_head():
    small = dict(reciprocal_rank_fusion([RankedList("s", [("a", 1.0), ("b", 1.0)])], k=1))
    large = dict(reciprocal_rank_fusion([RankedList("s", [("a", 1.0), ("b", 1.0)])], k=1000))
    assert small["a"] / small["b"] > large["a"] / large["b"]


def test_linear_fusion_keeps_confidence_information():
    sparse = RankedList("sparse", [("a", 10.0), ("b", 0.0)])
    dense = RankedList("dense", [("b", 1.0), ("a", 0.0)])
    fused = dict(linear_fusion([sparse, dense]))
    assert fused["a"] == pytest.approx(fused["b"])
    fused_weighted = dict(linear_fusion([RankedList("sparse", sparse.results, 3.0), dense]))
    assert fused_weighted["a"] > fused_weighted["b"]


def test_linear_fusion_handles_constant_scores():
    fused = linear_fusion([RankedList("s", [("a", 2.0), ("b", 2.0)])])
    assert dict(fused) == {"a": 1.0, "b": 1.0}


def test_fuse_dispatches_on_strategy():
    # rrf rewards agreement between the two retrievers; linear rewards a single
    # very strong score. On this input they disagree, so the dispatch is
    # observable rather than incidental.
    rankings = [RankedList("s", [("a", 3.0), ("b", 1.0)]), RankedList("d", [("b", 2.0), ("c", 1.0)])]
    # rrf: b = 1/62 + 1/61 beats a = 1/61 beats c = 1/62
    assert [chunk_id for chunk_id, _ in fuse(rankings, strategy="rrf")] == ["b", "a", "c"]
    # linear: a and b both normalise to 1.0, c to 0.0, so c sinks last
    assert [chunk_id for chunk_id, _ in fuse(rankings, strategy="linear")] == ["a", "b", "c"]


def test_fusion_of_empty_lists_is_empty():
    assert fuse([]) == []
    assert reciprocal_rank_fusion([RankedList("s", [])]) == []


# -- reranking ---------------------------------------------------------------
def test_lexical_reranker_prefers_clustered_terms():
    reranker = LexicalReranker()
    chunks = [
        chunk("scattered", "The parameter frequency of attention b appears near k but the words are spread " * 3),
        chunk("focused", "Term frequency saturation is controlled by the parameter k1 in Okapi BM25."),
    ]
    scores = reranker.score("what does the parameter k1 control in term frequency", chunks)
    assert scores[1] > scores[0]


def test_rerank_sorts_by_lexical_evidence_when_retrieval_ties():
    chunks = [
        chunk("off-topic", "gardening notes about soil and compost", 0.5),
        chunk("on-topic", "hybrid retrieval fuses sparse and dense rankings", 0.5),
    ]
    ordered = rerank(LexicalReranker(), "hybrid retrieval fuses sparse and dense rankings", chunks)
    assert ordered[0].chunk_id == "on-topic"
    assert all(item.rerank_score is not None for item in ordered)


def test_rerank_preserves_a_strong_retrieval_prior():
    # The reranker sharpens the retrieval order, it does not replace it: a small
    # lexical weight is what stops it from overriding the hybrid ranking with a
    # purely lexical guess. See LexicalReranker.weight.
    chunks = [
        chunk("retrieved", "gardening notes about soil and compost", 0.9),
        chunk("lexically-close", "hybrid retrieval fuses rankings", 0.1),
    ]
    ordered = rerank(LexicalReranker(), "hybrid retrieval fuses rankings", chunks)
    assert ordered[0].chunk_id == "retrieved"


def test_rerank_truncates_to_top_n():
    chunks = [chunk(f"c{i}", f"text {i}") for i in range(6)]
    assert len(rerank(LexicalReranker(), "text", chunks, top_n=2)) == 2


def test_rerank_on_empty_input_is_empty():
    assert rerank(LexicalReranker(), "anything", []) == []


def test_lexical_reranker_survives_an_empty_query():
    assert LexicalReranker().score("", [chunk("a", "text")]) == [0.0]


def test_rerank_is_stable_for_identical_chunks():
    chunks = [chunk("a", "same text", 0.5), chunk("b", "same text", 0.5)]
    ordered = rerank(LexicalReranker(), "same text", chunks)
    assert [item.chunk_id for item in ordered] == ["a", "b"]


def test_get_reranker_fallback_backend():
    assert isinstance(get_reranker(backend="fallback"), LexicalReranker)


def test_get_reranker_native_backend_loads_the_model():
    pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")
    reranker = get_reranker(backend="native", model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert reranker.name.endswith("ms-marco-MiniLM-L-6-v2")
