"""Metrics, anchor resolution and an end-to-end evaluation over the sample corpus."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from app.config import Settings
from app.evaluation.metrics import (
    DEFAULT_K_VALUES,
    average_precision,
    evaluate_ranking,
    evaluate_retriever,
    hit_at_k,
    index_coverage,
    load_testset,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    resolve_anchors,
)
from app.services.assistant import RAGAssistant

RETRIEVED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d"}


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == 0.0
    assert recall_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(0.5)
    assert recall_at_k(RETRIEVED, RELEVANT, 4) == 1.0
    assert recall_at_k(RETRIEVED, set(), 4) == 0.0


def test_precision_and_hit_rate():
    assert precision_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(0.5)
    assert precision_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(0.5)
    assert precision_at_k(RETRIEVED, RELEVANT, 0) == 0.0
    assert hit_at_k(RETRIEVED, RELEVANT, 1) == 0.0
    assert hit_at_k(RETRIEVED, RELEVANT, 3) == 1.0


def test_mrr_uses_the_first_relevant_rank():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == pytest.approx(0.5)
    assert reciprocal_rank(RETRIEVED, {"a"}) == 1.0
    assert reciprocal_rank(RETRIEVED, {"zzz"}) == 0.0


def test_average_precision():
    # hits at ranks 2 and 4 -> (1/2 + 2/4) / min(2, 4)
    assert average_precision(RETRIEVED, RELEVANT, 4) == pytest.approx(0.5)
    assert average_precision(RETRIEVED, set(), 4) == 0.0


def test_ndcg_matches_hand_computed_values():
    def discount(rank: int) -> float:
        return 1 / math.log2(rank + 1)

    # k=2: only the rank-2 hit counts; the ideal has hits at ranks 1 and 2.
    assert ndcg_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(
        discount(2) / (discount(1) + discount(2)), abs=1e-9
    )
    # k=4: hits at ranks 2 and 4, ideal at ranks 1 and 2.
    assert ndcg_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(
        (discount(2) + discount(4)) / (discount(1) + discount(2)), abs=1e-9
    )
    # A ranking that puts both hits first is perfect.
    assert ndcg_at_k(["b", "d", "a"], RELEVANT, 3) == pytest.approx(1.0, abs=1e-9)
    assert ndcg_at_k(RETRIEVED, set(), 4) == 0.0


def test_ndcg_rewards_better_ordering():
    good = ndcg_at_k(["b", "x", "d"], RELEVANT, 3)
    bad = ndcg_at_k(["x", "b", "d"], RELEVANT, 3)
    assert good > bad


def test_evaluate_ranking_aggregates_means():
    report = evaluate_ranking(
        "test",
        ["q1", "q2"],
        [{"b", "d"}, {"a"}],
        [["a", "b", "c", "d"], ["a", "z"]],
        k_values=(1, 2, 4),
    )
    assert report.n_queries == 2
    # q1 finds both of its two gold chunks, q2 finds its single gold chunk.
    assert report.metrics["recall@4"] == pytest.approx(1.0)
    assert report.metrics["recall@2"] == pytest.approx((0.5 + 1.0) / 2)
    assert report.metrics["mrr"] == pytest.approx((0.5 + 1.0) / 2)
    assert report.metrics["hit@1"] == pytest.approx(0.5)
    assert "map@4" in report.metrics
    assert len(report.per_query) == 2


def test_evaluate_ranking_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate_ranking("bad", ["q"], [{"a"}], [])


def test_report_table_and_dict():
    report = evaluate_ranking("cfg", ["q"], [{"a"}], [["a"]], k_values=(1,))
    assert "cfg" in report.table()
    payload = report.as_dict()
    assert payload["n_queries"] == 1
    assert payload["metrics"]["recall@1"] == 1.0


def test_evaluate_retriever_skips_unresolvable_records(assistant: RAGAssistant):
    testset = [
        {"query": "self attention heads", "relevant": []},
        {"query": "saturated term frequency", "relevant": []},
    ]
    report = evaluate_retriever(assistant.retriever, testset, k_values=(1, 5))
    assert report.n_queries == 0


def test_resolve_anchors_maps_sentences_to_chunks(assistant: RAGAssistant):
    gold = assistant.store.chunks[0]
    sentence = gold.text.split(".")[0] + "."
    records = resolve_anchors([{"query": "q", "anchor": sentence}], assistant.store.chunks)
    assert records[0]["relevant"] == [gold.chunk_id]


def test_resolve_anchors_marks_stale_labels_as_unresolved(assistant: RAGAssistant):
    records = resolve_anchors([{"query": "q", "anchor": "a sentence that is definitely not present"}], assistant.store.chunks)
    assert records[0]["relevant"] == []


def test_resolve_anchors_respects_an_existing_relevant_list(assistant: RAGAssistant):
    records = resolve_anchors([{"query": "q", "relevant": ["already::0"]}], assistant.store.chunks)
    assert records[0]["relevant"] == ["already::0"]


def test_load_testset_reads_jsonl_and_json(tmp_path: Path):
    jsonl = tmp_path / "t.jsonl"
    jsonl.write_text('{"query": "a", "anchor": "b"}\n// comment\n{"query": "c", "anchor": "d"}\n', encoding="utf-8")
    assert [record["query"] for record in load_testset(jsonl)] == ["a", "c"]

    payload = tmp_path / "t.json"
    payload.write_text(json.dumps({"queries": [{"query": "x"}]}), encoding="utf-8")
    assert load_testset(payload)[0]["query"] == "x"


def test_index_coverage_counts_labels(assistant: RAGAssistant):
    coverage = index_coverage(assistant.store.chunks, [{"relevant": ["a::0", "b::0"]}, {"relevant": []}])
    assert coverage["n_chunks"] == assistant.store.n_chunks
    assert coverage["n_labels"] == 2
    assert coverage["unresolved_queries"] == 1
    assert coverage["avg_gold_per_query"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# End-to-end: the shipped corpus and the shipped test set must work together.
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_assistant(sample_settings: Settings, sample_corpus: Path) -> RAGAssistant:
    return RAGAssistant.build(sample_settings, paths=[sample_corpus])


def test_shipped_testset_resolves_against_the_shipped_corpus(
    sample_assistant: RAGAssistant, sample_testset: Path
):
    records = load_testset(sample_testset)
    assert len(records) >= 20
    resolved = resolve_anchors(records, sample_assistant.store.chunks)

    unresolved = [record["query"] for record in resolved if not record.get("relevant")]
    assert unresolved == [], f"stale anchors in the shipped test set: {unresolved}"
    # Every record is labelled either with one anchor or with a coverage list.
    assert all(record.get("anchor") or record.get("anchors") for record in resolved)
    coverage = [record for record in resolved if len(record["relevant"]) > 1]
    assert coverage, "the test set has no multi-passage queries to measure coverage with"


def test_sample_corpus_retrieval_quality(sample_assistant: RAGAssistant, sample_testset: Path):
    """Guard against silent retrieval regressions.

    The thresholds are floors with headroom, not the measured values; see
    ``eval/README.md`` for the numbers this pipeline actually produces.
    """
    records = resolve_anchors(load_testset(sample_testset), sample_assistant.store.chunks)
    report = evaluate_retriever(
        sample_assistant.retriever,
        records,
        name="hybrid+rerank",
        k_values=(1, 5, 10),
    )
    assert report.n_queries >= 20
    assert report.metrics["recall@5"] > 0.8
    assert report.metrics["recall@10"] > 0.9
    assert report.metrics["mrr"] > 0.6


def test_hybrid_beats_or_matches_the_sparse_baseline(
    sample_assistant: RAGAssistant, sample_testset: Path
):
    records = resolve_anchors(load_testset(sample_testset), sample_assistant.store.chunks)
    queries = [record["query"] for record in records]
    relevant_sets = [set(record["relevant"]) for record in records]

    hybrid = evaluate_retriever(sample_assistant.retriever, records, k_values=(5,))
    sparse = evaluate_ranking(
        "sparse",
        queries,
        relevant_sets,
        [
            [chunk_id for chunk_id, _ in sample_assistant.store.bm25.search(query, k=5)]
            for query in queries
        ],
        k_values=(5,),
    )
    assert hybrid.metrics["recall@5"] >= sparse.metrics["recall@5"]


def test_default_k_values_are_sane():
    assert list(DEFAULT_K_VALUES) == sorted(DEFAULT_K_VALUES)
    assert min(DEFAULT_K_VALUES) >= 1


def test_setting_can_disable_reranking(settings: Settings, documents):
    no_rerank = settings.model_copy(update={"rerank": False, "reranker_backend": "fallback"})
    assistant = RAGAssistant.build(no_rerank)
    assistant.index_documents(documents)
    assert assistant.reranker is None
    results = assistant.search("saturated term frequency", top_k=3)
    assert results
    assert all(chunk.rerank_score is None for chunk in results)
