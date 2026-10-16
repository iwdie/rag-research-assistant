"""The retriever and the assistant: hybrid search, filters, rerank, citations."""

from __future__ import annotations

from app.core.filters import parse_filters
from app.services.assistant import RAGAssistant

TOPIC_QUERIES = [
    ("self attention heads attend to different relationships", "transformer"),
    ("inverse document frequency saturated term frequency", "bm25"),
    ("reverse process predicts the noise at each timestep", "diffusion"),
    ("server averages client models weighted by local examples", "federated"),
]


def test_each_query_retrieves_its_document(assistant: RAGAssistant):
    for query, expected_doc in TOPIC_QUERIES:
        results = assistant.search(query, top_k=3)
        assert results, f"no results for {query!r}"
        assert results[0].doc_id == expected_doc, f"{query!r} -> {[r.doc_id for r in results]}"


def test_hybrid_ranks_populate_both_retrievers(assistant: RAGAssistant):
    results = assistant.search("saturated term frequency", top_k=3)
    assert results
    assert all(set(chunk.ranks) == {"sparse", "dense", "fused"} for chunk in results)
    assert all(chunk.scores["sparse"] >= 0 and chunk.scores["dense"] >= 0 for chunk in results)


def test_reranking_attaches_scores_and_is_optional(assistant: RAGAssistant):
    reranked = assistant.search("attention heads", top_k=3, rerank_results=True)
    fused_only = assistant.search("attention heads", top_k=3, rerank_results=False)
    assert all(chunk.rerank_score is not None for chunk in reranked)
    assert all(chunk.rerank_score is None for chunk in fused_only)
    assert {c.chunk_id for c in reranked} == {c.chunk_id for c in fused_only}


def test_metadata_filter_restricts_documents(assistant: RAGAssistant):
    results = assistant.search("term frequency", top_k=5, filters=["category = retrieval"])
    assert results
    assert {chunk.metadata["category"] for chunk in results} == {"retrieval"}


def test_metadata_filter_with_no_matches_returns_nothing(assistant: RAGAssistant):
    assert assistant.search("term frequency", top_k=5, filters=["category = astronomy"]) == []


def test_doc_id_filter_selects_one_document(assistant: RAGAssistant):
    results = assistant.search("attention", top_k=5, filters={"doc_id": "transformer"})
    assert results
    assert {chunk.doc_id for chunk in results} == {"transformer"}


def test_year_range_filter(assistant: RAGAssistant):
    results = assistant.search("models", top_k=10, filters=["year >= 2017"])
    assert results
    assert all(chunk.metadata["year"] >= 2017 for chunk in results)


def test_debug_reports_pipeline_counts(assistant: RAGAssistant):
    assistant.search("attention heads", top_k=3)
    debug = assistant.retriever.last_debug
    assert debug is not None
    assert debug.fused_candidates >= 1
    assert debug.filtered_rows == assistant.store.n_chunks
    assert debug.top_k == 3
    assert debug.reranker == "lexical-coverage-proximity"


def test_filtered_debug_reports_the_reduced_row_count(assistant: RAGAssistant):
    assistant.search("term frequency", top_k=3, filters=["category = retrieval"])
    assert assistant.retriever.last_debug is not None
    assert assistant.retriever.last_debug.filtered_rows < assistant.store.n_chunks


def test_empty_query_returns_nothing(assistant: RAGAssistant):
    assert assistant.search("   ", top_k=3) == []


def test_query_returns_a_cited_answer(assistant: RAGAssistant):
    result = assistant.query("What controls how quickly term frequency saturates?")
    assert result.answer.answer
    assert result.answer.generator == "extractive"
    assert result.answer.citations
    assert result.answer.used_chunks
    assert result.answer.used_chunks[0] in {c.chunk_id for c in result.passages}
    assert "[1]" in result.answer.answer


def test_answer_admits_when_the_corpus_has_nothing(assistant: RAGAssistant):
    result = assistant.query("How do I proof sourdough bread overnight?")
    assert "No passage" in result.answer.answer
    assert result.answer.citations == []


def test_query_debug_is_included_only_on_request(assistant: RAGAssistant):
    assert assistant.query("attention", debug=False).debug is None
    assert assistant.query("attention", debug=True).debug is not None


def test_citations_point_at_real_chunks(assistant: RAGAssistant):
    result = assistant.query("How does the reverse diffusion process work?")
    for citation in result.answer.citations:
        assert assistant.store.get_chunk(citation.chunk_id) is not None
        assert citation.snippet


def test_query_result_serialises(assistant: RAGAssistant):
    payload = assistant.query("attention heads", debug=True).as_dict()
    for key in ("question", "answer", "citations", "passages", "generator", "debug"):
        assert key in payload
    assert payload["passages"][0]["chunk_id"]


def test_assistant_can_be_persisted_and_reloaded(settings, documents):
    assistant = RAGAssistant.build(settings)
    assistant.index_documents(documents)
    assistant.save()

    reloaded = RAGAssistant.load(settings)
    assert reloaded.store.n_chunks == assistant.store.n_chunks
    assert reloaded.query("term frequency saturation").answer.citations


def test_load_or_build_falls_back_to_the_sample_corpus(settings, sample_corpus):
    assistant = RAGAssistant.load_or_build(settings, sample_dir=sample_corpus)
    assert assistant.store.n_chunks > 0
    # The bootstrap index is written to disk, so the next start loads it.

    reloaded = RAGAssistant.load(settings)
    assert reloaded.store.n_chunks == assistant.store.n_chunks


def test_load_or_build_without_a_corpus_yields_an_empty_index(settings, tmp_path):
    assistant = RAGAssistant.load_or_build(settings, sample_dir=tmp_path / "absent")
    assert assistant.store.n_chunks == 0
    assert assistant.health()["status"] == "empty"
    assert "No passage" in assistant.query("anything at all").answer.answer


def test_indexing_keeps_the_retriever_in_sync(settings):
    from app.core.ingest import Document

    assistant = RAGAssistant.build(settings)
    assert assistant.search("anything") == []
    assistant.index_documents(
        [Document(doc_id="one", title="One", source="one.txt", pages=["retrieval fusion of sparse and dense lists"])]
    )
    assert assistant.search("sparse and dense lists")
    assert assistant.retriever.store is assistant.store


def test_removing_a_document_empties_the_hits(assistant: RAGAssistant):
    assistant.remove_document("bm25")
    results = assistant.search("inverse document frequency saturated", top_k=5)
    assert all(chunk.doc_id != "bm25" for chunk in results)


def test_parse_filters_accepts_the_api_shapes(assistant: RAGAssistant):
    assert assistant.search("term frequency", filters={"year": 1994})
    assert assistant.search("term frequency", filters=["year = 1994"])
    assert assistant.query("term frequency", filters={"category": "retrieval"}).filters == parse_filters(
        {"category": "retrieval"}
    )


def test_rerank_passages_is_a_no_op_without_a_reranker(settings, documents):
    settings = settings.model_copy(update={"rerank": False})
    assistant = RAGAssistant.build(settings)
    assistant.index_documents(documents)
    passages = assistant.search("attention", top_k=2, rerank_results=False)
    assert assistant.rerank_passages("attention", passages) == passages
