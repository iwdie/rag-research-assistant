"""BM25: ranking sanity, IDF stability, filtered search and persistence."""

from __future__ import annotations

import pytest
from app.core.bm25 import BM25Index, tokenize

DOCS = [
    ("d1", "okapi bm25 scores documents with a saturated term frequency factor and inverse document frequency"),
    ("d2", "the transformer uses multi head self attention over every position in the sequence"),
    ("d3", "diffusion models learn to reverse a gradual noising process with gaussian noise"),
    ("d4", "federated learning averages locally computed client updates on private data"),
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index().fit([doc_id for doc_id, _ in DOCS], [text for _, text in DOCS])


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("The effect of the query on the index") == ["effect", "query", "index"]
    assert tokenize("The effect of the query", keep_stopwords=True) == ["the", "effect", "of", "the", "query"]


def test_relevant_document_ranks_first(index: BM25Index):
    results = index.search("saturated term frequency factor", k=3)
    assert results
    assert results[0][0] == "d1"
    assert results[0][1] > 0.0


def test_scores_are_ordered_descending(index: BM25Index):
    results = index.search("attention position sequence", k=4)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_query_with_no_matching_terms_returns_nothing(index: BM25Index):
    assert index.search("zygote photosynthesis", k=5) == []


def test_idf_never_goes_negative(index: BM25Index):
    # "the" style common terms used to be able to produce negative IDF; the
    # floor guarantees a term that appears everywhere still contributes.
    results = index.search("client data private locally", k=4)
    assert all(score > 0 for _, score in results)


def test_filtered_search_only_scores_allowed_rows(index: BM25Index):
    allowed = [index.ids.index("d3"), index.ids.index("d4")]
    results = index.search("gaussian noise process", k=5, allowed_indices=allowed)
    assert results
    assert {doc_id for doc_id, _ in results} <= {"d3", "d4"}


def test_empty_allow_list_returns_nothing(index: BM25Index):
    assert index.search("attention", k=5, allowed_indices=[]) == []


def test_extend_adds_searchable_documents(index: BM25Index):
    index.extend(["d5"], ["reinforcement learning from human feedback optimises a learned reward model"])
    results = index.search("reward model human feedback", k=3)
    assert results[0][0] == "d5"
    assert index.n_docs == 5


def test_empty_index_is_safe():
    empty = BM25Index().fit([], [])
    assert empty.search("anything", k=5) == []
    assert empty.n_docs == 0


def test_state_round_trip_rebuilds_identical_scores(index: BM25Index):
    state = index.to_state()
    rebuilt = BM25Index.from_state(state, [text for _, text in DOCS])
    assert rebuilt.search("saturated term frequency", k=3) == index.search("saturated term frequency", k=3)


def test_length_normalisation_changes_the_ranking():
    # The long document repeats the query term, so with b=0 it wins on raw term
    # frequency; with b=1 the length penalty hands the top slot to the short one.
    docs = [
        ("short", "gaussian noise process"),
        ("long", "gaussian gaussian gaussian noise " + "filler " * 200),
    ]
    ids = [doc_id for doc_id, _ in docs]
    texts = [text for _, text in docs]
    b_off = BM25Index(k1=1.5, b=0.0).fit(ids, texts)
    b_on = BM25Index(k1=1.5, b=1.0).fit(ids, texts)
    assert b_off.search("gaussian", k=1)[0][0] == "long"
    assert b_on.search("gaussian", k=1)[0][0] == "short"
    assert b_off.avgdl == b_on.avgdl


def test_vocabulary_and_avgdl(index: BM25Index):
    assert index.vocabulary == len(set(tokenize(" ".join(text for _, text in DOCS))))
    assert index.avgdl > 0
