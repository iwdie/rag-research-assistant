"""Chunking: stability, coverage, overlap and the hard-split guard."""

from __future__ import annotations

from itertools import pairwise

from app.core.chunking import Chunk, chunk_pages, chunk_text, split_sentences, word_count


def make_sentences(count: int, words: int = 12) -> list[str]:
    """Generate sentence-shaped units the chunker will actually split.

    Each unit starts with a capital and ends with a period, because the sentence
    splitter is deliberately conservative: a boundary is only recognised when
    the next unit starts with something that looks like the start of a sentence.
    Lowercase filler would be joined into one enormous unit and would test
    nothing.
    """
    return [f"S{index} " + " ".join(f"w{index}x{position}" for position in range(words - 2)) + "." for index in range(count)]


def test_split_sentences_keeps_headings_separate():
    text = "Introduction\nSelf-attention removes recurrence. It is fully parallel.\n3.2 Training\nA warmup schedule is used."
    units = split_sentences(text)
    assert "Introduction" in units
    assert "3.2 Training" in units
    assert "Self-attention removes recurrence." in units
    assert "It is fully parallel." in units


def test_split_sentences_handles_empty_text():
    assert split_sentences("") == []
    assert split_sentences("   \n\n ") == []


def test_chunk_ids_are_sequential_and_namespaced():
    text = " ".join(make_sentences(12))
    chunks = chunk_text(text, doc_id="paper-a", chunk_words=30, overlap_words=8, min_words=5)
    assert [chunk.chunk_id for chunk in chunks] == [f"paper-a::{i}" for i in range(len(chunks))]
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.doc_id == "paper-a" for chunk in chunks)


def test_every_sentence_survives_chunking():
    sentences = make_sentences(15)
    chunks = chunk_text(" ".join(sentences), doc_id="d", chunk_words=30, overlap_words=6, min_words=2)
    joined = " ".join(chunk.text for chunk in chunks)
    for sentence in sentences:
        assert sentence.rstrip(".") in joined


def test_chunks_overlap_between_neighbours():
    sentences = make_sentences(20)
    chunks = chunk_text(" ".join(sentences), doc_id="d", chunk_words=30, overlap_words=12, min_words=2)
    assert len(chunks) >= 3
    overlaps = 0
    for previous, current in pairwise(chunks):
        tokens_prev, tokens_cur = set(previous.text.split()), set(current.text.split())
        if tokens_prev & tokens_cur:
            overlaps += 1
    assert overlaps >= 1, "consecutive chunks share no tokens despite overlap_words > 0"


def test_chunk_size_respects_budget_for_normal_sentences():
    chunks = chunk_text(" ".join(make_sentences(20)), doc_id="d", chunk_words=30, overlap_words=10, min_words=2)
    # A chunk of packed sentences may exceed the budget by at most one sentence.
    assert max(chunk.n_words for chunk in chunks) <= 30 + 12


def test_oversized_sentence_is_hard_split():
    monster = " ".join(f"word{i}" for i in range(400)) + "."
    chunks = chunk_text(monster, doc_id="d", chunk_words=20, overlap_words=0, min_words=1)
    assert len(chunks) >= 5
    assert max(chunk.n_words for chunk in chunks) <= 40


def test_trailing_stub_is_merged_into_previous_chunk():
    sentences = make_sentences(4)
    chunks = chunk_text(" ".join(sentences), doc_id="d", chunk_words=20, overlap_words=0, min_words=40)
    # Four ~11-word sentences at chunk_words=20 give four chunks; the final
    # one is under min_words and is folded back into its predecessor.
    assert len(chunks) == 3
    assert sentences[-1].rstrip(".") in chunks[-1].text
    assert chunks[-1].n_words == word_count(sentences[-2]) + word_count(sentences[-1])


def test_chunk_pages_keeps_page_numbers_and_continuous_ordinals():
    pages = [" ".join(make_sentences(4)), " ".join(make_sentences(4)), " ".join(make_sentences(4))]
    chunks = chunk_pages(pages, doc_id="doc", chunk_words=20, overlap_words=0, min_words=1)
    assert {chunk.page for chunk in chunks} == {1, 2, 3}
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunk_record_round_trip():
    chunk = Chunk(
        chunk_id="d::0",
        doc_id="d",
        text="hello world",
        ordinal=0,
        page=2,
        title="Title",
        metadata={"year": 2024},
    )
    assert Chunk.from_record(chunk.to_record()) == chunk


def test_metadata_is_copied_not_shared():
    shared = {"year": 2020}
    chunks = chunk_text(" ".join(make_sentences(6)), doc_id="d", chunk_words=20, min_words=1, metadata=shared)
    chunks[0].metadata["extra"] = True
    assert "extra" not in chunks[-1].metadata
    assert "extra" not in shared


def test_word_count_is_whitespace_based():
    assert word_count("one two three") == 3
    assert word_count("") == 0
