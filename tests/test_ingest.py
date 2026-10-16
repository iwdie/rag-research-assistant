"""Ingestion: normalisation, sidecar metadata, directory walking and PDFs."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.ingest import (
    chunk_documents,
    expand_paths,
    ingest_paths,
    load_document,
    load_text,
    normalise_text,
)


def test_normalise_repairs_hyphenated_line_breaks():
    assert "transformer" in normalise_text("the trans-\nformer model")
    assert "cooperate" in normalise_text("co-\noperate")


def test_normalise_strips_soft_hyphens_and_curly_quotes():
    text = normalise_text("re\u00adsearch with \u201cquotes\u201d and it\u2019s fine")
    assert "research" in text
    assert '"quotes"' in text
    assert "it's" in text


def test_normalise_collapses_blank_line_runs():
    assert "\n\n\n" not in normalise_text("a\n\n\n\n\nb")


def test_load_text_splits_on_page_markers(tmp_path: Path):
    path = tmp_path / "pages.txt"
    path.write_text("first page\n---\nsecond page\n---\nthird page", encoding="utf-8")
    assert load_text(path) == ["first page", "second page", "third page"]


def test_document_metadata_from_sidecar_and_filename(tmp_path: Path):
    path = tmp_path / "vaswani_2017_attention.txt"
    path.write_text("Body text about self attention.", encoding="utf-8")
    (tmp_path / "vaswani_2017_attention.txt.meta.json").write_text('{"venue": "NeurIPS"}', encoding="utf-8")

    document = load_document(path)
    assert document.metadata["venue"] == "NeurIPS"
    assert document.metadata["year"] == 2017  # recovered from the filename
    assert document.title == "Vaswani 2017 Attention"


def test_explicit_metadata_overrides_the_sidecar(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text("Some text.", encoding="utf-8")
    (tmp_path / "doc.txt.meta.json").write_text('{"year": 1999}', encoding="utf-8")
    document = load_document(path, metadata={"year": 2024})
    assert document.metadata["year"] == 2024


def test_malformed_sidecar_is_ignored(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text("Some text.", encoding="utf-8")
    (tmp_path / "doc.txt.meta.json").write_text("{not json", encoding="utf-8")
    assert load_document(path).metadata.get("year") is None


def test_doc_id_is_stable_and_content_addressed(tmp_path: Path):
    first = tmp_path / "a" / "paper.txt"
    second = tmp_path / "b" / "paper.txt"
    for path in (first, second):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Identical content about retrieval.", encoding="utf-8")
    assert load_document(first).doc_id == load_document(second).doc_id

    other = tmp_path / "c" / "paper.txt"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("Different content entirely.", encoding="utf-8")
    assert load_document(other).doc_id != load_document(first).doc_id


def test_unsupported_suffix_raises(tmp_path: Path):
    path = tmp_path / "sheet.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_document(path)


def test_expand_paths_walks_directories_and_filters_suffixes(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    for name in ("a.txt", "b.md"):
        (tmp_path / name).write_text("text", encoding="utf-8")
    (tmp_path / "sub" / "c.txt").write_text("text", encoding="utf-8")
    (tmp_path / "binary.png").write_bytes(b"\x89PNG")
    found = {path.name for path in expand_paths([tmp_path])}
    assert found == {"a.txt", "b.md", "c.txt"}


def test_expand_paths_warns_on_missing_input(tmp_path: Path):
    assert expand_paths([tmp_path / "nope.pdf"]) == []


def test_ingest_paths_reports_documents(tmp_path: Path):
    for name in ("one.txt", "two.txt"):
        (tmp_path / name).write_text(f"content of {name}", encoding="utf-8")
    documents = ingest_paths([tmp_path])
    assert len(documents) == 2
    assert all(document.pages for document in documents)


def test_chunk_documents_propagates_metadata(tmp_path: Path):
    path = tmp_path / "paper.txt"
    path.write_text(" ".join(f"word{i}" for i in range(400)), encoding="utf-8")
    (tmp_path / "paper.txt.meta.json").write_text('{"category": "nlp"}', encoding="utf-8")
    documents = ingest_paths([tmp_path])
    chunks = chunk_documents(documents, chunk_words=50, overlap_words=10, min_words=5)

    assert len(chunks) > 1
    assert all(chunk.metadata["category"] == "nlp" for chunk in chunks)
    assert all(chunk.metadata["doc_title"] == documents[0].title for chunk in chunks)
    assert all(chunk.doc_id == documents[0].doc_id for chunk in chunks)


def test_scanned_pdf_with_no_text_is_handled(tmp_path: Path):
    reportlab = pytest.importorskip("reportlab", reason="reportlab is not installed")
    from reportlab.pdfgen import canvas

    path = tmp_path / "blank.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.showPage()
    pdf.save()
    document = load_document(path)
    assert document.pages == [""] or document.pages == []
    assert reportlab is not None


def test_pdf_extraction_keeps_page_boundaries(tmp_path: Path):
    pytest.importorskip("reportlab", reason="reportlab is not installed")
    from reportlab.pdfgen import canvas

    path = tmp_path / "two_pages.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, "First page discusses hybrid retrieval with BM25.")
    pdf.showPage()
    pdf.drawString(72, 720, "Second page discusses cross encoder reranking.")
    pdf.save()

    document = load_document(path)
    assert len(document.pages) == 2
    assert "hybrid retrieval" in document.pages[0]
    assert "reranking" in document.pages[1]
    assert document.metadata["pages"] == 2


def test_document_text_joins_pages():
    from app.core.ingest import Document

    document = Document(doc_id="x", title="T", source="s", pages=["one", "two"])
    assert document.text == "one\n\ntwo"
    assert document.n_pages == 2
