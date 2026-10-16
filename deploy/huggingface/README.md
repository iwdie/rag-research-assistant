---
title: RAG Research Assistant
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
short_description: Hybrid BM25 + FAISS retrieval with cross-encoder reranking
---

# RAG Research Assistant

Hybrid retrieval (BM25 + FAISS) over research PDFs, fused with reciprocal rank
fusion, reranked with a cross-encoder, and answered with source citations.

- **Demo UI:** `/` · **API docs:** `/docs` · **Live backends:** `/health`
- **Source and full documentation:** https://github.com/iwdie/rag-research-assistant

This Space runs the full stack (Sentence Transformers + FAISS + cross-encoder).
Add a Space **Variable** named `WITH_MODELS` with value `true` and factory-reboot
to enable it; without it the Space builds the lighter fallback-only image.

Try questions such as:

- *Why does the Transformer need positional encodings?*
- *How does hybrid retrieval combine sparse and dense scores?*
- *What does the k1 parameter control in Okapi BM25?*
