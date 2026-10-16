# Single image for the API, built so the container can answer a question on
# first boot with no model download and no network access at runtime.
#
# WITH_MODELS=false produces a ~200 MB image that runs the NumPy + hashing
# fallbacks; the default installs FAISS, torch and Sentence Transformers, warms
# both model caches at build time, and pre-builds the sample index with the same
# embedder the runtime uses - mismatched embedding dimensions between a baked
# index and the configured model is the classic way this kind of service fails
# to start, so the build does it once, deliberately.
FROM python:3.12-slim

ARG WITH_MODELS=true
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependency layer first: application edits then re-use the installed wheels.
COPY requirements.txt requirements-vector.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && if [ "$WITH_MODELS" = "true" ]; then pip install -r requirements-vector.txt; fi

COPY app ./app
COPY scripts ./scripts
COPY eval ./eval
COPY data/sample ./data/sample

# Warm the model caches during the build so the runtime container never needs
# outbound network. Kept in one RUN so the (large) cache layer is deterministic.
RUN if [ "$WITH_MODELS" = "true" ]; then \
      python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; \
    fi

# Bake a ready index built with the embedder the container will actually use.
ENV RAG_INDEX_DIR=/app/data/index
RUN python scripts/ingest.py --sample --index-dir /app/data/index

# Run as a non-root user; /app/data is writable so uploads and the shut-down
# save both work.
RUN useradd --create-home --uid 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Uses python rather than curl so the image needs no extra OS package.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# One worker on purpose: the index lives in process memory, so more workers
# would each hold a private copy. Scale out with replicas behind a proxy.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
