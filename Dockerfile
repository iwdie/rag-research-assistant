# Single image for the API, built so the container can answer a question on
# first boot with no model download and no network access at runtime.
#
# The default build installs the light dependency set (NumPy + hashing + lexical
# fallbacks) and produces a ~250 MB image that starts in seconds. That is the
# right default for a free host (Render's free plan is 512 MB, which cannot hold
# torch). Pass --build-arg WITH_MODELS=true - as docker-compose.yml does - to
# install FAISS, torch and Sentence Transformers instead, warm both model caches
# at build time, and pre-build the sample index with the same embedder the
# runtime uses. Mismatched embedding dimensions between a baked index and the
# configured model is the classic way this kind of service fails to start, so
# the build does it once, deliberately.
FROM python:3.12-slim

ARG WITH_MODELS=false
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

# Run as a non-root user. UID 1000 is the convention Hugging Face Spaces expects
# (it runs the container as UID 1000); /app/data is writable by that user so
# uploads and the shut-down save both work.
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app
USER appuser

# Most PaaS hosts (Render, Railway, Fly, Heroku) inject the port to bind via
# PORT. 8000 is the local / docker-compose default.
ENV PORT=8000

EXPOSE 8000

# Uses python rather than curl so the image needs no extra OS package.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,sys,urllib.request; port=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+port+'/health', timeout=4).status==200 else 1)"

# One worker on purpose: the index lives in process memory, so more workers
# would each hold a private copy. Scale out with replicas behind a proxy.
# `exec` keeps uvicorn as PID 1 so it receives the host's SIGTERM directly.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
