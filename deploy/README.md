# Deploying the RAG Research Assistant

The repository is a normal Docker app, so it runs on any host that builds a
`Dockerfile`. Two free options are set up here. Pick one:

| | Render | Hugging Face Spaces |
| --- | --- | --- |
| Setup | one click, no account secrets | create a token, push |
| Result | `https://<name>.onrender.com` | `https://huggingface.co/spaces/<you>/<name>` |
| RAM | 512 MB | 16 GB |
| Backends | fallback (hashing + lexical) | **full** (Sentence Transformers + cross-encoder) |
| Idle behaviour | spins down after 15 min, ~1 min cold start | sleeps after 48 h, wakes on visit |

**Recommendation:** Render is the fastest way to a live URL; Hugging Face Spaces
is the better *demo* because 16 GB of RAM lets the real models run.

---

## Option A — Render (one click)

1. Click this button:

   [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/iwdie/rag-research-assistant)

2. Sign in with GitHub when prompted (no card required).
3. Render reads [`render.yaml`](../render.yaml), builds the `Dockerfile`, and
   serves the app. First build takes a few minutes.
4. Open the `onrender.com` URL it gives you. The demo UI is at `/`, the API docs
   at `/docs`, and `GET /health` names the live backends.

Nothing to configure: the Blueprint sets the index directory and fusion
settings, and Render injects the port the app binds to.

> Free Render services **spin down after 15 minutes** with no traffic. The next
> request takes about a minute to wake it. That is Render's free tier, not the
> app.
>
> The free plan is **0.1 CPU / 512 MB**, so it runs the fallback backends. The
> demo answers questions and cites sources, but the embedder is lexical — see
> the "Honest limitations" section of the [main README](../README.md).

### Manual (no Blueprint)

```bash
# In the Render dashboard: New → Web Service → build from your GitHub repo,
# Runtime: Docker, Instance type: Free, Health check path: /health.
```

---

## Option B — Hugging Face Spaces (full models)

A Docker Space gives 2 vCPU / 16 GB RAM for free, which is enough for
Sentence Transformers, FAISS and the cross-encoder — the configuration the
project is really built around.

### 1. Create the Space

1. Sign up / sign in at https://huggingface.co.
2. Go to https://huggingface.co/new-space.
3. Name it `rag-research-assistant`, choose **Docker** as the SDK, select the
   **Free** hardware, and create it.

### 2. Create an access token

https://huggingface.co/settings/tokens → **New token** → type **Write** → copy it.

### 3. Push this repository to the Space

The Space reads its configuration from the YAML block at the top of its
`README.md`. That block lives in
[`deploy/huggingface/README.md`](huggingface/README.md) so the GitHub README can
stay clean. From the repository root:

<details>
<summary>Bash / Git Bash / macOS / Linux</summary>

```bash
# Stage a Space-ready copy: the Space card becomes README.md.
rm -rf ../hf-space && mkdir ../hf-space
git archive HEAD | tar -x -C ../hf-space
cp deploy/huggingface/README.md ../hf-space/README.md

cd ../hf-space
git init -b main
git add .
git commit -m "Deploy RAG research assistant to Hugging Face Spaces"
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/rag-research-assistant
git push --force space main
# When asked, use your HF username and the Write token as the password.
```

</details>

<details>
<summary>PowerShell (Windows)</summary>

```powershell
Remove-Item ..\hf-space -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory ..\hf-space | Out-Null
git archive HEAD | tar -x -C ..\hf-space
Copy-Item deploy\huggingface\README.md ..\hf-space\README.md -Force

Set-Location ..\hf-space
git init -b main
git add .
git commit -m "Deploy RAG research assistant to Hugging Face Spaces"
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/rag-research-assistant
git push --force space main
# When asked, use your HF username and the Write token as the password.
```

</details>

### 4. Turn the models on

The Space builds with the light image by default. To run the real bi-encoder and
cross-encoder, open **Settings → Variables and secrets**, add a **Variable**
named `WITH_MODELS` with value `true`, then **Factory reboot** the Space. The
build then installs FAISS + Sentence Transformers and bakes the index with the
real embedder.

`GET /health` should report `"embedder": "sentence-transformers/all-MiniLM-L6-v2"`
once that build finishes.

---

## Anywhere else

The image is portable; only two things matter:

- **Bind to `$PORT`.** The container already does
  (`uvicorn --port ${PORT:-8000}`), so Fly, Railway, Heroku, Cloud Run and
  Kubernetes work without changes.
- **Build arguments are platform-specific.** `WITH_MODELS=true` needs a way to
  pass a Docker build arg. On hosts that cannot, run the light image and accept
  the fallback backends.

```bash
# Local check, exactly what a host runs:
docker build -t rag-research-assistant .
docker run --rm -p 8000:8000 rag-research-assistant
curl -s localhost:8000/health
```
