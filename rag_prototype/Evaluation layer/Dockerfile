# ── Stage 1: Base image ───────────────────────────────────
# python:3.13-slim keeps the image small while matching your dev environment
FROM python:3.13-slim

# ── System dependencies ───────────────────────────────────
# gcc + g++ needed to compile some sentence-transformers / scikit-learn C extensions
# curl is used by the docker-compose healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────
# Copy requirements first (before source code) so Docker can cache this
# layer — if only source code changes, pip install is NOT re-run on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-bake the sentence-transformers model into the image ───────────────
# Downloads all-MiniLM-L6-v2 weights at BUILD time, not at runtime.
# Without this, first request would hang for 30-60s while the model downloads.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy source code ──────────────────────────────────────
COPY adapters.py .
COPY scorer.py .
COPY fusion.py .
COPY task_analyzer.py .
COPY bayesian.py .
COPY main.py .

# ── Runtime config ────────────────────────────────────────
# Port the FastAPI server listens on
EXPOSE 8000

# ── Start command ─────────────────────────────────────────
# --host 0.0.0.0  → listen on all interfaces (required inside Docker)
# --port 8000     → match EXPOSE above
# --workers 1     → single worker (safe for in-memory store dict)
#                   increase only after migrating store to PostgreSQL
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]