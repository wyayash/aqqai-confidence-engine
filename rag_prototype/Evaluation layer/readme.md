# AQQAI — Multi-Model AI Orchestration System

**Intern:** Jeet Tanwar  
**Project:** AQQAI (Aqua AI) — Evaluation Layer  
**Stack:** Python 3.13, FastAPI, scikit-learn, sentence-transformers

---

## What This System Does

AQQAI takes a user query, classifies it by task type, sends it to multiple AI models in parallel, scores every response using a heuristic evaluation system (RCKS), updates Bayesian confidence priors, and fuses all responses into one combined answer.

```
User Query
    ↓
Task Analyzer → classify (coding / factual / reasoning / summary / creative / general)
    ↓
Aqua Orchestrator → 3 Models in Parallel (Gemini · Cerebras · Mistral)
    ↓
Collect All Responses
    ↓
RCKS Scorer → grade each response (Relevance · Coherence · Completeness · Consistency)
    ↓
Bayesian Confidence Engine → update per-model trust scores using RCKS scores
    ↓
Get Bayesian Weights → normalised trust scores per model (sum to 1.0)
    ↓
Fusion Engine → blend responses using Bayesian weights
    ↓
Return: task_type + final_response + fusion_weights + individual scores
```

---

## Project Structure

```
D:\Evaluation layer\
│
├── main.py            # FastAPI server — full 6-step orchestration pipeline
├── run.py             # CLI test runner — test pipeline without starting server
├── scorer.py          # RCKS heuristic scoring engine
├── adapters.py        # Model adapters (Gemini, Cerebras, Mistral)
├── task_analyzer.py   # Query classifier — 6 task types
├── bayesian.py        # Bayesian confidence engine (PriorStore + update logic)
├── fusion.py          # Response fusion — additive Bayesian-weighted blending
│
├── Dockerfile         # Docker image — pre-bakes all-MiniLM-L6-v2
├── docker-compose.yml # One-command startup with .env loading
├── requirements.txt   # Python dependencies
│
├── .env               # API keys (never committed)
├── .env.example       # Key template for new developers
├── priors.json        # Bayesian prior store (auto-created on first run)
└── outputs/           # Saved JSON results (auto-created, only with --save)
```

---

## Models

| Model | Provider | Adapter Type |
|---|---|---|
| gemini-3.1-flash-lite | Google AI Studio | Native Gemini format |
| gpt-oss-120b | Cerebras | OpenAI-compatible |
| mistral-small-latest | Mistral AI | OpenAI-compatible |

---

## Setup

### Option A — Local (venv)

**1. Create virtual environment**
```powershell
cd "D:\Evaluation layer"
python -m venv venv
venv\Scripts\activate
```

**2. Install dependencies**
```powershell
pip install -r requirements.txt
```

> `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90MB) on first run, then caches it locally.

**3. Add API keys** — create `.env` in the project root:
```
GEMINI_API_KEY=your_gemini_key_here
CEREBRAS_API_KEY=your_cerebras_key_here
MISTRAL_API_KEY=your_mistral_key_here
```

| Key | Get it from |
|---|---|
| GEMINI_API_KEY | aistudio.google.com → Get API Key |
| CEREBRAS_API_KEY | cloud.cerebras.ai → API Keys |
| MISTRAL_API_KEY | console.mistral.ai → API Keys |

---

### Option B — Docker (recommended)

**Prerequisites:** Docker Desktop installed and running.

**1. Create `priors.json` so the volume mount works on first run**
```powershell
echo "{}" > priors.json
```

**2. Build and start**
```powershell
docker-compose up --build
```

API is live at `http://localhost:8000`. Subsequent starts (no code changes):
```powershell
docker-compose up
```

**Stop:**
```powershell
docker-compose down
```

> API keys are loaded from `.env` at runtime — never baked into the image. Bayesian priors persist across restarts via the `priors.json` volume mount.

---

## Running the System

### CLI — no server needed

```powershell
# Activate venv first
venv\Scripts\activate

# Single query
python run.py --query "Explain vector databases in simple terms"

# Single query + save output to JSON in outputs/
python run.py --query "What is RAG?" --save

# Interactive mode
python run.py --interactive
```

### FastAPI Server

```powershell
uvicorn main:app --reload
```

- API: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## API Endpoints

### `POST /api/v1/query`
Runs the full pipeline. Returns fused response + all scores.

**Request:**
```json
{
  "query": "Explain vector databases in simple terms",
  "user_id": "anonymous"
}
```

**Response:**
```json
{
  "request_id": "req_a3f2c1d4e5b6",
  "query": "Explain vector databases in simple terms",
  "task_type": "factual",
  "final_response": "A vector database stores data as high-dimensional numerical embeddings...",
  "fusion_weights": {
    "gemini-2.5-flash": 0.412,
    "gpt-oss-120b": 0.321,
    "mistral-small-latest": 0.267
  },
  "model_responses": [
    {
      "model_id": "gemini-2.5-flash",
      "relevance": 0.72,
      "coherence": 0.81,
      "completeness": 0.90,
      "consistency": 1.00,
      "weighted_score": 0.836,
      "confidence_tier": "MEDIUM",
      "latency_ms": 3241.0,
      "success": true
    }
  ],
  "total_time_ms": 4821.3
}
```

---

### `GET /api/v1/responses/{request_id}`
All scored model responses for a past query.

### `GET /api/v1/evaluate/{request_id}`
Fused response + full score breakdown for a past query.

### `GET /health`
System health check — includes live Bayesian priors per model.

```json
{
  "status": "ok",
  "models": ["gemini-2.5-flash", "gpt-oss-120b", "mistral-small-latest"],
  "scorer": "heuristic_rcks_v2",
  "fusion": "bayesian_weighted_additive",
  "task_analyzer": "keyword_matching_v1",
  "bayesian_engine": "active",
  "version": "4.0.0"
}
```

---

## Task Analyzer

Classifies every query into one of 6 task types before routing. Task type determines which Bayesian prior track is updated.

| Task Type | Example Query |
|---|---|
| coding | "Write a Python function to sort a list" |
| factual | "What is a transformer model?" |
| reasoning | "Why does gradient descent work?" |
| summary | "Summarize this paper on RAG" |
| creative | "Write a poem about neural networks" |
| general | Anything that doesn't match the above |

---

## Scoring System — RCKS

Every model response is scored on 4 dimensions. Weights are fixed across all queries.

```
weighted_score = (R × 0.30) + (C × 0.25) + (K × 0.30) + (S × 0.15)
```

### R — Relevance (30%)
Does the response address what was asked?
- `0.6 × TF-IDF cosine similarity` + `0.4 × keyword overlap`
- TF-IDF catches semantic similarity; keyword overlap ensures literal query terms are present.

### C — Coherence (25%)
Does the response flow logically sentence to sentence?
- Adjacent sentence similarity using `all-MiniLM-L6-v2` (sentence-transformers)
- Falls back to TF-IDF if sentence-transformers unavailable

### K — Completeness (30%)
Did the response answer all parts of the query?
- Sub-part coverage: splits query on conjunctions and question words, checks how many are addressed
- Length signal: `expected = sub-parts × 40 words`
- Final: `0.70 × coverage + 0.30 × length_signal`

### S — Consistency (15%)
Does the response contradict itself?
Starts at 1.0, penalties deducted:

| Check | Penalty |
|---|---|
| Contradiction word pairs (fast/slow, always/never) | −0.08 each |
| Named entity with conflicting attributes | −0.15 |
| Uncertainty phrases (I think, I'm not sure) | −0.08 each |
| Repeated 4-word sequences (3+ times) | −0.20 |

### Confidence Tiers

| Tier | Score | Meaning |
|---|---|---|
| HIGH | ≥ 0.85 | Serve directly |
| MEDIUM | 0.60 – 0.84 | Serve, optionally flag |
| LOW | < 0.60 | Trigger fallback or human review |

---

## Bayesian Confidence Engine

After every query, each model's trust score (prior) is updated using Bayes' theorem.

**Formula:**
```
P(model | output) = P(output | model) × P(model) / P(output)

where:
  likelihood  = RCKS weighted_score for this model
  prior       = current trust score for this model on this task type
  evidence    = Σ (likelihood_j × prior_j) across ALL models  ← normalises relative to full set
  new_prior   = (1 - α) × old_prior + α × posterior           ← learning rate smoothing (α = 0.20)
```

Priors are tracked per model × task type (e.g. `gemini-2.5-flash / coding` is independent of `gemini-2.5-flash / factual`). Persisted to `priors.json` after every query.

---

## Fusion Engine

Responses are blended using an additive algorithm, not extractive ranking.

1. Sort models by Bayesian weight (highest first)
2. Top model's full response = base
3. For each remaining model (in weight order): add sentences that are not already covered (cosine similarity < 0.82 threshold)
4. Max 4 additions per model to prevent padding

The result is a single coherent response that draws on all models proportional to their trust.

---

## Architecture Decisions

**Adapter pattern** — Every model has its own adapter class. Adding a new model means writing one class and adding it to `ALL_ADAPTERS`. Nothing else in the pipeline changes.

**Parallel execution** — `ThreadPoolExecutor` runs all model calls simultaneously. Total time = slowest model, not sum of all models.

**Retry logic** — Non-retryable errors (400/401/403/404/422) fail immediately. Retryable errors (429/500/502/503/504) use exponential backoff: 1s → 2s → 4s.

**Failure isolation** — One model failing never cancels the others. Failed models receive `eval_score = 0.1` (not 0.0) to avoid collapsing their Bayesian prior in a single query.

**RCKS feeds Bayesian, not replaces it** — RCKS is Layer 1 evaluation. Its `weighted_score` is the likelihood input to the Bayesian update. Bayesian weights come out and go into fusion. They are sequential layers.

---

## Yashveer's Contributions

Three items from Yashveer's research are integrated:

- **3-tier confidence system** — HIGH/MEDIUM/LOW thresholds (0.85, 0.60)
- **Uncertainty phrase list** — phrases like "I think", "I'm not sure" flagged as hallucination signals (−0.08 consistency penalty each)
- **Named entity consistency check** — same entity with conflicting attributes across sentences flagged as hallucination signal
- **Bayesian confidence layer** — `PriorStore` class, Bayes update formula, `process_confidence_request()` entry point

---

## Known Gaps — Production Readiness

| Gap | Current State | Production Fix |
|---|---|---|
| Storage | In-memory dict (resets on restart) | PostgreSQL |
| Bayesian persistence | JSON file (single instance only) | Redis or PostgreSQL |
| Logging | Terminal output | Loki |
| Metrics | None | Prometheus + Grafana |
| Workers | 1 (in-memory store) | Increase after migrating to DB |

---

## Sample CLI Output

```
============================================================
Query: Why does Python use indentation?
============================================================

  🔍 Task Analyzer
     Task type : REASONING
     reasoning   : 1 match(es) — ['why'] ← winner

  Calling 3 models in parallel...

  ✅ gpt-oss-120b           — 812ms
  ✅ gemini-2.5-flash       — 2341ms
  ✅ mistral-small-latest   — 3102ms

  📊 Updating Bayesian priors (task_type=reasoning)...
     gemini-2.5-flash          eval=0.821  prior 0.700 → 0.714  Δ=+0.014 ↑
     gpt-oss-120b              eval=0.756  prior 0.700 → 0.707  Δ=+0.007 ↑
     mistral-small-latest      eval=0.698  prior 0.700 → 0.699  Δ=-0.001 ↓

  ⚖️  Fusion weights (Bayesian, normalised to 1.0):
     gemini-2.5-flash          0.3382  ██████████
     gpt-oss-120b              0.3361  ██████████
     mistral-small-latest      0.3257  █████████

  🔀 Fusing responses using Bayesian weights...

────────────────────────────────────────────────────────────
INDIVIDUAL MODEL SCORES (RCKS)
────────────────────────────────────────────────────────────
  Model                     Rel   Coh   Com   Con    Score Tier
  gemini-2.5-flash         0.72  0.81  0.90  1.00   0.8360 MEDIUM
  gpt-oss-120b             0.68  0.77  0.85  1.00   0.7960 MEDIUM
  mistral-small-latest     0.61  0.74  0.82  0.92   0.7430 MEDIUM

────────────────────────────────────────────────────────────
FINAL RESPONSE (Bayesian-weighted fusion)
────────────────────────────────────────────────────────────
Python uses indentation to define code blocks...

Task type      : REASONING
Fusion weights : {'gemini-2.5-flash': 0.3382, ...}
Total pipeline : 3847ms
============================================================
```

---

*Built by Jeet Tanwar — AQQAI Intern*