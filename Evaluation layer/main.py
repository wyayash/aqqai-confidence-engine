"""
AQQAI — FastAPI Pipeline (main.py)
====================================
Full pipeline:

  POST /api/v1/query
    User Query
        ↓
    Task Analyzer → classify query (coding/factual/reasoning/summary/creative/general)
        ↓
    Aqua Orchestrator → 3 models in parallel
        ↓
    Collect 3 Responses
        ↓
    Heuristic Scorer (RCKS) → Score each response — Layer 1 evaluation
        ↓
    Bayesian Confidence Engine → update priors with RCKS scores
        ↓
    Get Bayesian weights → normalised trust scores per model
        ↓
    Fusion Engine → blend responses using Bayesian weights
        ↓
    Return: task_type + fusion_weights + final_response + individual scores

Other endpoints:
  GET  /api/v1/responses/{request_id}  — all scored responses for a past query
  GET  /api/v1/evaluate/{request_id}   — fused response + all scores for a past query
  GET  /health                         — system health check
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adapters import ALL_ADAPTERS, BaseModelAdapter
from scorer import HeuristicScorer, ModelResponse, ScoredResponse
from fusion import fuse_responses
from task_analyzer import analyze_task
from pipeline_logger import log
import bayesian

# ──────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AQQAI Orchestrator API",
    description = "Multi-model AI orchestration with task classification, RCKS scoring, Bayesian confidence + response fusion",
    version     = "4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

scorer = HeuristicScorer()

# In-memory store — replace with PostgreSQL in production
store: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ──────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    model_count = len(bayesian.store.get_all())
    log.info(f"AQQAI v4.0.0 started — Bayesian store ready ({model_count} models loaded)")
    log.info(f"Registered models: {[a.model_id for a in ALL_ADAPTERS]}")


@app.on_event("shutdown")
def shutdown_event():
    bayesian.store.save()
    log.info("Shutdown — Bayesian priors saved to priors.json")


# ──────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:   str
    user_id: Optional[str] = "anonymous"


class QueryResponse(BaseModel):
    request_id:      str
    query:           str
    task_type:       str
    final_response:  str
    fusion_weights:  dict
    model_responses: list[dict]
    total_time_ms:   float


# ──────────────────────────────────────────────────────────
# ORCHESTRATOR
# ──────────────────────────────────────────────────────────

def _call_model(adapter: BaseModelAdapter, query: str) -> ModelResponse:
    try:
        return adapter.send_query(query)
    except Exception as e:
        log.warning(f"Model call failed | model={adapter.model_id} | error={str(e)}")
        return ModelResponse(
            model_id   = adapter.model_id,
            content    = "",
            latency_ms = 0.0,
            success    = False,
            error      = str(e),
        )


def orchestrate(query: str, request_id: str) -> tuple[list[ScoredResponse], str, str, dict, float]:
    """
    Full orchestration flow:
      1. Task Analyzer classifies the query
      2. Fan out to all models in parallel
      3. Score all responses (RCKS — Layer 1)
      4+5. Bayesian update + get fusion weights
      6. Fuse responses using Bayesian weights
    """
    start = time.time()

    # ── Step 1: Classify ──────────────────────────────────
    task_type = analyze_task(query)
    log.info(f"[{request_id}] Task classified | task_type={task_type} | query={query[:80]!r}")

    # ── Step 2+3: Parallel model calls ───────────────────
    log.info(f"[{request_id}] Calling {len(ALL_ADAPTERS)} models in parallel")
    raw_responses: list[ModelResponse] = []
    with ThreadPoolExecutor(max_workers=len(ALL_ADAPTERS)) as pool:
        futures = {
            pool.submit(_call_model, adapter, query): adapter.model_id
            for adapter in ALL_ADAPTERS
        }
        for future in as_completed(futures):
            r = future.result()
            if r.success:
                log.info(f"[{request_id}] Model OK      | model={r.model_id} | latency={r.latency_ms:.0f}ms")
            else:
                log.warning(f"[{request_id}] Model FAILED  | model={r.model_id} | error={r.error}")
            raw_responses.append(r)

    # ── Step 4: RCKS scoring (Layer 1) + Layer 2 if coding ──
    scored = scorer.score_all(query, raw_responses, task_type)

    for s in scored:
        log.debug(
            f"[{request_id}] RCKS | model={s.model_id} | "
            f"R={s.relevance:.3f} C={s.coherence:.3f} "
            f"K={s.completeness:.3f} S={s.consistency:.3f} | "
            f"score={s.weighted_score:.4f} | tier={s.confidence_tier}"
        )

    # ── Step 5: Bayesian update + fusion weights ──────────
    eval_scores = {
        s.model_id: (s.weighted_score if s.success else 0.1)
        for s in scored
    }

    bayes_result = bayesian.process_confidence_request(
        store   = bayesian.store,
        payload = {
            "task_type":   task_type,
            "eval_scores": eval_scores,
            "query_id":    request_id,
        },
        verbose = False
    )

    fusion_weights = bayes_result["weights"]
    log.debug(f"[{request_id}] Fusion weights | {fusion_weights}")

    # ── Step 6: Fuse ──────────────────────────────────────
    final_response = fuse_responses(query, scored, fusion_weights)

    if not final_response:
        log.error(f"[{request_id}] Fusion produced empty response — all models failed")
        raise ValueError("All model responses failed — cannot produce fused response.")

    bayesian.store.save()

    total_ms = round((time.time() - start) * 1000, 2)

    log.info(
        f"[{request_id}] Pipeline complete | "
        f"task={task_type} | "
        f"response_len={len(final_response)} chars | "
        f"total={total_ms}ms"
    )

    return scored, final_response, task_type, fusion_weights, total_ms


# ──────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse)
def submit_query(body: QueryRequest):
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    log.info(f"[{request_id}] Incoming query | user={body.user_id} | query={body.query[:80]!r}")

    try:
        scored, final_response, task_type, fusion_weights, total_ms = orchestrate(
            query      = body.query,
            request_id = request_id,
        )
    except ValueError as e:
        log.error(f"[{request_id}] Orchestration failed | error={str(e)}")
        raise HTTPException(status_code=503, detail=str(e))

    all_responses = [s.to_dict_with_content() for s in scored]

    store[request_id] = {
        "query":          body.query,
        "task_type":      task_type,
        "scored":         all_responses,
        "final_response": final_response,
        "fusion_weights": fusion_weights,
        "total_ms":       total_ms,
    }

    return QueryResponse(
        request_id      = request_id,
        query           = body.query,
        task_type       = task_type,
        final_response  = final_response,
        fusion_weights  = fusion_weights,
        model_responses = all_responses,
        total_time_ms   = total_ms,
    )


@app.get("/api/v1/responses/{request_id}")
def get_responses(request_id: str):
    if request_id not in store:
        log.warning(f"GET /responses — request_id not found: {request_id}")
        raise HTTPException(status_code=404, detail="Request ID not found.")
    data = store[request_id]
    return {
        "request_id":      request_id,
        "query":           data["query"],
        "task_type":       data["task_type"],
        "model_responses": data["scored"],
    }


@app.get("/api/v1/evaluate/{request_id}")
def get_evaluation(request_id: str):
    if request_id not in store:
        log.warning(f"GET /evaluate — request_id not found: {request_id}")
        raise HTTPException(status_code=404, detail="Request ID not found.")
    data = store[request_id]
    return {
        "request_id":      request_id,
        "query":           data["query"],
        "task_type":       data["task_type"],
        "final_response":  data["final_response"],
        "fusion_weights":  data["fusion_weights"],
        "model_responses": data["scored"],
        "total_time_ms":   data["total_ms"],
    }


@app.get("/health")
def health():
    log.debug("Health check called")
    return {
        "status":          "ok",
        "models":          [a.model_id for a in ALL_ADAPTERS],
        "scorer":          "heuristic_rcks_v2",
        "fusion":          "bayesian_weighted_additive",
        "task_analyzer":   "keyword_matching_v1",
        "bayesian_engine": "active",
        "model_priors":    bayesian.store.get_all(),
        "version":         "4.0.0",
    }