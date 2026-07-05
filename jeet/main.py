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
# STARTUP — initialise Bayesian store
# ──────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    model_ids = [adapter.model_id for adapter in ALL_ADAPTERS]
    # Ensure the store knows about all models and load existing priors from disk
    bayesian.store.initialize(model_ids)
    bayesian.store.load()  # auto-loads from confidence_store.json if it exists


@app.on_event("shutdown")
def shutdown_event():
    bayesian.store.save()  # persist priors to confidence_store.json


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
        return ModelResponse(
            model_id   = adapter.model_id,
            content    = "",
            latency_ms = 0.0,
            success    = False,
            error      = str(e),
        )


def orchestrate(query: str, request_id: str) -> tuple[list[ScoredResponse], str, str, dict, float]:
    """
    Full orchestration flow — Task 3 complete:
      1. Task Analyzer classifies the query
      2. Fan out to all models in parallel
      3. Score all responses (RCKS — Layer 1)
      4. Update Bayesian priors with RCKS scores
      5. Get Bayesian weights (normalised trust scores)
      6. Fuse responses using Bayesian weights
    """
    start = time.time()

    # Step 1: Classify
    task_type = analyze_task(query)

    # Step 2+3: Parallel model calls
    raw_responses: list[ModelResponse] = []
    with ThreadPoolExecutor(max_workers=len(ALL_ADAPTERS)) as pool:
        futures = {
            pool.submit(_call_model, adapter, query): adapter.model_id
            for adapter in ALL_ADAPTERS
        }
        for future in as_completed(futures):
            raw_responses.append(future.result())

    # Step 4: RCKS scoring
    scored = scorer.score_all(query, raw_responses)

    # Step 5 — build eval_scores dict, pass ALL models at once (YASHVEER'S WAY)
    eval_scores = {
        s.model_id: (s.weighted_score if s.success else 0.1)
        for s in scored
    }

    result = bayesian.process_confidence_request(
        store   = bayesian.store,
        payload = {
            "task_type":   task_type,
            "eval_scores": eval_scores,
            "query_id":    request_id
        },
        verbose = False   # set True for debug
    )

    # Step 6: Extract Bayesian weights
    fusion_weights = result["weights"]

    # Step 7: Fuse
    final_response = fuse_responses(query, scored, fusion_weights)

    if not final_response:
        raise ValueError("All model responses failed — cannot produce fused response.")

    # Persist updated priors after each request
    bayesian.store.save()

    total_ms = round((time.time() - start) * 1000, 2)
    return scored, final_response, task_type, fusion_weights, total_ms


# ──────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse)
def submit_query(body: QueryRequest):
    """
    Main endpoint — runs the full pipeline.

    Returns:
      - task_type       → coding / factual / reasoning / summary / creative / general
      - final_response  → fused answer blended from all models
      - fusion_weights  → Bayesian weights used (model_id → weight, sums to 1.0)
      - model_responses → individual RCKS scores per model
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    try:
        scored, final_response, task_type, fusion_weights, total_ms = orchestrate(body.query, request_id)
    except ValueError as e:
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
    return {
        "status":          "ok",
        "models":          [a.model_id for a in ALL_ADAPTERS],
        "scorer":          "heuristic_rcks_v2",
        "fusion":          "bayesian_weighted_additive",
        "task_analyzer":   "keyword_matching_v1",
        "bayesian_engine": "active",
        "model_priors":    bayesian.store.get_all(),   # fixed: use new PriorStore method
        "version":         "4.0.0",
    }