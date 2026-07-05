"""
AQQAI — CLI Test Runner (run.py)
==================================
Test the full pipeline from the terminal without starting the server.

Usage:
  python run.py --query "Explain vector databases in simple terms"
  python run.py --query "What is RAG?" --save
  python run.py --interactive
"""

import argparse
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from adapters import ALL_ADAPTERS, BaseModelAdapter
from scorer import HeuristicScorer, ModelResponse
from fusion import fuse_responses
from task_analyzer import analyze_task, analyze_task_detailed
from pipeline_logger import log
import bayesian

scorer = HeuristicScorer()
os.makedirs("outputs", exist_ok=True)


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

def _call_model(adapter: BaseModelAdapter, query: str) -> ModelResponse:
    try:
        return adapter.send_query(query)
    except Exception as e:
        log.warning(f"Model call failed | model={adapter.model_id} | error={str(e)}")
        return ModelResponse(
            model_id=adapter.model_id,
            content="",
            latency_ms=0.0,
            success=False,
            error=str(e),
        )


def run_pipeline(query: str) -> dict:
    request_id = f"req_{uuid.uuid4().hex[:10]}"
    start      = time.time()

    log.info(f"[{request_id}] Pipeline started | query={query[:80]!r}")

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # ── Step 1: Task Analysis ──────────────────────────────
    task_detail = analyze_task_detailed(query)
    task_type   = task_detail["task_type"]

    log.info(f"[{request_id}] Task classified | task_type={task_type}")

    print(f"\n  🔍 Task Analyzer")
    print(f"     Task type : {task_type.upper()}")
    matched_summary = {k: v for k, v in task_detail["matched_keywords"].items() if v}
    if matched_summary:
        for cat, kws in matched_summary.items():
            marker = " ← winner" if cat == task_type else ""
            print(f"     {cat:<12}: {task_detail['scores'][cat]} match(es) — {kws}{marker}")
    else:
        print(f"     No keywords matched → defaulting to general")

    # ── Step 2: Parallel model calls ──────────────────────
    print(f"\n  Calling {len(ALL_ADAPTERS)} models in parallel...\n")
    log.info(f"[{request_id}] Calling {len(ALL_ADAPTERS)} models in parallel")

    raw_responses = []
    with ThreadPoolExecutor(max_workers=len(ALL_ADAPTERS)) as pool:
        futures = {
            pool.submit(_call_model, adapter, query): adapter.model_id
            for adapter in ALL_ADAPTERS
        }
        for future in as_completed(futures):
            result = future.result()
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.model_id} — {result.latency_ms:.0f}ms")
            if result.success:
                log.info(f"[{request_id}] Model OK     | model={result.model_id} | latency={result.latency_ms:.0f}ms")
            else:
                log.warning(f"[{request_id}] Model FAILED | model={result.model_id} | error={result.error}")
            raw_responses.append(result)

    # ── Step 3: RCKS Scoring (Layer 1) + Layer 2 if coding ──
    scored = scorer.score_all(query, raw_responses, task_type)
    ranked = sorted(scored, key=lambda s: s.weighted_score, reverse=True)

    for s in scored:
        log.debug(
            f"[{request_id}] RCKS | model={s.model_id} | "
            f"R={s.relevance:.3f} C={s.coherence:.3f} "
            f"K={s.completeness:.3f} S={s.consistency:.3f} | "
            f"score={s.weighted_score:.4f} | tier={s.confidence_tier}"
        )

    # ── Step 4 + 5: Bayesian update + fusion weights ──────
    print(f"\n  📊 Bayesian update (task_type={task_type})...")

    eval_scores = {
        s.model_id: (s.weighted_score if s.success else 0.1)
        for s in scored
    }

    bayes_result = bayesian.process_confidence_request(
        store   = bayesian.store,
        payload = {
            "task_type":   task_type,
            "eval_scores": eval_scores,
            "query_id":    request_id
        },
        verbose = False
    )

    updated_priors = bayes_result["updated_priors"]
    for model_id, new_prior in updated_priors.items():
        history = bayesian.store.get_history(model_id, task_type)
        if history:
            old_prior = history[-1]["old_prior"]
            delta     = new_prior - old_prior
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        else:
            old_prior = bayesian.DEFAULT_PRIOR
            delta     = 0.0
            direction = "→"
        print(
            f"     {model_id:<25} "
            f"eval={eval_scores[model_id]:.3f}  "
            f"prior {old_prior:.3f} → {new_prior:.3f}  "
            f"Δ={delta:+.3f} {direction}"
        )
        log.debug(
            f"[{request_id}] Bayesian | model={model_id} | "
            f"old={old_prior:.3f} new={new_prior:.3f} delta={delta:+.3f}"
        )

    # ── Step 5: Show fusion weights ────────────────────────
    fusion_weights = bayes_result["weights"]
    log.debug(f"[{request_id}] Fusion weights | {fusion_weights}")

    print(f"\n  ⚖️  Fusion weights (Bayesian, normalised to 1.0):")
    for model_id, weight in sorted(fusion_weights.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 30)
        print(f"     {model_id:<25} {weight:.4f}  {bar}")

    # ── Step 6: Fuse responses ────────────────────────────
    print(f"\n  🔀 Fusing responses using Bayesian weights...")
    final_response = fuse_responses(query, scored, fusion_weights)

    total_ms = round((time.time() - start) * 1000, 2)

    bayesian.store.save()

    log.info(
        f"[{request_id}] Pipeline complete | "
        f"task={task_type} | "
        f"response_len={len(final_response)} chars | "
        f"fusion_weights={fusion_weights} | "
        f"total={total_ms}ms"
    )

    # ── Scores Table ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print("INDIVIDUAL MODEL SCORES (RCKS — ranked by weighted score)")
    print(f"{'─'*60}")
    print(f"  {'Model':<25} {'Rel':>5} {'Coh':>5} {'Com':>5} {'Con':>5} {'Score':>7} {'Tier':<8}")
    print(f"  {'─'*25} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*7} {'─'*8}")
    for s in ranked:
        print(
            f"  {s.model_id:<25} "
            f"{s.relevance:>5.2f} "
            f"{s.coherence:>5.2f} "
            f"{s.completeness:>5.2f} "
            f"{s.consistency:>5.2f} "
            f"{s.weighted_score:>7.4f} "
            f"{s.confidence_tier:<8}"
        )

    # ── All model responses JSON ───────────────────────────
    print(f"\n{'─'*60}")
    print("ALL MODEL RESPONSES (JSON)")
    print(f"{'─'*60}")
    all_responses_json = [
        {
            "model_id":        s.model_id,
            "relevance":       s.relevance,
            "coherence":       s.coherence,
            "completeness":    s.completeness,
            "consistency":     s.consistency,
            "weighted_score":  s.weighted_score,
            "confidence_tier": s.confidence_tier,
            "latency_ms":      s.latency_ms,
            "success":         s.success,
            "content":         s.content,
        }
        for s in ranked
    ]
    print(json.dumps(all_responses_json, indent=2))

    # ── Final Fused Response ───────────────────────────────
    print(f"\n{'─'*60}")
    print("FINAL RESPONSE (Bayesian-weighted fusion)")
    print(f"{'─'*60}")
    print(final_response)

    print(f"\nTask type      : {task_type.upper()}")
    print(f"Fusion weights : {fusion_weights}")
    print(f"Total pipeline : {total_ms}ms")
    print(f"{'='*60}\n")

    result = {
        "request_id":      request_id,
        "query":           query,
        "task_type":       task_type,
        "total_time_ms":   total_ms,
        "final_response":  final_response,
        "fusion_weights":  fusion_weights,
        "model_responses": all_responses_json,
    }
    return result


def save_result(result: dict):
    filename = f"outputs/result_{result['request_id']}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {filename}")
    log.info(f"Result saved | file={filename}")


# ──────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AQQAI Pipeline CLI")
    parser.add_argument("--query",       type=str,  help="Query to run")
    parser.add_argument("--save",        action="store_true", help="Save output to JSON")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive:
        log.info("Interactive mode started")
        print("AQQAI Interactive Mode — type 'exit' to quit\n")
        while True:
            query = input("Enter query: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                log.info("Interactive mode exited")
                break
            if not query:
                continue
            result = run_pipeline(query)
            if args.save:
                save_result(result)

    elif args.query:
        result = run_pipeline(args.query)
        if args.save:
            save_result(result)

    else:
        result = run_pipeline("Explain vector databases in simple terms")
        if args.save:
            save_result(result)


if __name__ == "__main__":
    main()