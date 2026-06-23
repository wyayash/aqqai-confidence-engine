AQQAI: Bayesian Confidence Engine

Overview

The Confidence Engine is a core component of the Aqua AI Orchestration pipeline. It acts as the mathematical memory of the system, tracking the historical reliability (Prior) of parallel AI models across different task types (e.g., coding, reasoning).

It uses Bayes' Theorem combined with an Exponential Moving Average (EMA) learning rate to update trust scores after every query, ensuring the Orchestrator continuously learns which models to trust without overreacting to single-query anomalies.

Core Features

Dynamic Model Ingestion: Automatically registers and tracks any new models passed via the evaluation payload.

Fault-Tolerant Math: Safeguards against division-by-zero anomalies and clamps float boundaries.

Volatility Dampening: Applies a 0.20 learning rate to prevent a single hallucination from crashing a model's historical prior.

Audit Logging: Maintains a rolling history of the last 50 updates per model/task type, capturing timestamps, previous priors, evaluation scores, and the new updated prior.

Backward Compatibility: Automatically detects and upgrades legacy priors.json schema formats.

System Interface

The engine exposes a single, clean entry point (process_confidence_request) designed to integrate directly with the downstream Fusion Engine.

Expected Input (From Evaluation Layer)

{
  "task_type": "coding",
  "eval_scores": {
    "model_1": 0.91,
    "model_2": 0.74,
    "model_3": 0.63
  },
  "query_id": "q_001"
}


Output (To Fusion Engine)

Returns the normalized weights (summing to 1.0) and the newly updated priors.

{
  "weights": {
    "model_1": 0.40,
    "model_2": 0.32,
    "model_3": 0.28
  },
  "updated_priors": {
    "model_1": 0.72,
    "model_2": 0.68,
    "model_3": 0.55
  }
}


Running the Demo

A standalone demo is included to simulate processing a payload and updating the persistence layer.

python bayesian_confidence_layer2.py
