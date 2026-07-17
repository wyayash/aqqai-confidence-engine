"""
Confidence Engine
PURPOSE:
  Maintains and updates trust scores (priors) for each AI model per task type, using Bayes' theorem after every query.

FORMULA:
  P(mᵢ | oᵢ) = P(oᵢ | mᵢ) × P(mᵢ) / P(oᵢ)

PERSISTENCE:
  v1 — JSON file on disk (priors.json)
  v2 — Redis or PostgreSQL (migrate when scaling to production)
"""

import json
import os
from datetime import datetime, timezone


PRIORS_FILE   = "./priors.json"   # where trust scores are persisted
DEFAULT_PRIOR = 0.70

# Learning rate controls how fast priors update.
# 0.2 = each new result moves the prior only 20% toward the new posterior.
LEARNING_RATE = 0.20

# MODELS constant removed — model list is now passed as a parameter.
# Use model_1 through model_5 as placeholders until Tanvi confirms final names.
DEFAULT_MODELS = ["gemini-3.1-flash-lite", "gpt-oss-120b", "mistral-small-latest"]

TASK_TYPES = ["coding", "factual", "reasoning", "summary", "creative", "general"]
HISTORY_LIMIT  = 50   

class PriorStore:

    def __init__(self, filepath: str = PRIORS_FILE):
        self.filepath = filepath
        self.priors   = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                data = json.load(f)
            print(f"[PriorStore] Loaded priors from {self.filepath}")
            return data
        else:
            print(f"[PriorStore] No prior file found — initialising defaults")
            return self._initialise_defaults()

    def _initialise_defaults(self, models: list = None) -> dict:
        """
        Create the default prior structure for all models and task types.
        Model list is passed as a parameter — not hardcoded.
        """
        models = models or DEFAULT_MODELS
        priors = {}
        for model in models:
            priors[model] = {}
            for task_type in TASK_TYPES:
                priors[model][task_type] = {
                    "prior":   DEFAULT_PRIOR,
                    "history": []            # empty history on first init
                }
        self.save(priors)
        return priors

    def save(self, priors: dict = None):
        """Persist current priors to disk."""
        data = priors if priors is not None else self.priors
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _ensure_model(self, model: str):
        """Dynamically add a model that wasn't in the initial list."""
        if model not in self.priors:
            self.priors[model] = {}
        for task_type in TASK_TYPES:
            if task_type not in self.priors[model]:
                self.priors[model][task_type] = {
                    "prior":   DEFAULT_PRIOR,
                    "history": []
                }

    def get(self, model: str, task_type: str) -> float:
        """Get the current prior for a (model, task_type) pair."""
        self._ensure_model(model)
        return self.priors[model][task_type]["prior"]

    def set(self, model: str, task_type: str, value: float,
            eval_score: float = None, old_prior: float = None):
        """
        Update the prior for a (model, task_type) pair and persist.
        Also logs the update to history if eval_score and old_prior are provided.
        """
        self._ensure_model(model)

        self.priors[model][task_type]["prior"] = round(value, 6)

        # HISTORY LOGGING
        if eval_score is not None and old_prior is not None:
            record = {
                "model":      model,
                "task_type":  task_type,
                "old_prior":  round(old_prior, 6),
                "eval_score": round(eval_score, 6),
                "new_prior":  round(value, 6),
                "timestamp":  datetime.now(timezone.utc).isoformat()
            }
            history = self.priors[model][task_type]["history"]
            history.append(record)

            # Keep only the last HISTORY_LIMIT records
            if len(history) > HISTORY_LIMIT:
                self.priors[model][task_type]["history"] = history[-HISTORY_LIMIT:]

        self.save()

    def get_all(self) -> dict:
        """Return full priors dict."""
        return self.priors

    def get_history(self, model: str, task_type: str) -> list:
        """Return the update history for a specific model/task_type pair."""
        self._ensure_model(model)
        return self.priors[model][task_type]["history"]


# BAYES UPDATE FUNCTIONS

def compute_posterior(
    prior:      float,
    likelihood: float,
    evidence:   float
) -> float:
    """
     Apply Bayes' theorem to compute the posterior trust score.

    P(mᵢ | oᵢ) = P(oᵢ | mᵢ) × P(mᵢ) / P(oᵢ)
    Returns:
        posterior: P(mᵢ | oᵢ) — updated trust score (0 to 1)
    """
    if evidence == 0:
        # Edge case: avoid division by zero.
        # If evidence is 0, all models scored 0 — return prior unchanged.
        return prior

    posterior = (likelihood * prior) / evidence
    # Clamp to [0, 1] — numerical edge cases can push slightly outside
    return max(0.0, min(1.0, posterior))


def apply_learning_rate(
    old_prior:  float,
    posterior:  float,
    alpha:      float = LEARNING_RATE
) -> float:
    """
    Smooth the prior update using a learning rate to prevent overreaction
    to a single response.

    Formula: new_prior = (1 - α) × old_prior + α × posterior

    With α = 0.2:
      - If a model with prior 0.80 gets a terrible score (posterior 0.20):
        new_prior = 0.8 × 0.80 + 0.2 × 0.20 = 0.64 + 0.04 = 0.68
        (drops from 0.80 to 0.68, not catastrophically to 0.20)

      - If a model with prior 0.50 gets a perfect score (posterior 1.0):
        new_prior = 0.8 × 0.50 + 0.2 × 1.0 = 0.40 + 0.20 = 0.60
        (rises steadily, requires consistent good performance to reach high scores)
    """
    new_prior = (1 - alpha) * old_prior + alpha * posterior
    return round(max(0.0, min(1.0, new_prior)), 6)


def compute_evidence(
    likelihoods: dict,
    priors:      dict
) -> float:
    """
    Compute the evidence (normalisation constant) P(oᵢ).

    P(oᵢ) = Σ P(oᵢ | mⱼ) × P(mⱼ)  for all models j

    This is the sum of (likelihood × prior) across ALL models for this query.
    It normalises the posteriors so they are relative to the full set of models,
    not computed in isolation per model.

    Args:
        likelihoods: dict of {model_id: evaluation_score}
        priors:      dict of {model_id: current_prior}

    Returns:
        evidence: float (normalisation constant)
    """
    total = sum(likelihoods[m] * priors[m] for m in likelihoods)
    return total if total > 0 else 1.0   # avoid division by zero


# MAIN UPDATE FUNCTION 
def update_priors(
    store:        PriorStore,
    task_type:    str,
    eval_scores:  dict,
    verbose:      bool = True
) -> dict:
    """
    Takes evaluation scores for each model and updates their priors
    using Bayes' theorem + learning rate smoothing.

    Args:
        store:       PriorStore instance (manages persistence)
        task_type:   which type of task was just processed
        eval_scores: dict of {model_id: evaluation_score (0-1)}

    Returns:
        updated_priors: dict of {model_id: new_prior}
    """

    if verbose:
        print(f"\n{'='*55}")
        print(f"BAYESIAN UPDATE — task_type: '{task_type}'")
        print(f"{'='*55}")
        print(f"Eval scores received: {eval_scores}")

    # Step 1: Get current priors for all models on this task type.
    current_priors = {
        model: store.get(model, task_type)
        for model in eval_scores.keys()
    }

    if verbose:
        print(f"\nCurrent priors:  {current_priors}")

    # Step 2: Compute evidence (normalisation constant across all models)
    evidence = compute_evidence(eval_scores, current_priors)

    if verbose:
        print(f"Evidence P(o):   {round(evidence, 4)}")

    # Step 3: Compute posterior and new prior for each model
    updated_priors = {}

    for model, likelihood in eval_scores.items():
        old_prior = current_priors[model]
        posterior = compute_posterior(old_prior, likelihood, evidence)
        new_prior = apply_learning_rate(old_prior, posterior)

        updated_priors[model] = new_prior

        # Pass old_prior and eval_score so history is logged inside store.set()
        store.set(
            model     = model,
            task_type = task_type,
            value     = new_prior,
            eval_score = likelihood,
            old_prior  = old_prior
        )

        if verbose:
            direction = "↑" if new_prior > old_prior else "↓" if new_prior < old_prior else "→"
            print(f"\n  {model}:")
            print(f"    Prior (before):     {old_prior:.4f}")
            print(f"    Likelihood (score): {likelihood:.4f}")
            print(f"    Posterior:          {posterior:.4f}")
            print(f"    New prior (after):  {new_prior:.4f}  {direction}")

    if verbose:
        print(f"\n{'─'*55}")
        print(f"Priors saved to: {store.filepath}")

    return updated_priors


# FUSION WEIGHTS

def get_fusion_weights(
    store:      PriorStore,
    task_type:  str,
    model_ids:  list
) -> dict:
    """Answers: "Given how much we trust each model right now, what percentage
    of the final answer should each one contribute?"

    Steps:
      1. Look up the current trust score (prior) for each model on this task type
      2. Sum all trust scores together
      3. Divide each score by the total → weights that sum to exactly 1.0
      4. Return as a dict

    Args:
        store:      PriorStore instance
        task_type:  the task category for this query (e.g. "coding")
        model_ids:  list of model names to compute weights for
                    e.g. ["gemini-3.1-flash-lite", "gpt-oss-120b", "mistral-small-latest"]
    """
    # Step 1: Retrieve current trust score for each model
    raw_scores = {
        model: store.get(model, task_type)
        for model in model_ids
    }

    # Step 2: Sum all scores (= normalisation denominator)
    total = sum(raw_scores.values())

    # Edge case: if all priors are 0, fall back to equal weights
    if total == 0:
        equal_weight = round(1.0 / len(model_ids), 6)
        return {model: equal_weight for model in model_ids}

    # Step 3: Normalise — divide each score by total
    weights = {
        model: round(score / total, 6)
        for model, score in raw_scores.items()
    }

    return weights


# CONFIDENCE INTERFACE
# Agreed interface between the Confidence Engine and the Fusion Engine.
#
# INPUT PAYLOAD:
# {
#   "task_type":   "coding",
#   "eval_scores": {"gemini-3.1-flash-lite": 0.91, "gpt-oss-120b": 0.74, ...},
#   "query_id":    "q_001"
# }
#
# RESPONSE:
# {
#   "weights":         {"gemini-3.1-flash-lite": 0.34, "gpt-oss-120b": 0.22, ...},
#   "updated_priors":  {"gemini-3.1-flash-lite": 0.72, "gpt-oss-120b": 0.68, ...}
#}

def process_confidence_request(
    store:   PriorStore,
    payload: dict,
    verbose: bool = True
) -> dict:
    """
    Entry point for the Fusion Engine
    Receives an evaluation payload from the Fusion Engine, runs the full
    Bayesian prior update, computes normalised fusion weights, and returns
    both in a single structured response.
    """
    # Unpack payload
    task_type   = payload["task_type"]
    eval_scores = payload["eval_scores"]
    query_id    = payload.get("query_id", "unknown")
    model_ids   = list(eval_scores.keys())

    if verbose:
        print(f"\n{'='*55}")
        print(f"CONFIDENCE REQUEST — query_id: '{query_id}'")
        print(f"task_type: '{task_type}'")
        print(f"{'='*55}")

    # Step 1: Run Bayesian update; updates priors in store
    updated_priors = update_priors(
        store       = store,
        task_type   = task_type,
        eval_scores = eval_scores,
        verbose     = verbose
    )

    # Step 2: Compute fusion weights from the freshly updated priors.
    weights = get_fusion_weights(
        store     = store,
        task_type = task_type,
        model_ids = model_ids
    )

    if verbose:
        print(f"\nFUSION WEIGHTS:")
        for model, w in weights.items():
            bar = "█" * int(w * 40)
            print(f"  {model:<26} {w:.4f}  {bar}")
        weight_sum = round(sum(weights.values()), 6)
        print(f"\n  Sum check: {weight_sum} {'✓' if abs(weight_sum - 1.0) < 0.001 else '✗ ERROR'}")

    # Return the agreed interface format
    return {
        "weights":        weights,
        "updated_priors": updated_priors
    }

# Module-level store instance — used by main.py
store = PriorStore()

# DEMO
if __name__ == "__main__":
    print("Bayesian Confidence Layer Demo — v3")
    print("="*55)

    # Initialise the prior store.
    # Model passed as parameter
    store = PriorStore()

    # If first run, initialise with 5 placeholder models
    if not store.priors:
        store._initialise_defaults(models=DEFAULT_MODELS)

    print("\nInitial priors (all models, all task types):")
    for model, tasks in store.get_all().items():
        for task, record in tasks.items():
            print(f"  {model} / {task}: {record['prior']}")

    # Simulate 3 queries

    # Query 1: Reasoning task — 5 models
    print("\n\nSIMULATING QUERY 1 — Reasoning task")
    update_priors(store, "reasoning", {
        "gemini-3.1-flash-lite": 0.91,
        "gpt-oss-120b":          0.70,
        "mistral-small-latest":  0.54
    })

    # Query 2: Coding task — 5 models
    print("\n\nSIMULATING QUERY 2 — Coding task")
    update_priors(store, "coding", {
        "gemini-3.1-flash-lite": 0.48,
        "gpt-oss-120b":          0.95,
        "mistral-small-latest":  0.63
    })

    # Query 3: Reasoning task again — 5 models
    print("\n\nSIMULATING QUERY 3 — Reasoning task (again)")
    update_priors(store, "reasoning", {
        "gemini-3.1-flash-lite": 0.88,
        "gpt-oss-120b":          0.66,
        "mistral-small-latest":  0.71
    })
    print("\n\nFINAL PRIOR STATE (after 3 queries):")
    print("─"*55)
    for model, tasks in store.get_all().items():
        for task, record in tasks.items():
            bar = "█" * int(record["prior"] * 20)
            print(f"  {model}/{task:<12} {record['prior']:.4f}  {bar}")

    # Show history for one model/task pair
    print("\n\nHISTORY LOG — gemini-3.1-flash-lite / reasoning (last 50 records):")
    print("─"*55)
    history = store.get_history("gemini-3.1-flash-lite", "reasoning")
    for entry in history:
        print(f"  [{entry['timestamp']}]")
        print(f"    old_prior={entry['old_prior']}  eval_score={entry['eval_score']}  new_prior={entry['new_prior']}")

    print("\n✅ Priors saved to priors.json")
    print("   These become the starting point for the next session.")

    # Test get_fusion_weights() standalone
    print("\n\nSTANDALONE TEST — get_fusion_weights()")
    print("─"*55)
    weights = get_fusion_weights(
        store     = store,
        task_type = "reasoning",
        model_ids = DEFAULT_MODELS
    )
    print("Fusion weights for reasoning task:")
    for model, w in weights.items():
        bar = "█" * int(w * 40)
        print(f"  {model:<26} {w:.4f}  {bar}")
    print(f"  Sum: {round(sum(weights.values()), 6)} ✓")

    # Test process_confidence_request()
    print("\n\nINTERFACE TEST — process_confidence_request()")
    print("─"*55)
    print("Simulating incoming payload from Fusion Engine:")

    request_payload = {
        "task_type": "coding",
        "eval_scores": {
            "gemini-3.1-flash-lite": 0.91,
            "gpt-oss-120b":          0.74,
            "mistral-small-latest":  0.63
        },
        "query_id": "q_001"
    }
    print(f"  {request_payload}")

    response = process_confidence_request(store, request_payload, verbose=True)

    print("\nResponse returned to Fusion Engine:")
    print(f"  weights:        {response['weights']}")
    print(f"  updated_priors: {response['updated_priors']}")