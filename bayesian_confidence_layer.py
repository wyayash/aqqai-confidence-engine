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
from datetime import datetime

PRIORS_FILE    = "./priors.json"   # where trust scores are persisted
DEFAULT_PRIOR  = 0.70
LEARNING_RATE  = 0.20
HISTORY_LIMIT  = 50
TASK_TYPES = ["reasoning", "coding", "creative"]

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
            print(f"[PriorStore] No prior file found — initialising empty store")
            return {}

    def save(self, priors: dict = None):
        """Persist current priors to disk."""
        data = priors if priors is not None else self.priors
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _ensure_model_exists(self, model: str, task_type: str):
        """Helper to ensure the nested dict structure exists."""
        if model not in self.priors:
            self.priors[model] = {}
        if task_type not in self.priors[model]:
            self.priors[model][task_type] = {
                "prior": DEFAULT_PRIOR,
                "update_history": []
            }

    def get_prior(self, model: str, task_type: str) -> float:
        """Get the current prior for a (model, task_type) pair."""
        self._ensure_model_exists(model, task_type)
        return self.priors[model][task_type]["prior"]

    def log_update(self, model: str, task_type: str, old_prior: float, new_prior: float, eval_score: float):
        """Save the new prior and log the historical record."""
        self._ensure_model_exists(model, task_type)
        
        # Update prior
        self.priors[model][task_type]["prior"] = round(new_prior, 6)
        
        # Create history record
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "task_type": task_type,
            "old_prior": round(old_prior, 6),
            "eval_score": round(eval_score, 6),
            "new_prior": round(new_prior, 6)
        }
        
        # Append and enforce history limit
        history = self.priors[model][task_type]["update_history"]
        history.append(record)
        if len(history) > HISTORY_LIMIT:
            self.priors[model][task_type]["update_history"] = history[-HISTORY_LIMIT:]
            
        self.save()

    def get_all(self) -> dict:
        """Return full priors dict."""
        return self.priors


def compute_posterior(prior: float, likelihood: float, evidence: float) -> float:
    """Apply Bayes' theorem to compute the posterior trust score."""
    if evidence == 0:
        return prior
    posterior = (likelihood * prior) / evidence
    return max(0.0, min(1.0, posterior))

def apply_learning_rate(old_prior: float, posterior: float, alpha: float = LEARNING_RATE) -> float:
    """Smooth the prior update using a learning rate to prevent overreaction."""
    new_prior = (1 - alpha) * old_prior + alpha * posterior
    return round(max(0.0, min(1.0, new_prior)), 6)

def compute_evidence(likelihoods: dict, priors: dict) -> float:
    """Compute the evidence (normalisation constant) P(oᵢ)."""
    total = sum(likelihoods[m] * priors[m] for m in likelihoods)
    return total if total > 0 else 1.0 


def update_priors(store: PriorStore, task_type: str, eval_scores: dict, verbose: bool = True) -> dict:
    """Takes evaluation scores for each model and updates their priors."""
    if verbose:
        print(f"\n{'='*55}\nBAYESIAN UPDATE — task_type: '{task_type}'\n{'='*55}")
        print(f"Eval scores received: {eval_scores}")

    current_priors = {model: store.get_prior(model, task_type) for model in eval_scores.keys()}
    evidence = compute_evidence(eval_scores, current_priors)
    updated_priors = {}

    for model, likelihood in eval_scores.items():
        prior = current_priors[model]
        posterior = compute_posterior(prior, likelihood, evidence)
        new_prior = apply_learning_rate(prior, posterior)

        updated_priors[model] = new_prior
        # Log the update to the history array
        store.log_update(model, task_type, prior, new_prior, likelihood)

        if verbose:
            direction = "↑" if new_prior > prior else "↓" if new_prior < prior else "→"
            print(f"  {model}: Prior: {prior:.4f} | Likelihood: {likelihood:.4f} | New: {new_prior:.4f} {direction}")

    return updated_priors


def get_fusion_weights(store: PriorStore, task_type: str, model_ids: list) -> dict:
    """
    Task 1: Given how much we trust each model right now, what percentage 
    of the final answer should each one contribute?
    """
    scores = {m: store.get_prior(m, task_type) for m in model_ids}
    total = sum(scores.values())
    
    if total == 0:
        return {m: round(1.0 / len(model_ids), 4) for m in model_ids}
        
    weights = {m: round(score / total, 4) for m, score in scores.items()}
    return weights


def process_evaluation_payload(store: PriorStore, payload: dict) -> dict:
    """
    The agreed interface with the Fusion Engine. 
    Receives standard JSON payload, updates priors, and returns weights and new priors.
    """
    task_type = payload.get("task_type")
    eval_scores = payload.get("eval_scores", {})
    model_ids = list(eval_scores.keys())
    
    # 1. Update the priors based on the eval scores
    updated_priors = update_priors(store, task_type, eval_scores, verbose=False)
    
    # 2. Get the normalised weights for the Fusion Engine
    weights = get_fusion_weights(store, task_type, model_ids)
    
    # 3. Return exact agreed format
    return {
        "weights": weights,
        "updated_priors": updated_priors
    }


if __name__ == "__main__":
    print("Bayesian Confidence Layer Demo (5 Models)")
    print("="*55)

    store = PriorStore()
    
    # Simulate a payload from the Fusion Engine pipeline using 5 placeholder models
    evaluation_payload = {
        "task_type": "coding",
        "eval_scores": {
            "model_1": 0.91, 
            "model_2": 0.74,
            "model_3": 0.88,
            "model_4": 0.45,
            "model_5": 0.62
        },
        "query_id": "q_001"
    }
    
    print("\nSimulating Payload from Fusion Engine:")
    print(json.dumps(evaluation_payload, indent=2))
    
    engine_response = process_evaluation_payload(store, evaluation_payload)
    
    print("\nResponse sent back to Fusion Engine (Weights & Updated Priors):")
    print(json.dumps(engine_response, indent=2))
    
    print("\nVerifying History Log (Model 1):")
    m1_history = store.get_all()["model_1"]["coding"]["update_history"][-1]
    print(json.dumps(m1_history, indent=2))