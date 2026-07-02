"""Confidence Engine
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
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Updated Task Types for AQQAI Pipeline (Task 1)
TASK_TYPES = ["coding", "factual", "reasoning", "summary", "creative", "general"]

DEFAULT_PRIOR = 0.50
LEARNING_RATE = 0.20
MIN_PRIOR = 0.05
MAX_PRIOR = 0.95
HISTORY_LIMIT = 50

# Global embedder for the agreement signal
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class PriorStore:
    def __init__(self, filepath="priors.json"):
        self.filepath = filepath
        self.priors = {}
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.priors = json.load(f)
            except Exception:
                self.priors = {}

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump(self.priors, f, indent=2)

    def _ensure_exists(self, model: str, task_type: str):
        if model not in self.priors:
            self.priors[model] = {}
        
        if task_type not in self.priors[model] or isinstance(self.priors[model].get(task_type), (float, int)):
            self.priors[model][task_type] = {
                "prior": float(self.priors[model].get(task_type, DEFAULT_PRIOR)),
                "update_history": []
            }
            
    def get(self, model: str, task_type: str) -> float:
        self._ensure_exists(model, task_type)
        return self.priors[model][task_type]["prior"]

    def set(self, model: str, task_type: str, new_prior: float, eval_score: float, old_prior: float):
        self._ensure_exists(model, task_type)
        self.priors[model][task_type]["prior"] = new_prior
        
        history = self.priors[model][task_type]["update_history"]
        history.append({
            "old_prior": round(old_prior, 4),
            "eval_score": round(eval_score, 4),
            "new_prior": round(new_prior, 4)
        })
        if len(history) > HISTORY_LIMIT:
            self.priors[model][task_type]["update_history"] = history[-HISTORY_LIMIT:]


def update_priors(store: PriorStore, task_type: str, eval_scores: dict):
    """Core Bayesian Math"""
    evidence = sum(eval_scores[m] * store.get(m, task_type) for m in eval_scores)
    if evidence == 0:
        return

    for model, score in eval_scores.items():
        old_prior = store.get(model, task_type)
        raw_posterior = (score * old_prior) / evidence
        delta = raw_posterior - old_prior
        new_prior = old_prior + (LEARNING_RATE * delta)
        new_prior = max(MIN_PRIOR, min(MAX_PRIOR, new_prior))
        store.set(model, task_type, new_prior, score, old_prior)
    
    store.save()

def calculate_inter_model_agreement(responses: dict) -> float:
    """
    Task 2: Measures how much all models agree with each other using ML.
    Returns an agreement score between 0.0 and 1.0.
    """
    texts = list(responses.values())
    if len(texts) < 2:
        return 1.0
        
    embeddings = embedder.encode(texts)
    sim_matrix = cosine_similarity(embeddings)
    
    # Extract the upper triangle of the matrix to average the unique pairs
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]
    
    return float(np.mean(pairwise_sims))


def process_confidence_request(payload: dict, store: PriorStore) -> dict:
    """
    Canonical entry point. Expects: {"task_type": "...", "eval_scores": {...}, "responses": {...}}
    """
    task_type = payload.get("task_type", "general")
    eval_scores = payload.get("eval_scores", {})
    responses = payload.get("responses", {})
    models = list(eval_scores.keys())
    
    # 1. Update Priors mathematically
    update_priors(store, task_type, eval_scores)
    
    # 2. Calculate Inter-Model Agreement
    agreement_score = calculate_inter_model_agreement(responses)
    
    # 3. Dynamically adjust weights based on Agreement Signal
    total_prior = sum(store.get(m, task_type) for m in models)
    raw_weights = {m: (store.get(m, task_type) / total_prior) for m in models} if total_prior > 0 else {}
    
    final_weights = {}
    if agreement_score > 0.75:
        # High Agreement: Sharpen weights to aggressively favor the best model
        total = sum(w**2 for w in raw_weights.values())
        final_weights = {m: (w**2)/total for m, w in raw_weights.items()}
    elif agreement_score < 0.40:
        # Low Agreement: Flatten weights to be more conservative
        total = sum(math.sqrt(w) for w in raw_weights.values())
        final_weights = {m: math.sqrt(w)/total for m, w in raw_weights.items()}
    else:
        final_weights = raw_weights
        
    return {
        "agreement_score": round(agreement_score, 4),
        "weights": {m: round(w, 4) for m, w in final_weights.items()},
        "updated_priors": {m: round(store.get(m, task_type), 4) for m in models}
    }