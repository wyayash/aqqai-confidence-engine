"""
AQQAI — Fusion Engine (fusion.py)
===================================
Blends multiple model responses into one final answer using Bayesian weights.

Algorithm (per task spec):
  1. Sort responses by Bayesian weight (highest trust first)
  2. Take the top response as the base answer
  3. Go through remaining responses one by one (in weight order)
  4. For each response, check sentence by sentence —
     if a sentence adds new information not already in the base, add it
  5. Return the final combined answer

The model with the highest Bayesian weight contributes most — its full
response is the foundation. Lower-weight models only add sentences that
are genuinely new (not near-duplicates of what's already in the base).

This implements the architecture diagram formula:
  O* = Σ P(mᵢ|oᵢ) · oᵢ
  (weighted synthesis — not picking a winner — building a collective answer)

How weights connect:
  - Bayesian weights come from bayesian.get_weights(task_type)
  - These replace RCKS scores as the model trust signal
  - RCKS scores still feed INTO the Bayesian update (they're the eval input)
  - Sentence deduplication uses sentence-transformers embeddings
"""

from __future__ import annotations

import re
import logging
from typing import Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from scorer import ScoredResponse

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# MODEL — loaded once at import time (same model scorer uses)
# ──────────────────────────────────────────────────────────

_embed_model: Optional[SentenceTransformer] = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────

# Cosine similarity threshold above which two sentences are "the same"
DUPLICATE_THRESHOLD = 0.82

# Minimum sentence length to consider (filters filler/headers)
MIN_SENTENCE_WORDS = 6

# Maximum sentences to take from each non-base model
# (prevents one verbose model from dominating the additions)
MAX_ADDITIONS_PER_MODEL = 4


# ──────────────────────────────────────────────────────────
# SENTENCE UTILITIES
# ──────────────────────────────────────────────────────────

# Patterns for sentences that are structural noise, not content
_FILLER_PATTERNS = [
    r"^(certainly|sure|of course|absolutely|great question)[!,.]",
    r"^(here'?s?|below is|the following|in this|this is)",
    r"^(i hope|i'?m happy|feel free|let me know)",
    r"^#+\s",                        # markdown headers
    r"^\s*[-*•]\s",                  # bullet points
    r"^(in summary|to summarize|to conclude|in conclusion)",
]
_FILLER_RE = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)


def _split_sentences(text: str) -> list[str]:
    """
    Split response text into individual sentences.
    Filters out structural noise and very short sentences.
    """
    if not text or not text.strip():
        return []

    # Split on sentence-ending punctuation followed by whitespace/end
    raw = re.split(r'(?<=[.!?])\s+', text.strip())

    sentences = []
    for s in raw:
        s = s.strip()
        # Filter: too short
        if len(s.split()) < MIN_SENTENCE_WORDS:
            continue
        # Filter: structural filler
        if _FILLER_RE.match(s):
            continue
        sentences.append(s)

    return sentences


def _is_duplicate(candidate: str, existing_embeddings: list, model: SentenceTransformer) -> bool:
    """
    Returns True if candidate sentence is semantically similar to any
    sentence already in the base (cosine similarity > DUPLICATE_THRESHOLD).
    """
    if not existing_embeddings:
        return False

    candidate_emb = model.encode([candidate])
    existing_matrix = np.vstack(existing_embeddings)
    sims = cosine_similarity(candidate_emb, existing_matrix)[0]

    return float(np.max(sims)) > DUPLICATE_THRESHOLD


# ──────────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────────

def fuse_responses(
    query: str,
    scored_responses: list[ScoredResponse],
    weights: Optional[dict[str, float]] = None,
) -> str:
    """
    Fuse multiple model responses into one combined answer.

    Algorithm:
      1. Sort responses by Bayesian weight (highest first)
      2. Take the top response as the base answer
      3. For each remaining response (in weight order):
         - Split into sentences
         - Add only sentences that are NOT near-duplicates of the base
      4. Return combined answer

    Args:
        query:            Original user query (kept for future query-relevance filtering)
        scored_responses: List of ScoredResponse objects from scorer.py
        weights:          Dict of model_id → Bayesian weight from bayesian.get_weights()
                          If None or empty, falls back to RCKS weighted_score as proxy.

    Returns:
        Fused response string.
    """
    # Filter to successful responses only
    successful = [r for r in scored_responses if r.success and r.content.strip()]

    if not successful:
        logger.warning("[Fusion] No successful responses to fuse.")
        return ""

    # Single response — nothing to fuse
    if len(successful) == 1:
        logger.info("[Fusion] Only one successful response — returning as-is.")
        return successful[0].content.strip()

    # ── Step 1: Sort by Bayesian weights (highest trust first) ────────────
    if weights:
        # Use Bayesian weights — the primary path once Task 3 is active
        def sort_key(r: ScoredResponse) -> float:
            return weights.get(r.model_id, 0.0)
    else:
        # Fallback: use RCKS weighted_score as proxy for trust
        # This runs before Bayesian layer has any data (first few queries)
        logger.debug("[Fusion] No Bayesian weights provided — using RCKS scores as fallback")
        def sort_key(r: ScoredResponse) -> float:
            return r.weighted_score

    ranked = sorted(successful, key=sort_key, reverse=True)

    logger.info(
        f"[Fusion] Fusing {len(ranked)} responses. "
        f"Order: {[r.model_id for r in ranked]}"
    )

    # ── Step 2: Take the top response as the base ─────────────────────────
    base_response = ranked[0]
    base_sentences = _split_sentences(base_response.content)

    if not base_sentences:
        # Base model had no extractable sentences — use raw content
        base_sentences = [base_response.content.strip()]

    model = _get_embed_model()

    # Pre-compute embeddings for all base sentences
    base_embeddings = [model.encode([s]) for s in base_sentences]
    fused_sentences = list(base_sentences)

    logger.debug(
        f"[Fusion] Base: {base_response.model_id} "
        f"(weight={sort_key(base_response):.3f}) — {len(base_sentences)} sentences"
    )

    # ── Steps 3 + 4: Enrich with unique sentences from other models ───────
    for response in ranked[1:]:
        candidates = _split_sentences(response.content)
        additions = 0

        model_weight = sort_key(response)
        logger.debug(
            f"[Fusion] Checking {response.model_id} "
            f"(weight={model_weight:.3f}) — {len(candidates)} candidate sentences"
        )

        for sentence in candidates:
            if additions >= MAX_ADDITIONS_PER_MODEL:
                break

            if not _is_duplicate(sentence, base_embeddings, model):
                # New information — add it
                fused_sentences.append(sentence)
                base_embeddings.append(model.encode([sentence]))
                additions += 1
                logger.debug(f"[Fusion]   ✓ Added: {sentence[:80]}...")
            else:
                logger.debug(f"[Fusion]   ✗ Duplicate skipped")

        logger.debug(f"[Fusion]   → {additions} sentences added from {response.model_id}")

    # ── Step 5: Join and return ───────────────────────────────────────────
    fused = " ".join(fused_sentences)
    logger.info(f"[Fusion] Final response: {len(fused_sentences)} sentences from {len(ranked)} models")
    return fused