"""
Fusion Engine v1: Heuristic Sentence Blending

PURPOSE:
  Takes responses from 3 parallel AI models and blends them into one 
  final answer using Bayesian trust weights from the Confidence Engine.

INTERFACE WITH CONFIDENCE ENGINE:
  Input  (from Confidence Engine):
    weights:   {"gemini-3.1-flash-lite": 0.38, "gpt-oss-120b": 0.35, "mistral-small-latest": 0.27}
    responses: {"gemini-3.1-flash-lite": "...", "gpt-oss-120b": "...", "mistral-small-latest": "..."}

  Output (to Orchestrator):
    {
      "final_response": "...",
      "base_model":     "gemini-3.1-flash-lite",
      "models_used":    ["gemini-3.1-flash-lite", "gpt-oss-120b"],
      "models_filtered":["mistral-small-latest"],
      "sentences_added": 2
    }
"""

import re
from typing import Optional


# CONSTANTS
WEIGHT_THRESHOLD = 0.15
# 0.65 = if 65% of meaningful words already appear in the base, skip it.
OVERLAP_THRESHOLD = 0.65

# Common words that carry no meaning — excluded from overlap comparison
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they", "as",
    "so", "if", "not", "no", "than", "then", "when", "which", "who", "what",
    "how", "there", "their", "they", "also", "just", "more", "any", "all"
}

def split_into_sentences(text: str) -> list:
    """
    Splits a paragraph of text into individual sentences.
    """
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_keywords(text: str) -> set:
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def compute_overlap(sentence: str, base_text: str) -> float:
    """
    Overlap = |keywords(sentence) ∩ keywords(base)| / |keywords(sentence)|
    """
    sentence_keywords = extract_keywords(sentence)
    if not sentence_keywords:
        return 1.0  # empty/stopword-only sentence — skip it

    base_keywords = extract_keywords(base_text)
    overlap_count = len(sentence_keywords & base_keywords)
    return overlap_count / len(sentence_keywords)


# CORE FUSION FUNCTION 

def fuse_responses(
    responses: dict,
    weights:   dict,
    verbose:   bool = True
) -> dict:
    if verbose:
        print(f"\n{'='*60}")
        print("FUSION ENGINE v1 — Heuristic Sentence Blending")
        print(f"{'='*60}")
        print(f"Models received: {list(responses.keys())}")
        print(f"Weights:         {weights}")

    # Sort models by weight, highest first 
    sorted_models = sorted(weights.keys(), key=lambda m: weights[m], reverse=True)

    if verbose:
        print(f"\n[1/5] Models sorted by weight:")
        for m in sorted_models:
            bar = "█" * int(weights[m] * 30)
            print(f"      {m:<26} {weights[m]:.4f}  {bar}")

    # Select the base response 
    base_model    = sorted_models[0]
    base_response = responses[base_model].strip()

    if verbose:
        print(f"\n[2/5] Base model: '{base_model}' (weight={weights[base_model]:.4f})")
        print(f"      Base response ({len(base_response)} chars):")
        print(f"      \"{base_response[:120]}...\"" if len(base_response) > 120 else f"      \"{base_response}\"")

    # STEP 3: Filter out low-weight models 
    contributing_models = []
    filtered_models     = []

    for model in sorted_models[1:]:   # skip base model
        if weights[model] >= WEIGHT_THRESHOLD:
            contributing_models.append(model)
        else:
            filtered_models.append(model)

    if verbose:
        print(f"\n[3/5] Weight threshold: {WEIGHT_THRESHOLD}")
        print(f"      Contributing (weight ≥ {WEIGHT_THRESHOLD}): {contributing_models}")
        print(f"      Filtered out (weight < {WEIGHT_THRESHOLD}): {filtered_models}")

    # Extract new sentences and append
    current_base    = base_response  
    sentences_added = 0
    addition_log    = []

    for model in contributing_models:
        model_response = responses[model].strip()
        sentences      = split_into_sentences(model_response)

        if verbose:
            print(f"\n[4/5] Processing '{model}' ({len(sentences)} sentences):")

        for sentence in sentences:
            overlap = compute_overlap(sentence, current_base)

            if overlap < OVERLAP_THRESHOLD:
                # New information — append to base
                current_base += " " + sentence
                sentences_added += 1
                addition_log.append({
                    "model":    model,
                    "sentence": sentence,
                    "overlap":  round(overlap, 3)
                })
                if verbose:
                    print(f"      ✅ ADDED   (overlap={overlap:.2f}): \"{sentence[:80]}...\"" 
                          if len(sentence) > 80 else 
                          f"      ✅ ADDED   (overlap={overlap:.2f}): \"{sentence}\"")
            else:
                if verbose:
                    print(f"      ⏭  SKIPPED (overlap={overlap:.2f}): \"{sentence[:80]}...\"" 
                          if len(sentence) > 80 else 
                          f"      ⏭  SKIPPED (overlap={overlap:.2f}): \"{sentence}\"")

    # STEP 5: Return result 
    final_response = current_base.strip()

    if verbose:
        print(f"\n[5/5] Fusion complete:")
        print(f"      Sentences added: {sentences_added}")
        print(f"      Models used:     {[base_model] + contributing_models}")
        print(f"      Models filtered: {filtered_models}")
        print(f"\n{'─'*60}")
        print(f"FINAL RESPONSE ({len(final_response)} chars):")
        print(f"\"{final_response}\"")
        print(f"{'─'*60}")

    return {
        "final_response":   final_response,
        "base_model":       base_model,
        "models_used":      [base_model] + contributing_models,
        "models_filtered":  filtered_models,
        "sentences_added":  sentences_added,
        "addition_log":     addition_log
    }


# ORCHESTRATOR ENTRY POINT

def run_fusion_pipeline(
    responses: dict,
    weights:   dict,
    verbose:   bool = True
) -> dict:

    # Validate: every model in responses must have a weight
    missing_weights = [m for m in responses if m not in weights]
    if missing_weights:
        raise ValueError(
            f"Models in responses have no corresponding weight: {missing_weights}. "
            f"Ensure get_fusion_weights() is called before run_fusion_pipeline()."
        )

    # Validate: at least one response present
    if not responses:
        raise ValueError("responses dict is empty — no model responses to fuse.")

    # Validate: weights sum to approximately 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(
            f"Weights do not sum to 1.0 (got {weight_sum:.4f}). "
            f"Pass normalised weights from get_fusion_weights()."
        )

    return fuse_responses(responses, weights, verbose)


# DEMO

if __name__ == "__main__":

    print("FUSION ENGINE v1 — Demo")
    print("="*60)
    print("Scenario: 3 models answered a reasoning query about spintronics.")
    print()

    # Simulated model responses — as if from the Aqua AI orchestrator
    demo_responses = {
        "gemini-3.1-flash-lite": (
            "Spintronics stores data using the magnetic spin of electrons rather than "
            "their charge. Electron spin can point either UP or DOWN, representing 1 or 0. "
            "This requires almost no energy to flip — approximately 0.14 femtojoules "
            "compared to 10 to 50 femtojoules for silicon transistors. "
            "AQQAI uses magnetite from Indian beach sand as the spintronic memory layer."
        ),
        "gpt-oss-120b": (
            "Unlike traditional silicon chips that generate heat by moving electrons, "
            "spintronics encodes information in the spin state of electrons. "
            "The result is near-zero heat generation. "
            "Spintronic memory is also non-volatile — it holds data permanently without "
            "requiring constant power, unlike DRAM which loses all data when powered off."
        ),
        "mistral-small-latest": (
            "Spintronics uses electron spin to store data. "
            "Electron spin can be UP or DOWN representing binary values. "
            "The energy required is minimal."
        )
    }

    # Weights from the Confidence Engine (normalised, sum to 1.0)
    demo_weights = {
        "gemini-3.1-flash-lite": 0.45,
        "gpt-oss-120b":          0.38,
        "mistral-small-latest":  0.17
    }

    print(f"Weights: {demo_weights}")
    print(f"Weight sum: {sum(demo_weights.values())} ✓")
    print()

    result = run_fusion_pipeline(demo_responses, demo_weights, verbose=True)

    print("\n\nSUMMARY")
    print("─"*60)
    print(f"Base model:       {result['base_model']}")
    print(f"Models used:      {result['models_used']}")
    print(f"Models filtered:  {result['models_filtered']}")
    print(f"Sentences added:  {result['sentences_added']}")
    print(f"\nFinal response:\n{result['final_response']}")

    # Edge case: model below threshold is excluded 
    print("\n\nEDGE CASE TEST — All weights above threshold")
    print("─"*60)
    low_weight_responses = {
        "gemini-3.1-flash-lite": "Spintronics is a memory technology using electron spin.",
        "gpt-oss-120b":          "It stores data without heat generation.",
        "mistral-small-latest":  "Spin-based storage is energy efficient."
    }
    low_weights = {
        "gemini-3.1-flash-lite": 0.70,
        "gpt-oss-120b":          0.20,
        "mistral-small-latest":  0.10   # below 0.15 — will be filtered
    }
    result2 = run_fusion_pipeline(low_weight_responses, low_weights, verbose=False)
    print(f"mistral-small-latest weight = 0.10 (below threshold {WEIGHT_THRESHOLD})")
    print(f"Models filtered: {result2['models_filtered']} ✓")
    print(f"Final: {result2['final_response']}")

    print("\n✅ Fusion Engine v1 demo complete.")