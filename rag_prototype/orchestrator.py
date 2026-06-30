import os
import time
import asyncio
import random # Added for mock evaluation
from dotenv import load_dotenv

# ── API SDK IMPORTS ────────────────────────────────────────────────────────
# We have completely unified the architecture! 
# Google, Groq, and Mistral all support the standard OpenAI SDK format.
from openai import AsyncOpenAI

# ── CUSTOM PIPELINE IMPORTS ────────────────────────────────────────────────
# Adjusted to exactly match the filenames in your folder!
from bayesian_confidence_layer2 import PriorStore, update_priors, get_fusion_weights
from fusion_engine import run_fusion_pipeline

# ── 1. LOAD ENVIRONMENT VARIABLES ──────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# ── 2. CONFIGURE CLIENTS ───────────────────────────────────────────────────

# Configure Gemini (Using OpenAI SDK pointed to Google's compatibility URL)
gemini_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Configure Groq (Using the OpenAI SDK pointed to Groq's free servers)
groq_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Configure Mistral (Using OpenAI SDK pointed to Mistral's URL)
mistral_client = AsyncOpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1"
)

# ── 3. ASYNC FETCH FUNCTIONS ───────────────────────────────────────────────

async def fetch_gemini(prompt: str) -> str:
    """Fetches a response from Google's Gemini."""
    try:
        response = await gemini_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Gemini Error]: {str(e)}"

async def fetch_groq(prompt: str) -> str:
    """Fetches a response from Groq's Llama-3 model."""
    try:
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Groq Error]: {str(e)}"

async def fetch_mistral(prompt: str) -> str:
    """Fetches a response from Mistral's small model."""
    try:
        response = await mistral_client.chat.completions.create(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Mistral Error]: {str(e)}"


# ── 4. MAIN FAN-OUT ORCHESTRATOR ───────────────────────────────────────────

async def generate_all_responses(prompt: str, verbose: bool = True) -> dict:
    """
    Fires off requests to all 3 models at the exact same time.
    Waits for the slowest one to finish, then returns all responses.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"ORCHESTRATOR FAN-OUT: Contacting 3 live models simultaneously...")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*60}\n")
    
    start_time = time.time()

    # asyncio.gather runs all tasks concurrently
    results = await asyncio.gather(
        fetch_gemini(prompt),
        fetch_groq(prompt),
        fetch_mistral(prompt)
    )

    elapsed_time = time.time() - start_time

    # Map the results back to their respective model names
    responses = {
        "gemini-3.1-flash-lite": results[0], 
        "gpt-oss-120b":          results[1], 
        "mistral-small-latest":  results[2]
    }

    if verbose:
        print(f"✅ All models returned successfully in {elapsed_time:.2f} seconds!")
        
    return responses


# ── 5. MOCK EVALUATION LAYER ───────────────────────────────────────────────

def mock_evaluate_responses(responses: dict) -> dict:
    """
    Simulates the Evaluation Layer.
    In the future, this will grade answers against RAG context.
    For now, it assigns a random heuristic score between 0.4 and 0.95.
    """
    print("\n" + "="*60)
    print("EVALUATION LAYER: Grading responses...")
    print("="*60)
    scores = {}
    for model in responses.keys():
        # Simulating a grading process
        score = round(random.uniform(0.40, 0.95), 2)
        scores[model] = score
        print(f"  {model} received score: {score}")
    return scores


# ── FULL PIPELINE EXECUTION ────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Initialize the Bayesian Confidence Store
    store = PriorStore()
    task_type = "general"
    
    # Initialization and Schema Migration Fix:
    # 1. Ensure the store structure exists
    if not hasattr(store, 'priors'):
        store.priors = {}
    
    for model in ["gemini-3.1-flash-lite", "gpt-oss-120b", "mistral-small-latest"]:
        if model not in store.priors:
            store.priors[model] = {}
        
        # 2. Migration: Convert legacy float/int or fix missing dict keys
        if task_type not in store.priors[model] or isinstance(store.priors[model].get(task_type), (float, int)):
            store.priors[model][task_type] = {
                "prior": float(store.priors[model].get(task_type, 0.70)),
                "update_history": [],
                "history": [] # Explicitly add 'history' to satisfy the internal API
            }
        
        # 3. Ensure 'history' exists even if we already had a dictionary
        if "history" not in store.priors[model][task_type]:
            store.priors[model][task_type]["history"] = []
    
    test_query = "In exactly two short sentences, explain what a Quantum Computer is."
    
    # 2. Phase 2: Fan-out to get live responses
    live_responses = asyncio.run(generate_all_responses(test_query))
    
    # 3. Phase 3: Pass through the Evaluation Layer (Mock)
    eval_scores = mock_evaluate_responses(live_responses)
    
    # 4. Phase 3b: Bayesian Confidence Layer (Get Weights & Update Priors)
    print("\n" + "="*60)
    print("BAYESIAN CONFIDENCE LAYER: Updating Priors & Calculating Weights...")
    print("="*60)
    
    # Executing the individual functions from your bayesian_confidence_layer2.py
    update_priors(store, task_type, eval_scores, verbose=False)
    weights = get_fusion_weights(store, task_type, list(eval_scores.keys()))
    
    for model, weight in weights.items():
        print(f"  {model} normalized weight: {weight}")
    
    # 5. Phase 4: Fusion Engine (Synthesize final answer)
    print("\n" + "="*60)
    print("FUSION ENGINE: Blending answers based on trust weights...")
    print("="*60)
    
    # Using run_fusion_pipeline from your fusion_engine.py
    fusion_result = run_fusion_pipeline(live_responses, weights, verbose=False)
    
    print("\n" + "★"*60)
    print("FINAL SYNTHESIZED RESPONSE (DELIVERED TO USER):")
    print("-"*60)
    print(f"{fusion_result['final_response']}\n")
    print(f"(Base Model: {fusion_result['base_model']} | Sentences Appended: {fusion_result['sentences_added']})")