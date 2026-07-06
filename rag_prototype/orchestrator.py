"""
AQQAI Master Orchestrator
Connects Task Analysis, Model Fan-Out, Evaluation, Bayesian Memory, and Fusion.
"""
import os
import sys
import time
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 1. Handle folder path for Jeet's files
sys.path.append(os.path.abspath("Evaluation layer"))

# 2. Import Jeet's Modules
from task_analyzer import analyze_task
from scorer import HeuristicScorer, ModelResponse

# 3. Import Yashveer's Modules
from bayesian_confidence_layer2 import PriorStore, process_confidence_request
from fusion_engine import fuse_responses

load_dotenv()

# Initialize SDK Clients
gemini_client = AsyncOpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
groq_client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
mistral_client = AsyncOpenAI(api_key=os.getenv("MISTRAL_API_KEY"), base_url="https://api.mistral.ai/v1")

async def fetch_api(client, model_name: str, prompt: str) -> ModelResponse:
    """Fetches API response and formats it into Jeet's ModelResponse dataclass."""
    start = time.time()
    try:
        res = await client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
        content = res.choices[0].message.content
        latency = (time.time() - start) * 1000
        return ModelResponse(model_id=model_name, content=content, latency_ms=latency, success=True)
    except Exception as e:
        latency = (time.time() - start) * 1000
        return ModelResponse(model_id=model_name, content="", latency_ms=latency, success=False, error=str(e))

async def run_pipeline():
    print("\n" + "="*60)
    print("AQQAI UNIFIED PIPELINE")
    print("="*60)
    
    query = input("\nEnter your query: ")
    if not query.strip():
        return

    # STEP 1: Task Analysis (Jeet's Code)
    task_type = analyze_task(query)
    print(f"\n[1/5] Task Analysis: Classified as '{task_type}'")

    # STEP 2: Model Fan-Out
    print("[2/5] Fan-Out: Fetching Gemini, Llama-3, and Mistral...")
    responses_list = await asyncio.gather(
        fetch_api(gemini_client, "gemini-2.5-flash", query),
        fetch_api(groq_client, "llama-3.3-70b-versatile", query),
        fetch_api(mistral_client, "mistral-small-latest", query)
    )

    # STEP 3: Evaluation Layer (Jeet's Code)
    print("[3/5] Evaluation: Running Heuristic Scorer (RCKS)...")
    scorer = HeuristicScorer()
    scored_results = scorer.score_all(query, responses_list, task_type)
    
    # Extract weights for the Bayesian layer
    eval_scores = {res.model_id: res.weighted_score for res in scored_results if res.success}
    raw_responses = {res.model_id: res.content for res in scored_results if res.success}

    if not eval_scores:
        print("Pipeline Failed: All models returned errors.")
        return

    # STEP 4: Bayesian Memory Layer (Your Code)
    print("[4/5] Memory: Updating Bayesian Confidence Priors...")
    store = PriorStore("priors.json")
    payload = {
        "task_type": task_type,
        "eval_scores": eval_scores,
        "responses": raw_responses
    }
    bayesian_output = process_confidence_request(payload, store)
    
    # STEP 5: Fusion Layer (Your Code)
    print("[5/5] Synthesizer: Blending responses via Machine Learning...\n")
    final_answer = fuse_responses(raw_responses, bayesian_output["weights"])

    print("\n")
    print("FINAL SYNTHESIZED RESPONSE:")
    print("\n")
    print(final_answer)
    print("\nPipeline Execution Complete.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())