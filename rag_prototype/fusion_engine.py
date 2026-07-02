"""
AQQAI Fusion Engine (v1)
Uses sentence-transformers to semantically blend responses from multiple AI models.
"""
import os
import re
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 1. Initialize ML Model Globally (so it only loads once)
print("[System] Loading Sentence-Transformers model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_sentences(text: str) -> list:
    """Helper to split a paragraph into individual sentences."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def fuse_responses(scored_responses: dict, weights: dict) -> str:
    """
    v1 Fusion Engine — weighted sentence blending using semantic similarity.
    """
    # 1. Sort models by their weight (highest first)
    sorted_models = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    if not sorted_models:
        return ""
        
    # 2. Establish Base Answer
    base_model = sorted_models[0]
    base_text = scored_responses.get(base_model, "")
    
    fused_sentences = extract_sentences(base_text)
    
    if fused_sentences:
        fused_embeddings = embedder.encode(fused_sentences).tolist()
    else:
        fused_embeddings = []
        
    # 3. Iterate through other responses
    for model in sorted_models[1:]:
        weight = weights.get(model, 0.0)
        
        # Skip models below the confidence threshold
        if weight < 0.15:
            continue
            
        candidate_text = scored_responses.get(model, "")
        candidate_sentences = extract_sentences(candidate_text)
        
        for sentence in candidate_sentences:
            if not fused_embeddings:
                fused_sentences.append(sentence)
                fused_embeddings = embedder.encode([sentence]).tolist()
                continue
                
            candidate_emb = embedder.encode([sentence])
            similarities = cosine_similarity(candidate_emb, fused_embeddings)[0]
            max_similarity = np.max(similarities)
            
            # 4. Semantic Similarity Check (< 0.75 means it's new info!)
            if max_similarity < 0.75:
                fused_sentences.append(sentence)
                fused_embeddings.append(candidate_emb[0].tolist())
                
    # 5. Return combined response
    return " ".join(fused_sentences)

# =====================================================================
# LIVE API DEMO BLOCK (Runs when you execute `python fusion_engine.py`)
# =====================================================================
if __name__ == "__main__":
    load_dotenv()
    
    # Configure SDKs
    gemini_client = AsyncOpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    groq_client = AsyncOpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    mistral_client = AsyncOpenAI(api_key=os.getenv("MISTRAL_API_KEY"), base_url="https://api.mistral.ai/v1")

    async def fetch_api(client, model_name, prompt):
        try:
            res = await client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}])
            return res.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    async def run_live_test():
        print("\n" + "="*60)
        print("AQQAI FUSION ENGINE - LIVE TEST")
        print("="*60)
        query = input("\nEnter a prompt to test live fusion: ")
        
        print("\n[1/3] Fetching live responses from 3 models simultaneously...")
        results = await asyncio.gather(
            fetch_api(gemini_client, "gemini-2.5-flash", query),
            fetch_api(groq_client, "llama-3.3-70b-versatile", query),
            fetch_api(mistral_client, "mistral-small-latest", query)
        )
        
        responses = {
            "gemini-3.1-flash-lite": results[0],
            "gpt-oss-120b": results[1],
            "mistral-small-latest": results[2]
        }
        
        # Mock weights for the standalone test (The real pipeline gets these from Bayesian layer)
        weights = {"gemini-3.1-flash-lite": 0.40, "gpt-oss-120b": 0.35, "mistral-small-latest": 0.25}
        
        print("\n[2/3] Embedding sentences and calculating semantic overlap...")
        final_answer = fuse_responses(responses, weights)
        
        print("\n")
        print("FINAL SYNTHESIZED RESPONSE:")
        print(final_answer)
        print("\n")

    asyncio.run(run_live_test())