"""
AQQAI — Fusion Engine (fusion_engine.py)
Synthesizes responses from multiple LLMs into a single optimal answer.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

print("[System] Loading Sentence-Transformers model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("[System] Loading NLI model for contradiction detection...")
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')

# TASK 1: FORMAT CONSTRAINT HANDLING

FORMAT_KEYWORDS = [
    'one sentence', '1 sentence', 'in a sentence',
    'one line', '1 line', 'single line',
    'one paragraph', '1 paragraph',
    'one word', '1 word',
    r'\d+ sentences',
    r'\d+ lines',
    r'\d+ words',
    r'\d+ bullet',
    r'\d+-line',
    'haiku', 'poem', 'limerick', 
]

def has_format_constraint(query: str) -> bool:
    """Returns True if the query contains a format constraint."""
    query_lower = query.lower()
    for kw in FORMAT_KEYWORDS:
        if re.search(r'\b' + kw + r'\b', query_lower):
            return True
    return False

def check_compliance(query: str, text: str) -> bool:
    text = text.strip()
    query_lower = query.lower()

    #check exact word count
    word_match = re.search(r'(\d+)\s*word', query_lower)
    if word_match:
        required_words = int(word_match.group(1))
        actual_words = len(text.split())
        return actual_words == required_words

    #check exact sentence count
    sentence_match = re.search(r'(\d+)\s*sentence', query_lower)
    if sentence_match:
        required_sentences = int(sentence_match.group(1))
        actual_sentences = len([s for s in re.split(r'(?<=[.!?]) +', text) if s.strip()])
        return actual_sentences == required_sentences

    #check exact line or bullet count
    line_match = re.search(r'(\d+)\s*(?:line|bullet)', query_lower)
    if line_match:
        required_lines = int(line_match.group(1))
        actual_lines = len([line for line in text.split('\n') if line.strip()])
        return actual_lines == required_lines

    #check Haiku constraint (must be exactly 3 lines)
    if "haiku" in query_lower:
        actual_lines = len([line for line in text.split('\n') if line.strip()])
        return actual_lines == 3

    return True

def pick_format_compliant_response(scored_responses: dict, weights: dict, query: str) -> str:
    """Checks compliance and returns the best single response to prevent stacking."""
    sorted_models = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    
    for model in sorted_models:
        text = scored_responses.get(model, "")
        if check_compliance(query, text):
            return text
            
    print("[Log] Constraint violation: No models perfectly followed the format.")
    return scored_responses.get(sorted_models[0], "")


# TASK 2: CODE BLOCK HANDLING

def contains_code_block(response: str) -> bool:
    """Checks if the response contains code elements."""
    return '```' in response or response.strip().startswith('def ') \
           or response.strip().startswith('class ') \
           or response.strip().startswith('import ')

def blend_non_code_sections(scored_responses: dict, weights: dict) -> str:
    """Strips out code blocks and blends only the remaining text explanations."""
    text_only_responses = {}
    for model, text in scored_responses.items():
        # Remove markdown code blocks using regex before blending
        clean_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text_only_responses[model] = clean_text.strip()
        
    return blend_responses(text_only_responses, weights)


# V1 SEMANTIC BLENDING LOGIC

def extract_sentences(text: str) -> list[str]:
    """Splits text into sentences while preserving paragraph breaks and structure."""
    # Removed the |\n+ so the engine stops destroying line breaks!
    parts = re.split(r'(?<=[.!?]) +', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 5]

def blend_responses(scored_responses: dict, weights: dict) -> str:
    """v1 Fusion Engine — weighted sentence blending using semantic similarity & NLI."""
    sorted_models = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    if not sorted_models:
        return ""
        
    base_model = sorted_models[0]
    base_text = scored_responses.get(base_model, "")
    
    fused_sentences = extract_sentences(base_text)
    
    if fused_sentences:
        fused_embeddings = embedder.encode(fused_sentences).tolist()
    else:
        fused_embeddings = []
        
    for model in sorted_models[1:]:
        weight = weights.get(model, 0.0)
        if weight < 0.15:
            continue
            
        candidate_text = scored_responses.get(model, "")
        candidate_sentences = extract_sentences(candidate_text)
        
        if not candidate_sentences:
            continue
            
        # LATENCY OPTIMIZATION: Batch encode all candidate sentences at once
        candidate_embs = embedder.encode(candidate_sentences)
        
        for idx, sentence in enumerate(candidate_sentences):
            if not fused_embeddings:
                # NLI CHECK: Verify against base_text before adding
                nli_score = nli_model.predict([(base_text, sentence)])[0]
                if nli_score.argmax() != 0:  # 0 is contradiction
                    fused_sentences.append(sentence)
                    fused_embeddings = embedder.encode([sentence]).tolist()
                continue
                
            # Use the batch-encoded embedding
            candidate_emb = candidate_embs[idx].reshape(1, -1)
            similarities = cosine_similarity(candidate_emb, fused_embeddings)[0]
            max_similarity = np.max(similarities)
            
            if max_similarity < 0.75:
                # NLI CHECK: Verify against base_text before adding
                nli_score = nli_model.predict([(base_text, sentence)])[0]
                if nli_score.argmax() != 0:  # 0 is contradiction
                    fused_sentences.append(sentence)
                    fused_embeddings.append(candidate_emb[0].tolist())
                    
    return " ".join(fused_sentences)

# MASTER FUSION ROUTER

def fuse_responses(scored_responses: dict, weights: dict, query: str) -> str:
    """Master router: Checks constraints, handles code, then blends if safe."""
    
    # 1. Format Constraints
    if has_format_constraint(query):
        return pick_format_compliant_response(scored_responses, weights, query)
        
    # 2. Code Block Handling
    code_models = [m for m, text in scored_responses.items() if contains_code_block(text)]
    
    if code_models:
        # Pick the highest-weighted model that has code
        best_model = max(code_models, key=lambda m: weights.get(m, 0.0))
        best_code = scored_responses[best_model]
         
        # Blend text explanations from other responses
        other_responses = {m: text for m, text in scored_responses.items() if m != best_model}
        explanation = blend_non_code_sections(other_responses, weights)
        
        if explanation:
            return best_code + '\n\n' + explanation
        return best_code

    # 3. Normal Prose — blend as before
    return blend_responses(scored_responses, weights)

# LIVE API DEMO BLOCK
if __name__ == "__main__":
    import os
    import sys
    import asyncio
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    
    # 1. Point to Task Analyzer
    sys.path.append(os.path.abspath("Evaluation layer"))
    from task_analyzer import analyze_task
    
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
        
        if not query.strip():
            return
            
        # --- NEW: Task Analysis Step ---
        task_type = analyze_task(query)
        print(f"\n[0/3] Task Analysis: Classified as '{task_type.upper()}'")
            
        print("\n[1/3] Fetching live responses from 3 models simultaneously...")
        results = await asyncio.gather(
            fetch_api(gemini_client, "gemini-2.5-flash", query),
            fetch_api(groq_client, "llama-3.3-70b-versatile", query),
            fetch_api(mistral_client, "mistral-small-latest", query)
        )
        
        responses = {
            "gemini-2.5-flash": results[0],
            "llama-3.3-70b-versatile": results[1],
            "mistral-small-latest": results[2]
        }
        
        # Mock weights for the standalone test
        weights = {"gemini-2.5-flash": 0.40, "llama-3.3-70b-versatile": 0.35, "mistral-small-latest": 0.25}
        
        print("\n[2/3] Checking constraints & blending...")
        final_answer = fuse_responses(responses, weights, query)
        
        print("\n")
        print("FINAL SYNTHESIZED RESPONSE:")
        print(final_answer)
        print("\n")

    asyncio.run(run_live_test())