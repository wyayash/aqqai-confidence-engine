"""
AQQAI Fusion Engine
Synthesizes responses from multiple LLMs into a single optimal answer.
"""
import re
import numpy as np
from model_registry import embedder, nli_model
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────
# FILLER FILTERING (from Jeet's fusion.py — merged in per Vineet's
# consolidation plan). Skips structural boilerplate that isn't actual
# content — "Certainly!", markdown headers, bullet markers, "In
# summary" — so the fused answer doesn't accumulate filler sentences
# from every model that used one.
# ──────────────────────────────────────────────────────────

_FILLER_PATTERNS = [
    r"^(certainly|sure|of course|absolutely|great question)[!,.]",
    r"^(here'?s?|below is|the following|in this|this is)",
    r"^(i hope|i'?m happy|feel free|let me know)",
    r"^#+\s",                        # markdown headers
    r"^\s*[-*•]\s",                  # bullet points
    r"^(in summary|to summarize|to conclude|in conclusion)",
]
_FILLER_RE = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)

MIN_SENTENCE_WORDS = 6          # filters fragments — replaces the old bare len()>5 CHAR check
MAX_ADDITIONS_PER_MODEL = 4     # caps one verbose model from dominating the fused answer

# TASK 1: FORMAT CONSTRAINT HANDLING

FORMAT_KEYWORDS = [
    'one sentence', '1 sentence', 'in a sentence',
    'one line', '1 line', 'single line',
    'one paragraph', '1 paragraph',
    'one word', '1 word',
    r'\d+[\s-]*sentences?',
    r'\d+[\s-]*words?',
    r'\d+[\s-]*bullets?',
    r'\d+[\s-]*lines?',
    'haiku', 'poem', 'limerick', 
]

def has_format_constraint(normalized_query: str) -> bool:
    """Returns True if the query contains a format constraint."""
    for kw in FORMAT_KEYWORDS:
        if re.search(r'\b' + kw + r'\b', normalized_query):
            return True
    return False

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def _clean_complete_sentences(text: str) -> str:
    """Removes trailing incomplete sentences cut off by model token limits."""
    text = text.strip()
    # Find the last sentence-ending punctuation mark
    match = re.search(r"^(.*[.!?])", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""  # <--- Changed this so it drops the fragment if no punctuation exists

def _normalize_query(query: str) -> str:
    q = query.lower()
    for word, digit in WORD_NUMBERS.items():
        q = re.sub(rf'\b{word}\b', str(digit), q)
    return q

def check_compliance(normalized_query: str, text: str) -> bool:
    text = text.strip()

    # 1. Check word count with ±10% tolerance
    word_match = re.search(r'(\d+)[\s-]*word', normalized_query)
    if word_match:
        required_words = int(word_match.group(1))
        actual_words = len(text.split())
        tolerance = max(1, int(required_words * 0.10))
        return abs(actual_words - required_words) <= tolerance

    # 2. Check exact sentence count
    sentence_match = re.search(r'(\d+)[\s-]*sentence', normalized_query)
    if sentence_match:
        required_sentences = int(sentence_match.group(1))
        actual_sentences = len([s for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()])
        return actual_sentences == required_sentences

    # 3. Check exact line or bullet count (handles spaces or hyphens like "4-line")
    line_match = re.search(r'(\d+)[\s-]*(?:line|bullet)', normalized_query)
    if line_match:
        required_lines = int(line_match.group(1))
        actual_lines = len([line for line in text.split('\n') if line.strip()])
        return actual_lines == required_lines

    # 4. Check Haiku constraint (must be exactly 3 lines)
    if "haiku" in normalized_query:
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
    """
    Strips out the PRIMARY code block only, and blends the rest.

    FIX (Vineet's baseline.json finding): the old version stripped ALL
    ``` fences from every model's text via a single regex.sub, which also
    deleted secondary/example fences (e.g. under an "### Example Usage"
    header). That left orphaned headers with nothing under them, since
    their content had already been stripped here (and separately dumped,
    stacked with every other fence, at the top by fuse_responses()).
    Only stripping the FIRST fence per model — the primary implementation,
    which is the one already pulled out to the top separately — leaves
    any secondary/example fences intact in the explanation text, attached
    to their own headers where they belong.
    """
    text_only_responses = {}
    for model, text in scored_responses.items():
        # count=1 — strip only the first fenced block, not every one
        clean_text = re.sub(r'```.*?```', '', text, count=1, flags=re.DOTALL)
        text_only_responses[model] = clean_text.strip()
        
    return blend_responses(text_only_responses, weights)


# V1 SEMANTIC BLENDING LOGIC
def extract_base_sentences(text: str) -> list[str]:
    """
    Splits the winning base text into sentences/paragraphs WITHOUT 
    stripping headers, bullets, or structural formatting.
    """
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def extract_sentences(text: str) -> list[str]:
    """
    Splits text into sentences while preserving paragraph breaks and structure.
    Filters out fragments (< MIN_SENTENCE_WORDS words) and structural
    filler ("Certainly!", markdown headers, bullets, "In summary") —
    these aren't real content and shouldn't compete for a fusion slot.
    """
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p.split()) < MIN_SENTENCE_WORDS:
            continue
        if _FILLER_RE.match(p):
            continue
        sentences.append(p)
    return sentences

def blend_responses(scored_responses: dict, weights: dict) -> str:
    """v1 Fusion Engine — weighted sentence blending using semantic similarity & NLI."""
    sorted_models = sorted(weights.keys(), key=lambda k: weights[k], reverse=True)
    if not sorted_models:
        return ""
        
    base_model = sorted_models[0]
    base_text = scored_responses.get(base_model, "")
    
    # Use lighter extraction for the base model so headers and structure remain intact
    fused_sentences = extract_base_sentences(base_text)
    base_sentence_count = len(fused_sentences)
    
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
            
        # LATENCY OPTIMIZATION
        candidate_embs = embedder.encode(candidate_sentences)

        additions = 0   # caps this model to MAX_ADDITIONS_PER_MODEL, so one
                         # verbose model can't dominate the fused answer

        for idx, sentence in enumerate(candidate_sentences):
            if additions >= MAX_ADDITIONS_PER_MODEL:
                break

            if not fused_embeddings:
                # NLI CHECK
                nli_score = nli_model.predict([(base_text, sentence)])[0]
                if nli_score.argmax() != 0:  # 0 is contradiction
                    fused_sentences.append(sentence)
                    fused_embeddings = embedder.encode([sentence]).tolist()
                    additions += 1
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
                    additions += 1
                    
    added_sentences = [cleaned for s in fused_sentences[base_sentence_count:] if (cleaned := _clean_complete_sentences(s))]
    
    if added_sentences:
        # FIX (Vineet's decision, mail 2 item 4): clean trailing incomplete
        # sentences on the APPENDED insights only — the base stays
        # untouched by this same rule, as designed (extract_base_sentences
        # deliberately preserves structure). If a candidate insight has no
        # terminal punctuation, it's a fragment cut off by the model's
        # token limit — clean or drop it here rather than let it dangle.
        cleaned_additions = []
        for s in added_sentences:
            cleaned = _clean_complete_sentences(s)
            if cleaned:
                cleaned_additions.append(cleaned)
        if cleaned_additions:
            insights = "\n* " + "\n* ".join(cleaned_additions)
            return base_text.strip() + "\n\n### Additional Insights" + insights
        return base_text.strip()
        
    # FIX (Vineet's finding, mail 3): even with the raised max_tokens
    # adapter fix above, trim any dangling incomplete final sentence on
    # the base response itself as a safety net, so a truncated base_text
    # never ships mid-sentence even if a model still hits its cap.
    return _clean_complete_sentences(base_text.strip())

# MASTER FUSION ROUTER
def fuse_responses(scored_responses: dict, weights: dict, query: str, task_type: str = "general") -> str:
    """Master router: Checks constraints, handles code, then blends if safe."""
    
    # --- NEW: Filter out API errors before doing anything ---
    valid_responses = {m: text for m, text in scored_responses.items() if not text.startswith("Error:")}
    
    if not valid_responses:
        return "Error: All APIs failed to return a valid response."

    # --- NEW: Normalize query ONCE here ---
    # Log: We accept that \n+ split in extract_sentences flattens paragraph structure for v1
    normalized_query = _normalize_query(query)
        
    # 1. Format Constraints (Pass the normalized query!)
    if has_format_constraint(normalized_query):
        return pick_format_compliant_response(valid_responses, weights, normalized_query)
        
    # 2. Code Block Handling
    code_models = [m for m, text in valid_responses.items() if task_type == 'coding' or contains_code_block(text)]
    
    if code_models:
        # Pick the highest-weighted model that has code
        best_model = max(code_models, key=lambda m: weights.get(m, 0.0))
        best_full_text = valid_responses[best_model]
        
        # Extract ONLY the PRIMARY markdown code block — the first one.
        # FIX (Vineet's baseline.json finding): re.findall() previously
        # grabbed EVERY fenced block (main implementation + any secondary
        # "Example Usage" snippets) and stacked them all here, which is
        # what caused 4 code blocks to appear stacked at the top while
        # their own headers were left hollow further down (see
        # blend_non_code_sections() above for the other half of this fix).
        code_blocks = re.findall(r'```.*?```', best_full_text, flags=re.DOTALL)
        
        if code_blocks:
            best_code_only = code_blocks[0]
            
            # Pass ALL models to the blender so the winning model acts as the base_text
            # This ensures cosine similarity filters out redundant sentences
            explanation = blend_non_code_sections(valid_responses, weights)
            
            return best_code_only + '\n\n' + explanation
        else:
            # Fallback: If no markdown backticks are found, safely return the raw text
            return best_full_text

    # 3. Normal Prose, blend as before
    return blend_responses(valid_responses, weights)

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
    gemini_client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    groq_client = AsyncOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    mistral_client = AsyncOpenAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1"
    )

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
            
        #NEW: Task Analysis Step
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
        final_answer = fuse_responses(responses, weights, query, task_type)
        
        print("\n")
        print("FINAL SYNTHESIZED RESPONSE:")
        print(final_answer)
        print("\n")

    asyncio.run(run_live_test())