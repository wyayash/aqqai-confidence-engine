import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"
QUERY_ENDPOINT = f"{BASE_URL}/api/v1/query"
SCORE_ENDPOINT = f"{BASE_URL}/api/v1/score"

PROMPTS = [
    {"task_type": "factual", "query": "What is the capital of Japan?"},
    {"task_type": "summary", "query": "Summarize in one sentence: 'The library was filled with students preparing for exams. Some were reading quietly, others typing on laptops, and a few discussing notes in hushed voices.'"},
    {"task_type": "creative", "query": "Give a creative analogy explaining how a neural network learns, for a 10-year-old."},
    {"task_type": "creative", "query": "Write a 3-line poem about the ocean."},
    {"task_type": "factual", "query": "Why do leaves change color in autumn?"}
]

results = {"round_1": [], "round_2": []}

def run_tests(round_name):
    print(f"\n--- Starting {round_name} ---")
    for i, p in enumerate(PROMPTS, 1):
        print(f"Testing Prompt {i}/5: {p['query'][:30]}...")
        
        start_time = time.time()
        query_res = requests.post(QUERY_ENDPOINT, json={"query": p["query"], "task_type": p["task_type"], "user_id": "vineet_test"})
        total_time_ms = (time.time() - start_time) * 1000
        
        if query_res.status_code != 200:
            print(f"Error on query: {query_res.text}")
            continue
            
        query_data = query_res.json()
        
        best_model = None
        best_score = -1
        for m in query_data.get("model_responses", []):
            if m.get("success") and m.get("weighted_score", 0) > best_score:
                best_score = m.get("weighted_score", 0)
                best_model = m
                
        fused_text = query_data.get("final_response", "")
        score_res = requests.post(SCORE_ENDPOINT, json={"query": p["query"], "response": fused_text, "task_type": p["task_type"]})
        
        fused_scores = score_res.json() if score_res.status_code == 200 else {}

        results[round_name].append({
            "prompt": p["query"],
            "total_time_ms": round(total_time_ms, 2),
            "raw_query_response": query_data,
            "comparison": {
                "best_single_model": best_model.get("model_id") if best_model else "None",
                "best_single_model_score": best_score,
                "fused_response_scores": fused_scores
            }
        })

run_tests("round_1")
print("\nWaiting 2 seconds before Round 2...\n")
time.sleep(2)
run_tests("round_2")

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nTests complete! All data saved to results.json")