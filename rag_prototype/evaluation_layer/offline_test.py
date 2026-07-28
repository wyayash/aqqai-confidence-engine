import asyncio
from scorer import HeuristicScorer, ModelResponse

poem_round1 = "Waves crash in endless, salty song,\nBlue horizons stretch so wide and long,\nThe ocean hums its ancient tune."

poem_round2 = "Waves whisper secrets to the shore,\nBlue horizons, forevermore,\nTides dance with the moon's soft glow."

p5_text = "Leaves change color in autumn due to a combination of biochemical processes influenced by environmental factors like temperature and daylight. Here’s a step-by-step breakdown of what happens: 1. Chlorophyll Breakdown (The Green Fade). Chlorophyll is the pigment responsible for the green color in leaves, and it plays a crucial role in photosynthesis. As days shorten and temperatures drop, plants slow down chlorophyll production and begin breaking it down. Without fresh chlorophyll, the green color fades, revealing other pigments that were previously masked. 2. Revealing Hidden Pigments. Three key pigments contribute to autumn colors: Carotenoids (Yellow, Orange, Gold) are present in leaves year-round but masked by chlorophyll."

async def run_tests():
    scorer = HeuristicScorer()
    
    print("=== INVESTIGATION 1: P4 NON-DETERMINISM ===")
    
    print("\n--- Testing Poem 1 (Round 1) 5 times ---")
    for i in range(5):
        mock_resp = ModelResponse(model_id="test", content=poem_round1)
        score = scorer.score_one(query="Write a 3-line poem about the ocean.", response=mock_resp, task_type="creative")
        print(f"Run {i+1} | Poem 1 Completeness (K) Score: {score.completeness}")
        
    print("\n--- Testing Poem 2 (Round 2) 5 times ---")
    for i in range(5):
        mock_resp = ModelResponse(model_id="test", content=poem_round2)
        score = scorer.score_one(query="Write a 3-line poem about the ocean.", response=mock_resp, task_type="creative")
        print(f"Run {i+1} | Poem 2 Completeness (K) Score: {score.completeness}")

    print("\n=== INVESTIGATION 2: S-SCORE HARSHNESS ===")
    mock_resp_p5 = ModelResponse(model_id="test_p5", content=p5_text)
    score_p5 = scorer.score_one(query="Why do leaves change color in autumn?", response=mock_resp_p5, task_type="reasoning")
    print(f"P5 Consistency (S) Score: {score_p5.consistency}")

if __name__ == "__main__":
    asyncio.run(run_tests())