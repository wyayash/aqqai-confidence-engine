# AQQAI System Handover & Technical Walkthrough

## 1. Bayesian Confidence Engine
**Purpose:** Dynamically adjusts how much we "trust" each model based on its historical performance (measured by the Heuristic Scorer's RCKS dimensions), rather than relying on static weights.
* **How Weights are Computed:** Every time a model successfully returns a response, the RCKS score (Layer 1) and any penalty deductions (Layer 2 coding/factual) form a final `weighted_score`. The Bayesian engine takes this score and updates the model's "prior" (historical average). 
* **How to Tune Thresholds:** Priors are currently stored in memory (`priors.json`). To make the system react faster to sudden model degradation, increase the learning rate/update multiplier in the Bayesian update function. To make it more stable, lower it.

## 2. Fusion Engine & Routing Logic
**Purpose:** Blends the best elements of all three parallel model responses into a single optimal answer, governed by Bayesian weights.
* **The Routing Decision (3 Paths):**
    1. *Format Constraints:* If `has_format_constraint` detects strict rules (e.g., "3 lines", "haiku"), the router bypasses blending entirely and picks the highest-weighted model that perfectly followed the rule to prevent formatting corruption.
    2. *Code Blocks:* If the task is `coding` or backticks are detected, it isolates the code block from the best model, blends only the prose explanations from the other models, and stitches them back together.
    3. *Prose/General:* Standard semantic blending.
* **Blending Mechanism:** The engine takes the highest-weighted response as the `base_text`. It iterates through candidate sentences from the other models. It checks for redundancy using `cosine_similarity` (< 0.75 threshold) and checks for contradictions against the base text using the NLI model. If the sentence is novel and non-contradictory, it gets appended to "Additional Insights" (capped at `MAX_ADDITIONS_PER_MODEL` to prevent bloat). 

## 3. Skill Selector Service (Task Analyzer)
**Purpose:** Routes the query to the correct evaluation and fusion paths by classifying the user's intent.
* **Logic:** The `analyze_task()` function (called first in the Orchestrator) uses keyword matching and heuristics to classify prompts into `coding`, `factual`, `reasoning`, `summary`, `creative`, or `general`. 
* **Impact:** This classification dictates whether Layer 2 scoring runs (e.g., sandboxed code execution for `coding`, or hedge-phrase detection for `factual`), and informs the Fusion Engine on whether to look for code blocks.

## 4. Known Gaps & Future Work
* **Creative Keyword Coverage:** The Completeness (K) scorer still struggles with highly creative prompts (like analogies or poems) where a model uses beautiful synonyms instead of the literal prompt keywords.
* **Production Database:** The system currently uses an in-memory `store` dict for request logs and Bayesian priors. This must be migrated to PostgreSQL or Redis before horizontal scaling.
* **Latency Spikes:** The NLI CrossEncoder is accurate but computationally heavy. Heavy prose responses can cause latency spikes during the contradiction checks in both `scorer.py` and `fusion_engine.py`.