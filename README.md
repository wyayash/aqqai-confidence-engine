# AQQAI Confidence Engine

A robust, multi-agent LLM orchestration pipeline designed to evaluate, score, and synthesize responses from multiple models simultaneously (Gemini, Llama-3, Mistral). The system dynamically weighs model reliability using Bayesian probability and intelligently blends the best elements of each response to output a single, highly optimized answer.

---

## 🏗️ System Architecture

The pipeline processes user queries through a strict 5-step lifecycle, managed by `orchestrator.py`:

### 1. Task Analysis
Classifies the incoming user query into one of six core categories:
* `coding` | `factual` | `reasoning` | `summary` | `creative` | `general`

### 2. LLM Fan-Out
Simultaneously routes the query to three leading models via asynchronous API calls:
* **Gemini-2.5-flash**
* **Llama-3.3-70b-versatile**
* **Mistral-small-latest**

### 3. Heuristic Evaluation (RCKS)
Evaluates each raw response using a dedicated scoring layer based on:
* **Relevance**
* **Coherence**
* **Completeness**
* **Consistency**

### 4. Bayesian Confidence Memory
Maintains a dynamic `priors.json` memory bank. The system calculates Inter-Model Agreement and adjusts its historical trust scores for each model based on the current task category. It outputs the final mathematical weights used to determine which model is currently the most trustworthy for the specific prompt.

### 5. Fusion Engine (Synthesizer)
Takes the weighted responses and applies context-aware synthesis:
* **Format Constraint Detection:** Automatically detects constraints (e.g., "one sentence", "4-line poem") and selects the single most compliant response to prevent format breaking.
* **Code Block Handling:** Identifies programming syntax and isolates code blocks to prevent syntax errors caused by sentence-level blending.
* **Semantic Blending:** For standard prose, uses `sentence-transformers` (`all-MiniLM-L6-v2`) and cosine similarity to extract and mathematically blend the highest-value sentences from multiple models.

---

## 🚀 Setup & Execution

### Prerequisites
Ensure you have the required API keys mapped in a `.env` file at the root of the project:
```env
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

### Running the Pipeline
To execute the full production pipeline (Task Analysis -> API -> Evaluation -> Memory -> Fusion):
```bash
python orchestrator.py
```

### Sandbox Testing
To rapidly debug the Fusion Engine's constraint and blending logic without executing the evaluation and memory layers:
```bash
python fusion_engine.py
```
