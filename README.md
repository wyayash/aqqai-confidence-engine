# AQQAI: AI Orchestration & Fusion Pipeline

## Overview
This repository contains the core orchestration layer for the Aqua AI (AQQAI) architecture. It is a multi-model pipeline designed to route user queries to specialized live AI models, mathematically evaluate their outputs in real-time using a continuously updating Bayesian trust system, and synthesize the results into a cohesive final answer.

## Core Architecture

The pipeline consists of three primary nodes working in tandem:

### 1. The Brain: `orchestrator.py`
* **Asynchronous Fan-Out:** Uses `asyncio` to simultaneously query multiple live LLM APIs (Google Gemini, Meta Llama-3 via Groq, Mistral), reducing overall latency to the speed of the slowest model.
* **Traffic Routing:** Manages the flow of data between the generation, evaluation, and synthesis layers.
* **Fault Tolerance:** Gracefully catches API timeouts or quota limits without crashing the pipeline.

### 2. The Memory: `bayesian_confidence_layer.py`
* **Bayesian Trust Updates:** Uses Bayes' Theorem to adjust the historical reliability (Prior) of each model based on incoming evaluation scores.
* **Volatility Dampening:** Applies an Exponential Moving Average (EMA) learning rate (`0.20`) to prevent single-query hallucinations from destroying a model's reputation.
* **Weight Normalization:** Converts raw Bayesian posteriors into percentage-based trust weights that sum to 1.0 for the Fusion Engine.
* **Audit Logging:** Maintains a rolling history of the last 50 updates per model/task type.

### 3. The Synthesizer: `fusion_engine_v1.py`
* **Heuristic Sentence Blending (v1):** Sorts responses by their Bayesian trust weight.
* **Base Selection:** The highest-trusted model's response is selected as the structural Base Answer.
* **Novelty Extraction:** Sentences from lower-weighted models are scanned for semantic overlap. Novel information is appended; redundant information is discarded.

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/wyayash/aqqai-confidence-engine.git](https://github.com/wyayash/aqqai-confidence-engine.git)
   cd aqqai-confidence-engine
   ```

2. **Install dependencies:**
   ```bash
   pip install openai mistralai google-generativeai python-dotenv sentence-transformers
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GEMINI_API_KEY=your_google_key
   GROQ_API_KEY=your_groq_key
   MISTRAL_API_KEY=your_mistral_key
   ```

## Usage

Run the main orchestrator to execute a full 5-stage pipeline simulation:
```bash
python orchestrator.py
```
