"""
AQQAI — Model Adapters (adapters.py)
=====================================
3 free model adapters:
  1. Gemini 2.0 Flash  — Google AI Studio  (own format)
  2. Cerebras          — Llama 4 Scout     (OpenAI-compatible)
  3. Groq              — Llama 3.3 70B     (OpenAI-compatible, truly free)

Keys loaded from .env file automatically.

.env file should contain:
  GEMINI_API_KEY=...     from aistudio.google.com
  CEREBRAS_API_KEY=...   from cloud.cerebras.ai
  GROQ_API_KEY=...       from console.groq.com
"""

import json
import os
import time
import urllib.request
import urllib.error
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from scorer import ModelResponse

load_dotenv()


# ──────────────────────────────────────────────────────────
# BASE ADAPTER
# ──────────────────────────────────────────────────────────

class BaseModelAdapter(ABC):
    model_id:    str
    timeout:     int = 30
    max_retries: int = 3

    @abstractmethod
    def send_query(self, query: str) -> ModelResponse:
        ...

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        """
        HTTP POST with retry + exponential backoff.
        Retryable:     429, 500, 502, 503, 504
        Non-retryable: 400, 401, 402, 403, 404, 422
        """
        NON_RETRYABLE = {400, 401, 402, 403, 404, 422}
        data = json.dumps(body).encode("utf-8")

        for attempt in range(self.max_retries):
            try:
                # Add default headers, then merge caller's headers on top
                default_headers = {
                    "User-Agent": "python-requests/2.31.0",
                    "Accept": "application/json",
}
                merged_headers = {**default_headers, **headers}
                req = urllib.request.Request(
                    url, data=data, headers=merged_headers, method="POST"
)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))

            except urllib.error.HTTPError as e:
                if e.code in NON_RETRYABLE:
                    # No point retrying — auth/billing issue
                    raise RuntimeError(
                        f"HTTP {e.code} from {self.model_id} — "
                        f"{'check API key' if e.code in {401,403} else 'check account credits/billing' if e.code == 402 else 'bad request'}"
                    )
                wait = 2 ** attempt
                print(f"  [{self.model_id}] HTTP {e.code} — retry {attempt+1}/{self.max_retries} in {wait}s")
                time.sleep(wait)

            except (urllib.error.URLError, TimeoutError) as e:
                wait = 2 ** attempt
                print(f"  [{self.model_id}] Network error — retry {attempt+1}/{self.max_retries} in {wait}s: {e}")
                time.sleep(wait)

        raise RuntimeError(f"{self.model_id} failed after {self.max_retries} attempts")

    def _failed(self, error: str) -> ModelResponse:
        """Return a failed ModelResponse instead of raising — keeps pipeline alive."""
        print(f"  [{self.model_id}] FAILED: {error}")
        return ModelResponse(
            model_id=self.model_id,
            content="",
            latency_ms=0.0,
            success=False,
            error=error,
        )


# ──────────────────────────────────────────────────────────
# OPENAI-COMPATIBLE BASE
# Cerebras and Groq both use this exact same format
# ──────────────────────────────────────────────────────────

class OpenAICompatibleAdapter(BaseModelAdapter):
    base_url:   str = ""
    api_key:    str = ""
    model_name: str = ""

    def send_query(self, query: str) -> ModelResponse:
        if not self.api_key or self.api_key.endswith("_here"):
            return self._failed(
                f"No API key set for {self.model_id} — add it to your .env file."
            )

        start = time.time()
        try:
            raw = self._post(
                url=self.base_url,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                body={
                    "model":       self.model_name,
                    "messages":    [{"role": "user", "content": query}],
                    "max_tokens":  600,
                    "temperature": 0.7,
                },
            )
            content    = raw["choices"][0]["message"]["content"].strip()
            latency_ms = round((time.time() - start) * 1000, 2)
            return ModelResponse(
                model_id=self.model_id,
                content=content,
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as e:
            return self._failed(str(e))


# ──────────────────────────────────────────────────────────
# ADAPTER 1 — GEMINI 3.5 FLASH
# Uses Google's own request/response format
# Free: 1000 req/day, 15 req/min
# Key from: aistudio.google.com → Get API Key
# ──────────────────────────────────────────────────────────

class GeminiAdapter(BaseModelAdapter):
    model_id   = "gemini-3.1-flash-lite"
    model_name = "gemini-3.1-flash-lite"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def send_query(self, query: str) -> ModelResponse:
        if not self.api_key or self.api_key.endswith("_here"):
            return self._failed("No GEMINI_API_KEY in .env file.")

        start = time.time()
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model_name}:generateContent?key={self.api_key}"
            )
            raw = self._post(
                url=url,
                headers={"Content-Type": "application/json"},
                body={
                    "contents": [{"parts": [{"text": query}]}],
                    "generationConfig": {
                        "maxOutputTokens": 600,
                        "temperature":     0.7,
                    },
                },
            )
            content    = raw["candidates"][0]["content"]["parts"][0]["text"].strip()
            latency_ms = round((time.time() - start) * 1000, 2)
            return ModelResponse(
                model_id=self.model_id,
                content=content,
                latency_ms=latency_ms,
                success=True,
            )
        except Exception as e:
            return self._failed(str(e))


# ──────────────────────────────────────────────────────────
# ADAPTER 2 — CEREBRAS (Llama 4 Scout)
# Fastest free inference — 2600+ tokens/sec
# Free: 1M tokens/day, no credit card
# Key from: cloud.cerebras.ai → API Keys
# ──────────────────────────────────────────────────────────

class CerebrasAdapter(OpenAICompatibleAdapter):
    model_id   = "cerebras-gpt-oss"
    model_name = "gpt-oss-120b"
    base_url   = "https://api.cerebras.ai/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("CEREBRAS_API_KEY", "")


# ──────────────────────────────────────────────────────────
# ADAPTER 3 — GROQ (Llama 3.3 70B)
# Truly free — no credit card, no expiring credits
# Free: 14,400 req/day, 30 req/min
# Key from: console.groq.com → API Keys → Create API Key
# ──────────────────────────────────────────────────────────

# class GroqAdapter(OpenAICompatibleAdapter):
#     model_id   = "groq-llama3"
#     model_name = "llama-3.3-70b-versatile"
#     base_url   = "https://api.groq.com/openai/v1/chat/completions"

#     def __init__(self):
#         self.api_key = os.getenv("GROQ_API_KEY", "")

class MistralAdapter(OpenAICompatibleAdapter):
    model_id   = "mistral-small"
    model_name = "mistral-small-latest"
    base_url   = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY", "")

# ──────────────────────────────────────────────────────────
# REGISTRY
# To add a new model: write one adapter class, add it here
# Nothing else in the codebase changes
# ──────────────────────────────────────────────────────────

ALL_ADAPTERS: list[BaseModelAdapter] = [
    GeminiAdapter(),
    CerebrasAdapter(),
    MistralAdapter(),
]