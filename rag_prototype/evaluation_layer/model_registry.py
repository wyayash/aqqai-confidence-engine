"""
AQQAI — Shared Model Registry (model_registry.py)
=================================================
Centralized singleton loader for heavy PyTorch / HuggingFace models.

Prevents duplicate model instantiations across scorer.py, fusion_engine.py,
and bayesian_confidence_layer2.py, protecting local RAM against OOM crashes.
"""

import sys

# ──────────────────────────────────────────────────────────
# SENTENCE TRANSFORMERS (Coherence, Relevance, Agreement)
# ──────────────────────────────────────────────────────────
embedder = None
ST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    print("[ModelRegistry] Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    ST_AVAILABLE = True
    print("[ModelRegistry] SentenceTransformer loaded successfully.")
except Exception as e:
    print(f"[ModelRegistry] WARNING: SentenceTransformer failed to load: {e}")
    ST_AVAILABLE = False


# ──────────────────────────────────────────────────────────
# NLI CROSS-ENCODER (Consistency & Contradiction Detection)
# ──────────────────────────────────────────────────────────
nli_model = None
NLI_AVAILABLE = False

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    print("[ModelRegistry] Loading CrossEncoder ('cross-encoder/nli-deberta-v3-small')...")
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    NLI_AVAILABLE = True
    print("[ModelRegistry] CrossEncoder NLI loaded successfully.")
except Exception as e:
    print(f"[ModelRegistry] WARNING: NLI CrossEncoder failed to load: {e}")
    NLI_AVAILABLE = False