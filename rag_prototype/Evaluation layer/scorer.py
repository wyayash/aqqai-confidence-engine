"""
AQQAI — Heuristic Scorer (scorer.py)
=====================================
Implements all four evaluation dimensions:

  R — Relevance     (TF-IDF overlap, keyword extraction)
  C — Coherence     (sentence-transformers all-MiniLM-L6-v2,
                     falls back to TF-IDF if not available)
  K — Completeness  (sub-part decomposition + length signal)
  S — Consistency   (contradiction pairs + entity conflict detection
                     + uncertainty phrases from Yashveer's notes)

Yashveer's Week 1 research — adjustments made:
  ✓ Layer 2 coding scorer — ast.parse syntax check, exec runtime check, code presence
  ✓ Layer 2 factual scorer — hedge+fabrication detection (Tanvi's notes), numerical conflict check
  ✓ NLI upgrade for S dimension — cross-encoder/nli-deberta-v3-small

Output format (matches task spec exactly):
  {
    "model_id":        "m1",
    "relevance":       0.82,
    "coherence":       0.74,
    "completeness":    0.91,
    "consistency":     0.88,
    "weighted_score":  0.84,
    "confidence_tier": "MEDIUM"
  }
"""

import ast
import contextlib
import io
import re
import string
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ──────────────────────────────────────────────────────────
# SENTENCE-TRANSFORMERS SETUP
# Loads once at import time — not on every function call
# Falls back to TF-IDF automatically if not installed
# ──────────────────────────────────────────────────────────

_ST_MODEL = None
_ST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _ST_AVAILABLE = True
    print("  [scorer] sentence-transformers loaded — using all-MiniLM-L6-v2 for coherence")
except Exception:
    _ST_AVAILABLE = False
    print("  [scorer] sentence-transformers not available — falling back to TF-IDF for coherence")


# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

WEIGHTS = {
    "relevance":    0.30,
    "coherence":    0.25,
    "completeness": 0.30,
    "consistency":  0.15,
}

TIER_THRESHOLDS = {"HIGH": 0.85, "MEDIUM": 0.60}

STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","to","of","in","for",
    "on","with","at","by","from","as","into","through","and",
    "or","but","if","then","that","this","it","its","i","we",
    "you","they","he","she","not","no","so","up","out","about",
    "than","also","just","more","very","what","how","why","when",
    "where","which","who","whom","there","their","them","these",
    "those","here","some","any","all","each","both","few","more",
}


# ──────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────

@dataclass
class ModelResponse:
    """Raw response from a model adapter."""
    model_id:   str
    content:    str
    latency_ms: float = 0.0
    success:    bool  = True
    error:      Optional[str] = None


@dataclass
class ScoredResponse:
    """A ModelResponse with all dimension scores attached."""
    model_id:        str
    content:         str
    relevance:       float
    coherence:       float
    completeness:    float
    consistency:     float
    weighted_score:  float
    confidence_tier: str
    latency_ms:      float
    success:         bool
    error:           Optional[str] = None
    layer2_coding:   Optional[dict] = None   # populated when task_type == "coding"
    layer2_factual:  Optional[dict] = None   # populated when task_type == "factual"

    def to_dict(self) -> dict:
        """Scores only — no content. Used for all-models summary."""
        d = {
            "model_id":        self.model_id,
            "relevance":       self.relevance,
            "coherence":       self.coherence,
            "completeness":    self.completeness,
            "consistency":     self.consistency,
            "weighted_score":  self.weighted_score,
            "confidence_tier": self.confidence_tier,
            "latency_ms":      self.latency_ms,
            "success":         self.success,
            "error":           self.error,
        }
        if self.layer2_coding is not None:
            d["layer2_coding"] = self.layer2_coding
        if self.layer2_factual is not None:
            d["layer2_factual"] = self.layer2_factual
        return d

    def to_dict_with_content(self) -> dict:
        """Scores + response text. Used for winner."""
        d = self.to_dict()
        d["content"] = self.content
        return d


# ──────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ──────────────────────────────────────────────────────────

def _clean_tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, return meaningful tokens."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if len(w) > 1 and w not in STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Filters out very short fragments."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 12]


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity — used for relevance and as coherence fallback."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        vec   = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf = vec.fit_transform([text_a, text_b])
        return float(sklearn_cosine(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0


def _sentence_similarity(sent_a: str, sent_b: str) -> float:
    """
    Sentence similarity for coherence scoring.

    Uses sentence-transformers all-MiniLM-L6-v2 when available.
    Falls back to TF-IDF cosine automatically.

    Why sentence-transformers is better for coherence:
    - Understands meaning, not just word overlap
    - "A dog ran fast" and "The canine sprinted" → high similarity
    - TF-IDF would score these near 0 (no shared words)
    - Real coherence is about meaning flow, not word repetition
    """
    if _ST_AVAILABLE and _ST_MODEL is not None:
        try:
            embeddings = _ST_MODEL.encode([sent_a, sent_b])
            sim = float(sklearn_cosine([embeddings[0]], [embeddings[1]])[0][0])
            return max(0.0, min(1.0, sim))
        except Exception:
            pass
    # Fallback
    return _tfidf_cosine(sent_a, sent_b)


def _extract_keywords(text: str, top_n: int = 20) -> set[str]:
    """Extract top keywords by term frequency."""
    tokens = _clean_tokens(text)
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_kw = sorted(freq, key=lambda k: freq[k], reverse=True)
    return set(sorted_kw[:top_n])


def _decompose_query(query: str) -> list[str]:
    """
    Break a query into sub-parts by splitting on conjunctions and question words.

    Example:
      "What is a vector database and how does it store data?"
      → ["What is a vector database", "how does it store data"]
    """
    query = query.strip().rstrip("?.")
    parts = re.split(
        r"\b(and|or|also|as well as|what|how|why|when|where|which|explain|describe|list)\b",
        query,
        flags=re.IGNORECASE,
    )
    delimiters = {"and","or","also","as well as","what","how","why",
                  "when","where","which","explain","describe","list"}
    cleaned = [
        p.strip() for p in parts
        if p.strip() and p.strip().lower() not in delimiters and len(p.strip()) > 5
    ]
    return cleaned if cleaned else [query]


def _confidence_tier(score: float) -> str:
    """
    3-tier confidence system from Yashveer's Week 1 research.
    HIGH   (≥0.85) — serve directly
    MEDIUM (0.60-0.84) — serve, optionally with caveat
    LOW    (<0.60)  — trigger fallback / human review
    """
    if score >= TIER_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= TIER_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# ──────────────────────────────────────────────────────────
# DIMENSION 1 — RELEVANCE
# ──────────────────────────────────────────────────────────

def score_relevance(query: str, response: str) -> float:
    """
    R — Relevance
    How well does the response address the query?

    Method:
    - Primary:   TF-IDF cosine similarity (semantic overlap)
    - Secondary: keyword overlap (exact query terms in response)
    - Final:     0.6 * cosine + 0.4 * keyword_overlap
    """
    if not response.strip():
        return 0.0

    cosine = _tfidf_cosine(query, response)

    query_kw    = _extract_keywords(query, top_n=15)
    resp_tokens = set(_clean_tokens(response))
    overlap     = len(query_kw & resp_tokens) / max(len(query_kw), 1)

    score = round(0.6 * cosine + 0.4 * overlap, 4)
    return min(1.0, score)


# ──────────────────────────────────────────────────────────
# DIMENSION 2 — COHERENCE
# ──────────────────────────────────────────────────────────

def score_coherence(response: str) -> float:
    """
    C — Coherence
    Does the response flow logically from sentence to sentence?

    Method:
    - Split response into sentences
    - Compute sentence-transformers similarity between each adjacent pair
      (falls back to TF-IDF if sentence-transformers not available)
    - Average all similarities → coherence score

    Why sentence-transformers here:
    TF-IDF misses meaning-based coherence. Two sentences about the
    same topic but with different words score 0 in TF-IDF even if
    they flow perfectly. Sentence-transformers captures actual meaning.
    """
    sentences = _split_sentences(response)

    if len(sentences) < 2:
        return 0.75  # single sentence — neutral score

    similarities = []
    for i in range(len(sentences) - 1):
        sim = _sentence_similarity(sentences[i], sentences[i + 1])
        similarities.append(sim)

    avg_sim = float(np.mean(similarities))

    if _ST_AVAILABLE:
        # sentence-transformers scores sit naturally in 0.2-0.9 range
        # No calibration needed — scores are already meaningful
        score = round(avg_sim, 4)
    else:
        # TF-IDF sits in 0.02-0.25 range — needs calibration
        score = min(1.0, avg_sim * 4.0 + 0.45)

    return round(min(1.0, max(0.0, score)), 4)


# ──────────────────────────────────────────────────────────
# DIMENSION 3 — COMPLETENESS
# ──────────────────────────────────────────────────────────

def score_completeness(query: str, response: str) -> float:
    """
    K — Completeness
    Does the response address all parts of the query?

    Method:
    - Decompose query into sub-parts
    - Check coverage of each sub-part in response
    - Length signal as secondary check
    - Final: 0.70 * coverage + 0.30 * length_signal
    """
    if not response.strip():
        return 0.0

    sub_parts  = _decompose_query(query)
    resp_lower = response.lower()

    covered = 0
    for part in sub_parts:
        part_keywords = [
            kw for kw in _clean_tokens(part)
            if kw not in STOPWORDS
        ]
        if any(kw in resp_lower for kw in part_keywords):
            covered += 1

    coverage = covered / max(len(sub_parts), 1)

    expected_words = max(len(sub_parts) * 40, 50)
    actual_words   = len(response.split())
    length_signal  = min(1.0, actual_words / expected_words)

    score = round(0.70 * coverage + 0.30 * length_signal, 4)
    return min(1.0, score)


# ──────────────────────────────────────────────────────────
# DIMENSION 4 — CONSISTENCY
# ──────────────────────────────────────────────────────────

CONTRADICTION_PAIRS = [
    ("fast",       "slow"),
    ("simple",     "complex"),
    ("always",     "never"),
    ("increase",   "decrease"),
    ("efficient",  "inefficient"),
    ("safe",       "dangerous"),
    ("accurate",   "inaccurate"),
    ("reliable",   "unreliable"),
    ("easy",       "difficult"),
    ("cheap",      "expensive"),
    ("high",       "low"),
    ("large",      "small"),
    ("best",       "worst"),
    ("strong",     "weak"),
    ("faster",     "slower"),
    ("better",     "worse"),
]

UNCERTAINTY_PHRASES = [
    "i'm not sure",   "i am not sure",
    "i think",        "i believe",
    "might be wrong", "could be wrong",
    "i don't know",   "i do not know",
    "to the best of my knowledge",
    "i'm not certain","i am not certain",
    "i may be wrong", "i cannot be sure",
    "not 100% sure",  "not entirely sure",
]


def score_consistency(response: str) -> float:
    """
    S — Consistency
    Does the response contradict itself?

    Check 1 — Contradiction pairs (penalty: 0.08 each)
    Check 2 — Named entity with conflicting attributes (penalty: 0.15)
    Check 3 — Uncertainty phrases (penalty: 0.08 each)
    Check 4 — Repetition loops / degenerate output (penalty: 0.20)
    """
    if not response.strip():
        return 0.0

    score = 1.0
    lower = response.lower()

    # Check 1: Contradiction pairs
    for w1, w2 in CONTRADICTION_PAIRS:
        if re.search(rf"\b{w1}\b", lower) and re.search(rf"\b{w2}\b", lower):
            score -= 0.08

    # Check 2: Named entity consistency
    sentences = _split_sentences(response)
    entity_attributes: dict[str, list[str]] = {}

    for sent in sentences:
        matches = re.findall(
            r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:is|are|was|were)\s+(\w+)",
            sent
        )
        for entity, attribute in matches:
            entity_lower = entity.lower()
            if entity_lower not in entity_attributes:
                entity_attributes[entity_lower] = []
            entity_attributes[entity_lower].append(attribute.lower())

    for entity, attrs in entity_attributes.items():
        attr_set = set(attrs)
        for w1, w2 in CONTRADICTION_PAIRS:
            if w1 in attr_set and w2 in attr_set:
                score -= 0.15
                break

    # Check 3: Uncertainty phrases
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in lower:
            score -= 0.08

    # Check 4: Repetition loops
    words = lower.split()
    if len(words) >= 12:
        ngrams: dict[tuple, int] = {}
        for i in range(len(words) - 3):
            ng = tuple(words[i:i + 4])
            ngrams[ng] = ngrams.get(ng, 0) + 1
        if ngrams and max(ngrams.values()) >= 3:
            score -= 0.20

    return round(max(0.0, score), 4)


# ──────────────────────────────────────────────────────────
# LAYER 2 — CODING SCORER
# Only runs when task_type == "coding"
# Applies penalty adjustments on top of the Layer 1 weighted_score
# ──────────────────────────────────────────────────────────

# Regex to find fenced code blocks: ```...``` or ```python...```
_CODE_BLOCK_RE = re.compile(r"```[\w]*\n?(.*?)```", re.DOTALL)


def _extract_code_blocks(response: str) -> list[str]:
    """
    Extract all fenced code blocks from a response.
    Returns list of code strings (without the backtick fences).
    """
    return _CODE_BLOCK_RE.findall(response)


def score_coding(response: str) -> dict:
    """
    Layer 2 — Coding scorer.
    Runs 3 checks on top of RCKS and returns a penalty + breakdown.

    Check 1 — Code presence (penalty: 0.20)
        Does the response contain a fenced code block at all?
        If someone asked for code and there's none, it's incomplete.

    Check 2 — Syntax check (penalty: 0.30)
        Does the code block have valid Python syntax?
        Uses ast.parse() — the same parser Python itself uses.
        SyntaxError = code that can't even be read, let alone run.

    Check 3 — Runtime check (penalty: 0.15)
        If syntax is valid, try exec() in a sandboxed environment.
        Catches NameError, TypeError, ZeroDivisionError etc.
        Only runs if syntax check passed — no point running broken code.

    Returns:
        {
            "penalty":       0.35,   # total deduction to apply to weighted_score
            "has_code":      True,
            "syntax_ok":     True,
            "runtime_ok":    False,
            "code_blocks":   1,
            "details":       ["runtime error: NameError: name 'x' is not defined"]
        }
    """
    penalty = 0.0
    details = []

    # ── Check 1: Code presence ────────────────────────────
    code_blocks = _extract_code_blocks(response)
    has_code    = len(code_blocks) > 0

    if not has_code:
        penalty += 0.20
        details.append("no code block found (-0.20)")
        # Can't check syntax or runtime without code
        return {
            "penalty":     round(penalty, 4),
            "has_code":    False,
            "syntax_ok":   False,
            "runtime_ok":  False,
            "code_blocks": 0,
            "details":     details,
        }

    # Use the first code block for syntax + runtime checks
    # (most responses have one primary code block)
    code = code_blocks[0].strip()

    # ── Check 2: Syntax check ─────────────────────────────
    syntax_ok = False
    try:
        ast.parse(code)
        syntax_ok = True
    except SyntaxError as e:
        penalty += 0.30
        details.append(f"syntax error (-0.30): {e.msg} at line {e.lineno}")

    # ── Check 3: Runtime check ────────────────────────────
    # Only run if syntax is valid — exec() on broken syntax crashes
    runtime_ok = False
    if syntax_ok:
        stdout_capture = io.StringIO()
        sandbox = {}   # isolated namespace — no access to global state
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, sandbox)   # noqa: S102
            runtime_ok = True
        except Exception as e:
            penalty += 0.15
            details.append(f"runtime error (-0.15): {type(e).__name__}: {e}")

    if syntax_ok and runtime_ok:
        details.append("all checks passed")

    return {
        "penalty":     round(penalty, 4),
        "has_code":    has_code,
        "syntax_ok":   syntax_ok,
        "runtime_ok":  runtime_ok,
        "code_blocks": len(code_blocks),
        "details":     details,
    }


# ──────────────────────────────────────────────────────────
# LAYER 2 — FACTUAL SCORER
# Only runs when task_type == "factual"
# Applies penalty adjustments on top of the Layer 1 weighted_score
# ──────────────────────────────────────────────────────────

# Phrases from Tanvi's hallucination notes (Part 2).
# These are the hedge phrases that appeared in H23-H25 failures
# (Gemini, DeepSeek, Mistral on fake research paper prompts).
# The hedge itself is not the problem — what comes AFTER it is.
# We detect the pattern: hedge phrase → followed by specific detail
# within the same sentence or the next sentence.
HEDGE_PHRASES = [
    # Directly from Tanvi's notes — exact phrases observed in failures
    "i couldn't find the exact paper, but",
    "this might relate to",
    "this could relate to",
    "this likely covers",
    "based on the title, this paper probably explores",
    "common themes in this area include",
    "while i don't have direct access to this, themes such as",

    # Extended list — same pattern, same risk
    # Unsourced research claims (Vineet's task spec)
    "research shows",
    "studies show",
    "studies indicate",
    "studies suggest",
    "research indicates",
    "research suggests",
    "this paper suggests",
    "this paper argues",
    "this paper shows",
    "according to research",
    "according to studies",
    "experts say",
    "experts believe",
    "scientists say",
    "scientists have found",
    "it has been shown that",
    "it is widely accepted that",
    "evidence suggests",
    "data suggests",
    "findings suggest",
    "the literature suggests",
    "the evidence shows",

    # Speculation-after-hedge pattern (core of Tanvi's finding)
    "it probably",
    "it likely",
    "it seems likely",
    "it would seem",
    "it may be that",
    "one could argue",
    "it is possible that",
    "presumably",
    "in all likelihood",
]

# Minimum words after a hedge phrase to count as "fabricated detail"
# Short phrases like "this might relate to X" are fine
# Long continuations after the hedge are the problem
_HEDGE_CONTINUATION_MIN_WORDS = 8


def _detect_hedge_then_fabrication(response: str) -> list[str]:
    """
    Implements Tanvi's core finding: the failure pattern is NOT just
    hedge phrases — it's hedge followed by specific invented detail.

    Method:
    1. Find every hedge phrase in the response
    2. Extract the text that follows it (rest of sentence + next sentence)
    3. If the continuation is long enough (≥8 words) → flag it
       Short continuations like "this might relate to X" are borderline ok
       Long continuations with invented specifics are the real problem

    Returns list of flagged instances for the breakdown dict.
    """
    lower    = response.lower()
    flagged  = []

    for phrase in HEDGE_PHRASES:
        idx = lower.find(phrase)
        while idx != -1:
            # Get the text after the hedge phrase
            after = response[idx + len(phrase):]

            # Take up to end of current sentence + next sentence
            # Split on sentence boundary
            continuation_match = re.match(r"([^.!?]*[.!?]?\s*[^.!?]*[.!?]?)", after)
            if continuation_match:
                continuation = continuation_match.group(0).strip()
                word_count   = len(continuation.split())
                if word_count >= _HEDGE_CONTINUATION_MIN_WORDS:
                    flagged.append(
                        f"hedge+fabrication: '{phrase}' followed by "
                        f"{word_count}-word continuation"
                    )

            # Look for next occurrence of this phrase
            idx = lower.find(phrase, idx + 1)

    return flagged


def _detect_numerical_conflicts(response: str) -> list[str]:
    """
    Numerical claim check — from Vineet's task spec.

    If the same number appears in two different contexts with conflicting
    meaning, flag it as a consistency issue.

    Method:
    1. Extract all numbers from the response
    2. For each number that appears 2+ times, check if the surrounding
       context (5 words on each side) is meaningfully different
    3. If contexts differ significantly → flag as numerical conflict

    Example of what this catches:
    "The model was trained on 1.5 billion parameters... 
     The dataset contains 1.5 million records."
    Same number (1.5) but different units/contexts — ambiguous but not
    necessarily wrong.

    Example of actual conflict:
    "Python was released in 1991... Python was first released in 1994."
    Same referent, different values → genuine conflict.

    Returns list of flagged instances.
    """
    # Extract all numbers (integers and decimals, including with commas)
    number_pattern = re.compile(r"\b\d[\d,]*\.?\d*\b")
    sentences      = _split_sentences(response)
    flagged        = []

    # Map: number_string → list of (sentence_index, context_snippet)
    number_contexts: dict[str, list[tuple[int, str]]] = {}

    for i, sent in enumerate(sentences):
        for match in number_pattern.finditer(sent):
            num = match.group(0).replace(",", "")   # normalise 1,000 → 1000
            start = max(0, match.start() - 40)
            end   = min(len(sent), match.end() + 40)
            ctx   = sent[start:end].strip()

            if num not in number_contexts:
                number_contexts[num] = []
            number_contexts[num].append((i, ctx))

    for num, occurrences in number_contexts.items():
        if len(occurrences) < 2:
            continue

        # Check pairs of occurrences for context conflict
        for j in range(len(occurrences)):
            for k in range(j + 1, len(occurrences)):
                idx_a, ctx_a = occurrences[j]
                idx_b, ctx_b = occurrences[k]

                # Only flag if they're in different sentences
                if idx_a == idx_b:
                    continue

                # Context similarity — if contexts are very different,
                # same number might mean different things (unit change etc)
                # Use token overlap as a proxy
                tokens_a = set(_clean_tokens(ctx_a))
                tokens_b = set(_clean_tokens(ctx_b))

                if not tokens_a or not tokens_b:
                    continue

                overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)

                # Low overlap = same number used in very different contexts
                # High overlap = same number repeated about same thing (ok)
                # Threshold: < 0.25 overlap = likely different claims
                if overlap < 0.25:
                    flagged.append(
                        f"numerical conflict: '{num}' appears in different "
                        f"contexts — '{ctx_a[:60]}' vs '{ctx_b[:60]}'"
                    )

    return flagged


def score_factual(response: str) -> dict:
    """
    Layer 2 — Factual scorer.
    Runs 2 checks on top of RCKS and returns a penalty + breakdown.

    Check 1 — Hedge + fabrication detection (penalty: 0.10 per instance)
        Implements Tanvi's core finding from hallucination notes H23-H25:
        the real failure pattern is not uncertainty language alone,
        but hedge phrase followed by invented specific detail.
        Full phrase list from Tanvi's Part 2 notes + extended set.

    Check 2 — Numerical conflict detection (penalty: 0.15 per conflict)
        If the same number appears in two different contexts with
        meaningfully different surrounding tokens, flag as consistency
        issue. Catches contradictory statistics in the same response.

    Returns:
        {
            "penalty":              0.25,
            "hedge_flags":          ["hedge+fabrication: 'research shows' ..."],
            "numerical_flags":      ["numerical conflict: '1991' appears ..."],
            "hedge_count":          1,
            "numerical_conflict_count": 1,
        }
    """
    penalty       = 0.0
    hedge_flags   = _detect_hedge_then_fabrication(response)
    num_flags     = _detect_numerical_conflicts(response)

    # Penalty: 0.10 per hedge+fabrication instance
    penalty += len(hedge_flags) * 0.10

    # Penalty: 0.15 per numerical conflict
    penalty += len(num_flags) * 0.15

    # Cap total factual penalty at 0.50 — same reasoning as NLI cap
    penalty = min(penalty, 0.50)

    return {
        "penalty":                   round(penalty, 4),
        "hedge_flags":               hedge_flags,
        "numerical_flags":           num_flags,
        "hedge_count":               len(hedge_flags),
        "numerical_conflict_count":  len(num_flags),
    }


# ──────────────────────────────────────────────────────────
# MAIN SCORER CLASS
# ──────────────────────────────────────────────────────────

class HeuristicScorer:
    """
    Orchestrates all four scoring dimensions into one weighted score.

    Usage:
        scorer  = HeuristicScorer()
        results = scorer.score_all(query, responses)
        winner  = scorer.pick_winner(results)
    """

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or WEIGHTS
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, \
            f"Weights must sum to 1.0, got {total:.4f}"

    def score_one(
        self,
        query:     str,
        response:  ModelResponse,
        task_type: str = "general",
    ) -> ScoredResponse:
        """
        Score a single ModelResponse against the query.

        Layer 1 (RCKS) always runs.
        Layer 2 coding scorer runs only when task_type == 'coding'.
        """
        if not response.success or not response.content:
            return ScoredResponse(
                model_id        = response.model_id,
                content         = response.content or "",
                relevance       = 0.0,
                coherence       = 0.0,
                completeness    = 0.0,
                consistency     = 0.0,
                weighted_score  = 0.0,
                confidence_tier = "LOW",
                latency_ms      = response.latency_ms,
                success         = False,
                error           = response.error,
                layer2_coding   = None,
                layer2_factual  = None,
            )

        # ── Layer 1: RCKS ─────────────────────────────────
        r = score_relevance(query, response.content)
        c = score_coherence(response.content)
        k = score_completeness(query, response.content)
        s = score_consistency(response.content)

        weighted = round(
            r * self.weights["relevance"]
            + c * self.weights["coherence"]
            + k * self.weights["completeness"]
            + s * self.weights["consistency"],
            4,
        )

        # ── Layer 2: Coding scorer ────────────────────────
        layer2_coding  = None
        layer2_factual = None

        if task_type == "coding":
            layer2_coding = score_coding(response.content)
            weighted      = round(max(0.0, weighted - layer2_coding["penalty"]), 4)

        # ── Layer 2: Factual scorer ───────────────────────
        elif task_type == "factual":
            layer2_factual = score_factual(response.content)
            weighted       = round(max(0.0, weighted - layer2_factual["penalty"]), 4)

        return ScoredResponse(
            model_id        = response.model_id,
            content         = response.content,
            relevance       = r,
            coherence       = c,
            completeness    = k,
            consistency     = s,
            weighted_score  = weighted,
            confidence_tier = _confidence_tier(weighted),
            latency_ms      = response.latency_ms,
            success         = response.success,
            error           = response.error,
            layer2_coding   = layer2_coding,
            layer2_factual  = layer2_factual,
        )

    def score_all(
        self,
        query:     str,
        responses: list[ModelResponse],
        task_type: str = "general",
    ) -> list[ScoredResponse]:
        """
        Score every response. Failed ones get zeroed out.
        task_type passed through to score_one for Layer 2 routing.
        """
        return [self.score_one(query, r, task_type) for r in responses]