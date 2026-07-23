"""
AQQAI — Heuristic Scorer (scorer.py)
=====================================
Implements all four evaluation dimensions:

  R — Relevance     (TF-IDF overlap, keyword extraction)
  C — Coherence     (sentence-transformers all-MiniLM-L6-v2,
                     falls back to TF-IDF if not available)
  K — Completeness  (sub-part decomposition + length signal)
  S — Consistency   (NLI model: cross-encoder/nli-deberta-v3-small
                     falls back to keyword contradiction pairs + entity
                     conflict detection if NLI not available;
                     uncertainty phrases + repetition always run)

Week 4 additions:
  ✓ Layer 2 coding scorer  — ast.parse syntax check, exec runtime check, code presence
  ✓ Layer 2 factual scorer — hedge+fabrication detection (Tanvi's notes), numerical conflict check
  ✓ NLI upgrade for S      — cross-encoder/nli-deberta-v3-small replaces keyword contradiction
"""

import ast
import contextlib
import io
import os
import re
import string
import subprocess
import sys
import tempfile
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

_ST_MODEL     = None
_ST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL     = SentenceTransformer("all-MiniLM-L6-v2")
    _ST_AVAILABLE = True
    print("  [scorer] sentence-transformers loaded — using all-MiniLM-L6-v2 for coherence")
except Exception:
    _ST_AVAILABLE = False
    print("  [scorer] sentence-transformers not available — falling back to TF-IDF for coherence")


# ──────────────────────────────────────────────────────────
# NLI MODEL SETUP — cross-encoder/nli-deberta-v3-small
# Used for consistency (S) dimension — proper contradiction detection
# Falls back to keyword checks automatically if not available
# ──────────────────────────────────────────────────────────

_NLI_MODEL     = None
_NLI_AVAILABLE = False

try:
    from sentence_transformers.cross_encoder import CrossEncoder
    _NLI_MODEL     = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    _NLI_AVAILABLE = True
    print("  [scorer] NLI model loaded — using cross-encoder/nli-deberta-v3-small for consistency")
except Exception:
    _NLI_AVAILABLE = False
    print("  [scorer] NLI model not available — falling back to keyword checks for consistency")


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
    """Sentence similarity for coherence — uses ST model, falls back to TF-IDF."""
    if _ST_AVAILABLE and _ST_MODEL is not None:
        try:
            embeddings = _ST_MODEL.encode([sent_a, sent_b])
            sim = float(sklearn_cosine([embeddings[0]], [embeddings[1]])[0][0])
            return max(0.0, min(1.0, sim))
        except Exception:
            pass
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
    """Break a query into sub-parts by splitting on conjunctions and question words."""
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
    """HIGH ≥0.85 | MEDIUM 0.60-0.84 | LOW <0.60"""
    if score >= TIER_THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= TIER_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# ──────────────────────────────────────────────────────────
# DIMENSION 1 — RELEVANCE
# ──────────────────────────────────────────────────────────

_R_HIGH_THRESHOLD = 0.40   # top-chunk cosine sim at/above this → R = 1.0
_R_LOW_THRESHOLD  = 0.05   # top-chunk cosine sim at/below this → R = 0.0

# RECALIBRATION (Week 6): the previous thresholds (0.55/0.15) were a
# best-guess estimate — this sandbox has no Hugging Face access, so the
# max-chunk fix that introduced those numbers was never actually run
# against real embedding output before shipping.
#
# Tanvi's re-validation exposed the gap directly: reverse-engineering
# raw cosine from her 13 flagged rows (all manually rated R=1.0, across
# EVERY task category — reasoning, coding, summary, creative, factual)
# showed real cosine clustering at 0.20-0.41 for genuinely relevant
# responses. 0.55 was simply too high a bar — and it wasn't a
# category-specific gap (e.g. "creative isn't covered"), every category
# undershot by roughly the same margin. Re-tested these new thresholds
# against all 13 real flagged rows — every one moved substantially
# closer to 1.0 (e.g. CR3/Mistral, the specific case Vineet flagged as
# critical, moved from a derived-raw score of 0.12 to 0.42).
#
# Known remaining limitation: creative/analogy content (e.g. CR3) still
# doesn't reach a full 1.0 even after this recalibration — a genuinely
# on-topic creative analogy that deliberately avoids literal vocabulary
# overlap with the query can have real cosine as low as ~0.20, and no
# single threshold can both reward creative analogies properly AND
# correctly penalize truly off-topic responses — those need a different
# signal, not just a different cutoff. Flagging for a task-type-aware
# scoring path as a follow-up, not solved here.

# TF-IDF cosine runs much lower than embedding cosine for paraphrased text
# (different vocabulary → near-zero TF-IDF overlap even when semantically
# identical). Reusing the ST thresholds on TF-IDF output would collapse
# almost every fallback score to 0 — reintroducing the exact bug this fix
# removes. Separate, lower thresholds for the fallback path only.
_R_HIGH_THRESHOLD_TFIDF = 0.30
_R_LOW_THRESHOLD_TFIDF  = 0.05

# Instruction verbs at the start of a query ("write a", "explain",
# "summarize", "describe", "give me") carry no topical meaning but pull
# the query embedding toward a generic "task-request" region, dragging
# cosine similarity down against every response regardless of content.
# Stripped before embedding the query only — never the response.
_INSTRUCTION_PREFIX_RE = re.compile(
    r"^(write|create|generate|explain|describe|summarize|summarise|"
    r"tell me|give me|list|define|compose|draft|design)\s+(a|an|the|me)?\s*",
    re.IGNORECASE,
)


def _clean_query_for_embedding(query: str) -> str:
    cleaned = _INSTRUCTION_PREFIX_RE.sub("", query.strip())
    return cleaned if cleaned.strip() else query


def _scale_relevance(cosine: float, high: float = _R_HIGH_THRESHOLD, low: float = _R_LOW_THRESHOLD) -> float:
    """Linearly scale cosine similarity into [0,1] using the given thresholds."""
    if cosine >= high:
        return 1.0
    if cosine <= low:
        return 0.0
    return (cosine - low) / (high - low)


def score_relevance(query: str, response: str) -> float:
    """
    R — Relevance (Week 5 fix, revised)

    Week 5 v1 replaced keyword overlap with whole-response embedding
    cosine similarity — but Tanvi's re-validation showed this was STILL
    broken on every single row (avg gap 0.64), regardless of task type.
    The uniformity across coding/factual/creative/summary is the tell:
    it's not a category-specific bug, it's dilution. all-MiniLM-L6-v2 is
    trained on short sentence-pairs (STS-style). Embedding an entire
    multi-sentence response as one vector averages in every connector
    word, code fence, test case, and formatting artifact — diluting the
    on-topic content until cosine similarity against a short query
    settles in the 0.2-0.5 range even for a perfectly relevant answer.

    Fix: embed the query against each sentence/chunk of the response
    individually, and take the average of the top matching chunks
    (rather than one whole-response average). This asks "does any part
    of this response directly address the query" instead of "is the
    average meaning of this whole blob close to the query" — robust to
    length, extra examples, and formatting that would otherwise dilute
    a single whole-response embedding.

    Also strips leading instruction verbs ("write a", "explain",
    "summarize") from the query before embedding — these carry no
    topical content but pull the query embedding toward a generic
    "task-request" region rather than the actual subject matter.

    Falls back to whole-text TF-IDF cosine (own thresholds) if
    sentence-transformers isn't available.
    """
    if not response.strip():
        return 0.0

    clean_query = _clean_query_for_embedding(query)

    used_embeddings = False
    if _ST_AVAILABLE and _ST_MODEL is not None:
        try:
            chunks = _split_sentences(response)
            if not chunks:
                chunks = [response]

            query_emb  = _ST_MODEL.encode([clean_query])
            chunk_embs = _ST_MODEL.encode(chunks)
            sims = sklearn_cosine(query_emb, chunk_embs)[0]

            top_k = sorted(sims, reverse=True)[:3]
            cosine = float(np.mean(top_k))
            used_embeddings = True
        except Exception:
            cosine = _tfidf_cosine(query, response)
    else:
        cosine = _tfidf_cosine(query, response)

    cosine = max(0.0, min(1.0, cosine))

    if used_embeddings:
        return round(_scale_relevance(cosine, _R_HIGH_THRESHOLD, _R_LOW_THRESHOLD), 4)
    return round(_scale_relevance(cosine, _R_HIGH_THRESHOLD_TFIDF, _R_LOW_THRESHOLD_TFIDF), 4)


# ──────────────────────────────────────────────────────────
# DIMENSION 2 — COHERENCE
# ──────────────────────────────────────────────────────────

_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

_CODE_LINE_RE = re.compile(
    r"^\s*(def |class |return\b|import |from |for |while |if |elif |else:|"
    r"try:|except|print\(|@\w+|#.*)"
)


def _looks_like_fenceless_code(response: str) -> bool:
    """
    Catches code responses that don't use ``` fences — some model
    adapters return plain indented code without markdown fencing, which
    the old fence-only check missed entirely (this was the actual cause
    of coding responses like C1/Gemini scoring 0.47 instead of routing
    to score_code_coherence at all — the fix wasn't broken, it just
    never fired).
    """
    lines = [l for l in response.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    code_like = sum(
        1 for l in lines
        if _CODE_LINE_RE.match(l)
        or re.search(r"[{};:]\s*$", l.strip())
        or re.match(r"^\s{4,}\S", l)   # 4+ space indentation, typical of code
    )
    return (code_like / len(lines)) > 0.30


def _is_code_response(response: str) -> bool:
    stripped = response.strip()
    if "```" in response or stripped.startswith("def ") or stripped.startswith("class "):
        return True
    return _looks_like_fenceless_code(response)


def score_code_coherence(response: str) -> float:
    """
    Code responses: don't run sentence-flow analysis on code — a fenced
    code block full of short lines and no punctuation isn't "incoherent
    prose", it's correctly formatted code. Score based on whether there's
    explanatory text around the code (indicates a structured, coherent
    answer) rather than a bare code dump.
    """
    non_code_lines = [
        line for line in response.split("\n")
        if line.strip()
        and not line.strip().startswith("#")
        and "```" not in line
    ]
    has_explanation = len(non_code_lines) > 2
    return 0.85 if has_explanation else 0.75


def score_bullet_coherence(bullet_lines: list[str]) -> float:
    """
    Bullet-point responses: evaluate coherence within/across bullets
    instead of penalising the natural sentence breaks between them.
    Each bullet is checked for having actual content (not a fragment),
    and adjacent bullets are compared for topical continuity — but with
    a generous floor, since well-formed independent bullets are a
    coherent structure even with low adjacent similarity.
    """
    if not bullet_lines:
        return 0.75

    cleaned = [_BULLET_LINE_RE.sub("", b).strip() for b in bullet_lines]
    cleaned = [b for b in cleaned if b]

    if len(cleaned) < 2:
        return 0.80

    # substance check — well-formed bullets have real content, not just a word
    substantive = sum(1 for b in cleaned if len(b.split()) >= 3)
    substance_ratio = substantive / len(cleaned)

    if len(cleaned) >= 2:
        sims = [
            _sentence_similarity(cleaned[i], cleaned[i + 1])
            for i in range(len(cleaned) - 1)
        ]
        avg_sim = float(np.mean(sims))
    else:
        avg_sim = 0.5

    # generous floor (0.55) so independent-but-related bullets aren't
    # punished the way disjointed prose sentences would be
    score = 0.55 + 0.30 * substance_ratio + 0.15 * avg_sim
    return round(min(1.0, max(0.0, score)), 4)


_C_PROSE_HIGH_THRESHOLD = 0.35   # adjacent-sentence cosine at/above this → C = 1.0
_C_PROSE_LOW_THRESHOLD  = 0.05   # adjacent-sentence cosine at/below this → C = 0.0

# FIX (Week 6): score_prose_coherence() used to return the raw average
# adjacent-sentence cosine similarity DIRECTLY as the coherence score —
# no scaling at all (`score = round(avg_sim, 4)`). That conflates two
# different things: semantic similarity between adjacent sentences, and
# logical coherence/flow. A genuinely coherent response that moves
# through distinct sub-points (e.g. definition → example → use case)
# has LOWER adjacent-sentence similarity than a repetitive one — that's
# backwards from what C is supposed to reward.
#
# Tanvi's re-validation caught this directly: 6 responses she rated
# coherence=1.0 (clear, logical, no breaks in flow — including one she
# specifically called "one of the clearest explanations in the whole
# suite") had raw adjacent-sentence cosine of only 0.36-0.75. Every one
# of those 6 hits a full 1.0 under this threshold scaling.
#
# Known limitation: no negative (genuinely incoherent/rambling) examples
# were in this validation round, so these thresholds are calibrated
# only against known-good responses — worth adding disjointed/rambling
# test cases to the next validation pass to confirm this isn't overly
# generous on the low end.
def _scale_prose_coherence(avg_sim: float) -> float:
    if avg_sim >= _C_PROSE_HIGH_THRESHOLD:
        return 1.0
    if avg_sim <= _C_PROSE_LOW_THRESHOLD:
        return 0.0
    return (avg_sim - _C_PROSE_LOW_THRESHOLD) / (_C_PROSE_HIGH_THRESHOLD - _C_PROSE_LOW_THRESHOLD)


def score_prose_coherence(response: str) -> float:
    """Standard sentence-flow analysis for normal prose responses."""
    sentences = _split_sentences(response)

    if len(sentences) < 2:
        return 0.75

    similarities = [
        _sentence_similarity(sentences[i], sentences[i + 1])
        for i in range(len(sentences) - 1)
    ]
    avg_sim = float(np.mean(similarities))

    if _ST_AVAILABLE:
        score = _scale_prose_coherence(avg_sim)
    else:
        score = min(1.0, avg_sim * 4.0 + 0.45)

    return round(min(1.0, max(0.0, score)), 4)


def score_coherence(response: str) -> float:
    """
    C — Coherence (Week 5 fix)

    Tanvi's benchmark: C was wrong on 9/11 rows. The old scorer ran
    sentence-flow analysis on everything, including code blocks and
    bullet lists — which naturally break into short, punctuation-light
    fragments that look "incoherent" to that analysis even though the
    formatting is intentional and the response reads perfectly to a
    human (e.g. clean marketing copy scored 0.38; a working Python
    function scored 0.44).

    Now: detect format first, route accordingly.
      - Code response        → score_code_coherence()   (skip sentence flow)
      - Bullet-majority list → score_bullet_coherence()  (skip sentence flow)
      - Normal prose          → score_prose_coherence()  (original logic)
    """
    if not response.strip():
        return 0.0

    if _is_code_response(response):
        return score_code_coherence(response)

    lines = response.strip().split("\n")
    bullet_lines = [l for l in lines if _BULLET_LINE_RE.match(l)]
    if lines and (len(bullet_lines) / max(len(lines), 1)) > 0.5:
        return score_bullet_coherence(bullet_lines)

    return score_prose_coherence(response)


# ──────────────────────────────────────────────────────────
# DIMENSION 3 — COMPLETENESS
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# FORMAT CONSTRAINT DETECTION (Week 5, Task 4)
# Detects explicit format requirements in the query (e.g. "in one
# sentence", "3 bullet points") and checks whether the response
# actually follows them. Violations deduct from K (completeness) —
# a response that ignores a stated format requirement hasn't fully
# completed the task, regardless of how good the content is.
# ──────────────────────────────────────────────────────────

FORMAT_CONSTRAINTS = [
    ("sentences", re.compile(r"\bone\s+sentence\b"), 1),
    ("sentences", re.compile(r"\b(\d+)\s+sentences?\b"), None),
    ("points",    re.compile(r"\b(\d+)\s+(?:bullet\s+)?points?\b"), None),
    ("lines",     re.compile(r"\b(\d+)[- ]line\b"), None),
    ("lines",     re.compile(r"\bone\s+(?:opening\s+)?line\b"), 1),
    ("paragraphs", re.compile(r"\bone\s+paragraph\b"), 1),
    ("words",     re.compile(r"\b(\d+)\s+words?\b"), None),
]

_FORMAT_TOLERANCE = {
    # small allowances before a count is treated as a full violation
    "sentences": 0, "points": 0, "lines": 0, "paragraphs": 0, "words": 3,
}


def _count_sentences(response: str) -> int:
    parts = re.split(r"[.!?]+", response.strip())
    return len([p for p in parts if p.strip()])


def _count_words(response: str) -> int:
    return len(response.split())


def _count_points(response: str) -> int:
    lines = response.strip().split("\n")
    bullets = [l for l in lines if _BULLET_LINE_RE.match(l)]
    return len(bullets) if bullets else _count_sentences(response)


def _count_lines(response: str) -> int:
    return len([l for l in response.strip().split("\n") if l.strip()])


def _count_paragraphs(response: str) -> int:
    paras = [p for p in re.split(r"\n\s*\n", response.strip()) if p.strip()]
    return max(1, len(paras))


_COUNTERS = {
    "sentences":  _count_sentences,
    "words":      _count_words,
    "points":     _count_points,
    "lines":      _count_lines,
    "paragraphs": _count_paragraphs,
}


def extract_format_requirements(query: str) -> list[tuple[str, int]]:
    """Find every explicit format constraint stated in the query."""
    lowered = query.lower()
    found: list[tuple[str, int]] = []
    for kind, pattern, fixed_count in FORMAT_CONSTRAINTS:
        m = pattern.search(lowered)
        if not m:
            continue
        count = fixed_count if fixed_count is not None else int(m.group(1))
        if count > 0:
            found.append((kind, count))
    return found


def check_format_constraint(query: str, response: str) -> tuple[float, list[str]]:
    """
    If the query specifies a format constraint, check whether the
    response follows it.

    Returns (compliance_score, details) where compliance_score is in
    [0, 1] — 1.0 = fully compliant (or no constraint present), lower
    values indicate a proportionally larger violation. If multiple
    constraints are present, the worst compliance score wins.
    """
    requirements = extract_format_requirements(query)
    if not requirements or not response.strip():
        return 1.0, []

    worst = 1.0
    details = []

    for kind, required in requirements:
        actual = _COUNTERS[kind](response)
        tolerance = _FORMAT_TOLERANCE.get(kind, 0)
        diff = max(0, abs(actual - required) - tolerance)

        if diff == 0:
            compliance = 1.0
        else:
            compliance = max(0.0, 1.0 - diff / max(required, 1))

        if compliance < 1.0:
            details.append(
                f"format constraint violated: requested {required} {kind}, "
                f"response has {actual} (compliance={compliance:.2f})"
            )

        worst = min(worst, compliance)

    return worst, details


def score_completeness(query: str, response: str) -> float:
    """
    K — Completeness
    0.65 × sub-part coverage + 0.35 × length signal,
    minus a penalty (up to 0.30) if the query stated a format
    constraint (e.g. "in one sentence") that the response violated.

    ADJUSTMENT (Week 6): Tanvi's re-validation found K systematically
    overcounting — 3 flagged rows (C1/ChatGPT-worst, C4/DeepSeek-worst,
    F4/Gemini-worst) all hit K=1.0 when manually rated 0.9. All three
    share the same shape: correct, every sub-part technically covered,
    but shallower than the best response for that prompt ("missed edge
    cases", "less detailed", "no extra context"). The old length_signal
    saturated to 1.0 too easily (expected_words = sub_parts*40, floor
    50 — a fairly low bar), so once every sub-part was mentioned, a
    response barely past that word count got full credit with no room
    left to reflect "correct but shallow."

    Raised the saturation point (sub_parts*65, floor 80) and shifted
    weight from coverage (0.70->0.65) toward length (0.30->0.35), so
    brief-but-technically-complete responses have room to land below
    1.0 instead of maxing out immediately.

    Known limitation: this is a heuristic adjustment based on the
    pattern across 3 flagged rows, not exact-calibrated against their
    actual word counts (the validation sheet doesn't include full
    response text). K fundamentally can't measure "depth relative to
    what a thorough answer would cover" without either a reference
    response to compare against or real word-count data — this
    narrows the overcounting gap but won't perfectly replicate a
    human's judgment of thoroughness. Worth re-checking with actual
    word counts in the next validation pass.
    """
    if not response.strip():
        return 0.0

    sub_parts  = _decompose_query(query)
    resp_lower = response.lower()

    covered = sum(
        1 for part in sub_parts
        if any(kw in resp_lower for kw in _clean_tokens(part) if kw not in STOPWORDS)
    )

    coverage      = covered / max(len(sub_parts), 1)
    expected_words = max(len(sub_parts) * 65, 80)
    length_signal  = min(1.0, len(response.split()) / expected_words)

    base = 0.65 * coverage + 0.35 * length_signal

    constraint_score, _details = check_format_constraint(query, response)
    format_penalty = (1.0 - constraint_score) * 0.30   # up to 0.30 deducted

    return min(1.0, round(max(0.0, base - format_penalty), 4))


# ──────────────────────────────────────────────────────────
# DIMENSION 4 — CONSISTENCY
# ──────────────────────────────────────────────────────────

CONTRADICTION_PAIRS = [
    ("fast","slow"), ("simple","complex"), ("always","never"),
    ("increase","decrease"), ("efficient","inefficient"),
    ("safe","dangerous"), ("accurate","inaccurate"),
    ("reliable","unreliable"), ("easy","difficult"),
    ("cheap","expensive"), ("high","low"), ("large","small"),
    ("best","worst"), ("strong","weak"), ("faster","slower"),
    ("better","worse"),
]

UNCERTAINTY_PHRASES = [
    "i'm not sure", "i am not sure", "i think", "i believe",
    "might be wrong", "could be wrong", "i don't know",
    "i do not know", "to the best of my knowledge",
    "i'm not certain", "i am not certain", "i may be wrong",
    "i cannot be sure", "not 100% sure", "not entirely sure",
]


def _nli_contradiction_penalty(sentences: list[str]) -> float:
    """
    Run sentence pairs through NLI model to detect contradictions.

    Model: cross-encoder/nli-deberta-v3-small
    Output per pair: [contradiction, entailment, neutral] as raw logits
    We apply softmax to get probabilities.

    Pairs checked: adjacent sentences + first vs last.
    Penalty: 0.20 per contradiction pair with confidence > NLI_THRESHOLD.
    Cap: 0.60 total (prevents wiping score on long responses).

    ADJUSTMENT (Week 6) — UNCONFIRMED, flagging honestly: Tanvi's
    re-validation showed S underscoring on several structurally sound
    responses (e.g. a "detailed, covers mutability and use cases fully"
    response scoring S=0.6 vs manual 1.0, with no actual contradiction
    present). The specific culprit is genuinely unclear without the
    actual response text — the validation sheet only has category
    descriptions, not full response bodies, so this can't be diagnosed
    precisely the way the R/C fixes above were.

    Working hypothesis: the first-vs-last sentence check is the most
    likely source. A well-structured response that opens with one
    sub-point and closes with a different-but-related one (e.g. "lists
    are mutable..." -> "...use tuples for fixed collections") can be
    topically distant enough that an NLI model returns elevated
    non-entailment probability even with zero actual contradiction —
    NLI models are trained on direct entailment/contradiction pairs,
    not on judging whether a topic shift within one coherent response
    is contradictory.

    Raised the threshold from 0.80 to 0.90 as a conservative mitigation
    — a genuine contradiction (e.g. two models reporting different
    Nobel Prize years for the same award) should trigger very high
    confidence, comfortably above 0.90, so real contradictions should
    still be caught. This narrows the window for topic-shift false
    positives without confirmed proof it's the actual cause.

    NEEDS VERIFICATION: re-check against Tanvi's actual response text
    for the flagged S rows once available, to confirm this threshold
    change is the right lever and not just a plausible-sounding guess.
    """
    if len(sentences) < 2:
        return 0.0

    # Adjacent pairs + first vs last
    pairs = [(sentences[i], sentences[i + 1]) for i in range(len(sentences) - 1)]
    if len(sentences) > 2:
        pairs.append((sentences[0], sentences[-1]))

    try:
        scores = _NLI_MODEL.predict(pairs)   # shape: (n_pairs, 3)

        def softmax(x):
            e = np.exp(x - np.max(x))
            return e / e.sum()

        penalty       = 0.0
        NLI_THRESHOLD = 0.90   # was 0.80 — see docstring for reasoning
        NLI_PENALTY   = 0.20
        MAX_PENALTY   = 0.60

        for score_row in scores:
            probs           = softmax(score_row)
            contradiction_p = float(probs[0])   # index 0 = contradiction
            if contradiction_p > NLI_THRESHOLD:
                penalty += NLI_PENALTY
                if penalty >= MAX_PENALTY:
                    return MAX_PENALTY

        return round(penalty, 4)

    except Exception:
        return 0.0


def score_consistency(response: str) -> float:
    """
    S — Consistency
    Does the response contradict itself?

    PRIMARY (when NLI available):
        NLI check — cross-encoder/nli-deberta-v3-small
        Adjacent + first-vs-last sentence pairs
        Contradiction confidence > 0.90 → deduct 0.20 per pair
        Max NLI penalty capped at 0.60

    FALLBACK (when NLI not available):
        Keyword contradiction pairs     → deduct 0.08 each
        Named entity attribute conflict → deduct 0.15

    ALWAYS RUN:
        Uncertainty phrases             → deduct 0.08 each
        Repetition loops (4-gram × 3+) → deduct 0.20
    """
    if not response.strip():
        return 0.0

    score     = 1.0
    lower     = response.lower()
    sentences = _split_sentences(response)

    # ── Primary: NLI contradiction check ─────────────────
    if _NLI_AVAILABLE and _NLI_MODEL is not None:
        score -= _nli_contradiction_penalty(sentences)

    else:
        # ── Fallback: keyword contradiction pairs ─────────
        for w1, w2 in CONTRADICTION_PAIRS:
            if re.search(rf"\b{w1}\b", lower) and re.search(rf"\b{w2}\b", lower):
                score -= 0.08

        # ── Fallback: named entity consistency ────────────
        entity_attributes: dict[str, list[str]] = {}
        for sent in sentences:
            for entity, attribute in re.findall(
                r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:is|are|was|were)\s+(\w+)", sent
            ):
                entity_lower = entity.lower()
                if entity_lower not in entity_attributes:
                    entity_attributes[entity_lower] = []
                entity_attributes[entity_lower].append(attribute.lower())

        for attrs in entity_attributes.values():
            attr_set = set(attrs)
            for w1, w2 in CONTRADICTION_PAIRS:
                if w1 in attr_set and w2 in attr_set:
                    score -= 0.15
                    break

    # ── Always: uncertainty phrases ───────────────────────
    for phrase in UNCERTAINTY_PHRASES:
        if phrase in lower:
            score -= 0.08

    # ── Always: repetition loops ──────────────────────────
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
# ──────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```[\w]*\n?(.*?)```", re.DOTALL)

RUNTIME_CHECK_TIMEOUT = 4.0   # seconds — hard ceiling on how long a runtime check can take


def _extract_code_blocks(response: str) -> list[str]:
    return _CODE_BLOCK_RE.findall(response)


def _run_code_sandboxed(code: str, timeout: float = RUNTIME_CHECK_TIMEOUT) -> tuple[bool, str]:
    """
    Runs LLM-generated code in an ISOLATED SUBPROCESS with a hard timeout.

    SECURITY FIX (Vineet's review, item 4a): the old implementation ran
    `exec(code, {})` directly in the API process with no timeout at all.
    A response containing `while True: pass` (or any accidental/malicious
    infinite loop) would hang the worker thread handling that request
    forever — and since main.py runs scoring in FastAPI's threadpool,
    enough stuck requests would exhaust the pool and take down the whole
    API for every user, not just the one whose response triggered it.

    This runs the code in a separate OS process instead of the API's own
    process/thread, with `timeout=` on subprocess.run() as a hard kill
    switch — if the process is still running after `timeout` seconds,
    Python terminates it and raises TimeoutExpired, which we catch and
    treat as a normal runtime-check failure (same 0.15 penalty as any
    other runtime error), not a crash.

    Returns (success, error_message).
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            # keep the error message short — full tracebacks aren't
            # useful in a score breakdown, just the failure reason
            err = result.stderr.strip().splitlines()
            err_msg = err[-1] if err else f"exited with code {result.returncode}"
            return False, err_msg[:300]

        return True, ""

    except subprocess.TimeoutExpired:
        return False, f"execution exceeded {timeout}s timeout (likely an infinite loop)"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass   # best-effort cleanup — don't fail the check over a leftover temp file


def score_coding(response: str) -> dict:
    """
    Layer 2 — Coding scorer.

    Check 1 — Code presence   (penalty: 0.20) — no fenced code block found
    Check 2 — Syntax check    (penalty: 0.30) — ast.parse() throws SyntaxError
    Check 3 — Runtime check   (penalty: 0.15) — sandboxed execution fails
                                                 or times out (see
                                                 _run_code_sandboxed docstring)
    """
    penalty = 0.0
    details = []

    code_blocks = _extract_code_blocks(response)
    has_code    = len(code_blocks) > 0

    if not has_code:
        penalty += 0.20
        details.append("no code block found (-0.20)")
        return {
            "penalty": round(penalty, 4), "has_code": False,
            "syntax_ok": False, "runtime_ok": False,
            "code_blocks": 0, "details": details,
        }

    code      = code_blocks[0].strip()
    syntax_ok = False

    try:
        ast.parse(code)
        syntax_ok = True
    except SyntaxError as e:
        penalty += 0.30
        details.append(f"syntax error (-0.30): {e.msg} at line {e.lineno}")

    runtime_ok = False
    if syntax_ok:
        runtime_ok, error_msg = _run_code_sandboxed(code)
        if not runtime_ok:
            penalty += 0.15
            details.append(f"runtime error (-0.15): {error_msg}")

    if syntax_ok and runtime_ok:
        details.append("all checks passed")

    return {
        "penalty": round(penalty, 4), "has_code": has_code,
        "syntax_ok": syntax_ok, "runtime_ok": runtime_ok,
        "code_blocks": len(code_blocks), "details": details,
    }


# ──────────────────────────────────────────────────────────
# LAYER 2 — FACTUAL SCORER
# Only runs when task_type == "factual"
# ──────────────────────────────────────────────────────────

HEDGE_PHRASES = [
    # From Tanvi's hallucination notes — H23/H24/H25 exact phrases
    "i couldn't find the exact paper, but",
    "this might relate to", "this could relate to",
    "this likely covers",
    "based on the title, this paper probably explores",
    "common themes in this area include",
    "while i don't have direct access to this, themes such as",
    # Unsourced research claims 
    "research shows", "studies show", "studies indicate",
    "studies suggest", "research indicates", "research suggests",
    "this paper suggests", "this paper argues", "this paper shows",
    "according to research", "according to studies",
    "experts say", "experts believe", "scientists say",
    "scientists have found", "it has been shown that",
    "it is widely accepted that", "evidence suggests",
    "data suggests", "findings suggest",
    "the literature suggests", "the evidence shows",
    # Speculation-after-hedge
    "it probably", "it likely", "it seems likely",
    "it would seem", "it may be that", "one could argue",
    "it is possible that", "presumably", "in all likelihood",
]

_HEDGE_CONTINUATION_MIN_WORDS = 8


def _detect_hedge_then_fabrication(response: str) -> list[str]:
    """
    Tanvi's core finding: flag hedge phrase + long continuation (≥8 words).
    Short continuations are borderline ok. Long ones signal fabrication.
    """
    lower   = response.lower()
    flagged = []

    for phrase in HEDGE_PHRASES:
        idx = lower.find(phrase)
        while idx != -1:
            after = response[idx + len(phrase):]
            m     = re.match(r"([^.!?]*[.!?]?\s*[^.!?]*[.!?]?)", after)
            if m:
                continuation = m.group(0).strip()
                if len(continuation.split()) >= _HEDGE_CONTINUATION_MIN_WORDS:
                    flagged.append(
                        f"hedge+fabrication: '{phrase}' followed by "
                        f"{len(continuation.split())}-word continuation"
                    )
            idx = lower.find(phrase, idx + 1)

    return flagged


def _detect_numerical_conflicts(response: str) -> list[str]:
    """
    Flag same number appearing in two different low-overlap contexts.
    Catches contradictory statistics in the same response.
    """
    number_pattern  = re.compile(r"\b\d[\d,]*\.?\d*\b")
    sentences       = _split_sentences(response)
    number_contexts: dict[str, list[tuple[int, str]]] = {}
    flagged         = []

    for i, sent in enumerate(sentences):
        for match in number_pattern.finditer(sent):
            num   = match.group(0).replace(",", "")
            start = max(0, match.start() - 40)
            end   = min(len(sent), match.end() + 40)
            ctx   = sent[start:end].strip()
            number_contexts.setdefault(num, []).append((i, ctx))

    for num, occurrences in number_contexts.items():
        if len(occurrences) < 2:
            continue
        for j in range(len(occurrences)):
            for k in range(j + 1, len(occurrences)):
                idx_a, ctx_a = occurrences[j]
                idx_b, ctx_b = occurrences[k]
                if idx_a == idx_b:
                    continue
                tokens_a = set(_clean_tokens(ctx_a))
                tokens_b = set(_clean_tokens(ctx_b))
                if not tokens_a or not tokens_b:
                    continue
                overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
                if overlap < 0.25:
                    flagged.append(
                        f"numerical conflict: '{num}' in different contexts — "
                        f"'{ctx_a[:60]}' vs '{ctx_b[:60]}'"
                    )

    return flagged


def score_factual(response: str) -> dict:
    """
    Layer 2 — Factual scorer.

    Check 1 — Hedge + fabrication (penalty: 0.10 per instance)
    Check 2 — Numerical conflict  (penalty: 0.15 per conflict)
    Total penalty capped at 0.50.
    """
    hedge_flags = _detect_hedge_then_fabrication(response)
    num_flags   = _detect_numerical_conflicts(response)
    penalty     = min(0.50, len(hedge_flags) * 0.10 + len(num_flags) * 0.15)

    return {
        "penalty":                  round(penalty, 4),
        "hedge_flags":              hedge_flags,
        "numerical_flags":          num_flags,
        "hedge_count":              len(hedge_flags),
        "numerical_conflict_count": len(num_flags),
    }


# ──────────────────────────────────────────────────────────
# MAIN SCORER CLASS
# ──────────────────────────────────────────────────────────

class HeuristicScorer:
    """Orchestrates all four scoring dimensions into one weighted score."""

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or WEIGHTS
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total:.4f}"

    def score_one(
        self,
        query:     str,
        response:  ModelResponse,
        task_type: str = "general",
    ) -> ScoredResponse:
        """
        Layer 1 (RCKS) always runs.
        Layer 2 coding scorer runs only when task_type == 'coding'.
        Layer 2 factual scorer runs only when task_type == 'factual'.
        """
        if not response.success or not response.content:
            return ScoredResponse(
                model_id        = response.model_id,
                content         = response.content or "",
                relevance       = 0.0, coherence    = 0.0,
                completeness    = 0.0, consistency   = 0.0,
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

        # ── Layer 2 ───────────────────────────────────────
        layer2_coding  = None
        layer2_factual = None

        if task_type == "coding":
            layer2_coding = score_coding(response.content)
            weighted      = round(max(0.0, weighted - layer2_coding["penalty"]), 4)

        elif task_type == "factual":
            layer2_factual = score_factual(response.content)
            weighted       = round(max(0.0, weighted - layer2_factual["penalty"]), 4)

        return ScoredResponse(
            model_id        = response.model_id,
            content         = response.content,
            relevance       = r, coherence    = c,
            completeness    = k, consistency   = s,
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
        """Score every response. task_type passed through for Layer 2 routing."""
        return [self.score_one(query, r, task_type) for r in responses]