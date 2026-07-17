"""
AQQAI — Task Analyzer (task_analyzer.py)
=========================================
Classifies a user query into one of 6 task types using keyword matching.
"""

from __future__ import annotations

TASK_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code", "function", "bug", "debug", "python", "javascript", "typescript",
        "java", "c++", "c#", "rust", "golang", "script", "error", "implement",
        "algorithm", "class", "method", "array", "loop", "recursion", "api",
        "library", "framework", "sql", "database query", "db schema", "json", "xml",
        "html", "css", "react", "django", "flask", "fastapi", "docker",
        "kubernetes", "git", "github", "compile", "runtime", "syntax",
        "exception", "stacktrace", "refactor", "unit test", "async", "thread",
        "endpoint", "request", "response", "variable", "import", "module",
        "package", "pip", "npm", "build", "deploy", "fix this", "what's wrong",
        "how to write", "write a program", "write a script",
    ],
    "factual": [
        # ── unambiguous — safe to match anywhere in the query ──
        "define", "definition", "meaning of", "tell me about", "explain what",
        "capital of", "founded", "invented", "discovered", "born", "died",
        "population", "distance", "size of", "height of", "speed of",
        "temperature", "formula", "chemical", "element", "planet",
        "history of", "origin of", "causes of", "cause of", "gdp of",
        "who invented", "who discovered", "examples of", "types of",
        "kinds of", "list of", "name of", "names of", "which country",
    ],
    "reasoning": [
        "why", "how does", "how do", "compare", "comparison", "difference between",
        "versus", " vs ", "analyse", "analyze", "analysis", "evaluate",
        "pros and cons", "advantages", "disadvantages", "trade-off", "tradeoff",
        "should i", "is it better", "which is better", "what would happen",
        "what if", "reason for", "explain why", "justify",
        "impact of", "effect of", "consequence", "relationship between",
        "how would", "argue", "critique", "assess",
        "why does", "why did", "why is", "why are",
    ],
    "summary": [
        "summarize", "summarise", "summary", "tldr", "tl;dr", "key points",
        "main points", "brief", "overview", "recap", "highlights",
        "in short", "condense", "shorten", "bullet points of",
        "what are the main", "give me a summary", "sum up",
        "what happened in", "what is the gist",
    ],
    "creative": [
        "write a", "create a", "generate a", "poem", "story", "essay",
        "describe", "imagine", "fiction", "narrative", "dialogue",
        "script for", "song", "lyrics", "haiku", "metaphor",
        "creative", "brainstorm", "ideas for", "come up with",
        "invent", "design a", "draft a", "compose",
    ],
}

# ── ambiguous factual "starters" — these phrases are common openers for
# factual questions ("What is the capital of Japan?") but also appear
# harmlessly inside reasoning/creative/summary sentences ("What is the
# difference between X and Y?", "Describe what is happening in..."). They
# only count as a factual signal when the query actually STARTS with them
# — not whenever they show up anywhere in the text. This is what was
# causing reasoning/creative queries to tie with (and lose to) factual.
FACTUAL_ANCHOR_KEYWORDS: list[str] = [
    "what is", "what are", "who is", "who are", "when did", "when was",
    "where is", "where are", "how many", "how much", "what year",
    "what was", "what were", "how old", "how long", "how far",
    "what happened", "what is happening", "what causes",
]

import re

_FACTUAL_ANCHOR_PREFIX_CHARS = 20   # how far into the query an anchor must start


def _keyword_in(kw: str, text: str) -> bool:
    """
    Word-boundary aware keyword match. Plain substring checks (`kw in text`)
    let short keywords match inside unrelated longer words — e.g. "java"
    and "script" both match inside "javascript", inflating the coding
    score for any query that just mentions JavaScript. \b works fine for
    alphanumeric phrases; for keywords with symbols (e.g. "c++", "c#")
    fall back to plain substring matching since \b doesn't apply cleanly.
    """
    if re.fullmatch(r"[a-z0-9 ]+", kw):
        return re.search(rf"\b{re.escape(kw)}\b", text) is not None
    return kw in text

PRIORITY_ORDER = ["coding", "factual", "reasoning", "summary", "creative", "general"]


# explicit comparison/decision markers are a strong, unambiguous reasoning
# signal — but "Compare Python and JavaScript" or "Should I use React or
# Vue" was losing to coding, because "python"/"javascript"/"react" are
# (correctly) strong coding keywords on their own. Only treat them as a
# coding request if an actual coding action verb is present too.
STRONG_REASONING_MARKERS = [
    "compare", "comparison", "difference between", " vs ", "versus",
    "pros and cons", "which is better", "is it better", "should i",
    # recommendation / "best X" phrasing — "what is the best substitute for
    # daisy" is asking for a judgement call, not a factual lookup, even
    # though it opens with the "what is" anchor phrase.
    "best substitute", "substitute for", "substitute of", "alternative to",
    "alternatives to", "best option", "best choice", "which one should",
    "what's the best", "what is the best", "recommend", "recommendation",
    "suggest a", "suggestions for",
]
CODING_ACTION_VERBS = [
    "write a", "implement", "fix", "debug", "refactor", "code for",
    "how to write",
]


def analyze_task(query: str) -> str:
    """
    Classify a query into one of: coding, factual, reasoning, summary,
    creative, general.

    Args:
        query: Raw user query string.

    Returns:
        Task type as a lowercase string.

    Examples:
        >>> analyze_task("Write a Python function to sort a list")
        'coding'
        >>> analyze_task("What is the capital of Japan?")
        'factual'
        >>> analyze_task("Why is the sky blue?")
        'reasoning'
        >>> analyze_task("What causes inflation?")
        'factual'
        >>> analyze_task("Summarize this article for me")
        'summary'
        >>> analyze_task("Write me a short poem about rain")
        'creative'
        >>> analyze_task("Hello")
        'general'
    """
    if not query or not query.strip():
        return "general"

    lowered = query.lower()
    prefix  = lowered[:_FACTUAL_ANCHOR_PREFIX_CHARS]

    scores: dict[str, int] = {task: 0 for task in TASK_KEYWORDS}

    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if _keyword_in(kw, lowered):
                scores[task] += 1

    # anchor-only factual signal — only counts if the query STARTS with it
    for kw in FACTUAL_ANCHOR_KEYWORDS:
        if prefix.startswith(kw):
            scores["factual"] += 1

    has_reasoning_marker = any(m in lowered for m in STRONG_REASONING_MARKERS)
    has_coding_verb      = any(v in lowered for v in CODING_ACTION_VERBS)
    if has_reasoning_marker and not has_coding_verb:
        scores["reasoning"] += 3

    best_score = max(scores.values())

    if best_score == 0:
        return "general"

    tied = [task for task, score in scores.items() if score == best_score]

    for task in PRIORITY_ORDER:
        if task in tied:
            return task

    return "general"


def analyze_task_detailed(query: str) -> dict:
    """
    Same as analyze_task() but returns full breakdown for debugging and run.py.

    Returns:
        {
            "task_type": "factual",
            "scores": {"coding": 0, "factual": 2, ...},
            "matched_keywords": {"factual": ["what causes", "causes of"], ...}
        }
    """
    if not query or not query.strip():
        return {
            "task_type": "general",
            "scores": {task: 0 for task in TASK_KEYWORDS},
            "matched_keywords": {task: [] for task in TASK_KEYWORDS},
        }

    lowered = query.lower()
    prefix  = lowered[:_FACTUAL_ANCHOR_PREFIX_CHARS]
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}

    for task, keywords in TASK_KEYWORDS.items():
        hits = [kw for kw in keywords if _keyword_in(kw, lowered)]
        scores[task] = len(hits)
        matched[task] = hits

    anchor_hits = [kw for kw in FACTUAL_ANCHOR_KEYWORDS if prefix.startswith(kw)]
    scores["factual"] += len(anchor_hits)
    matched["factual"] += anchor_hits

    has_reasoning_marker = any(m in lowered for m in STRONG_REASONING_MARKERS)
    has_coding_verb      = any(v in lowered for v in CODING_ACTION_VERBS)
    if has_reasoning_marker and not has_coding_verb:
        scores["reasoning"] += 3
        matched["reasoning"].append("[comparison-marker boost +3]")

    best_score = max(scores.values())

    if best_score == 0:
        task_type = "general"
    else:
        tied = [task for task, score in scores.items() if score == best_score]
        task_type = next(t for t in PRIORITY_ORDER if t in tied)

    return {
        "task_type":        task_type,
        "scores":           scores,
        "matched_keywords": matched,
    }