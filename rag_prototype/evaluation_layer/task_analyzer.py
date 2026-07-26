"""
AQQAI — Task Analyzer (task_analyzer.py)
=========================================
Classifies a user query into one of 6 task types using keyword matching.
"""

from __future__ import annotations
import re

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
        "what happened in", "what is the gist", "in one sentence", "1 sentence",
    ],
    "creative": [
        "write a", "create a", "generate a", "poem", "story", "essay",
        "describe", "imagine", "fiction", "narrative", "dialogue",
        "script for", "song", "lyrics", "haiku", "metaphor",
        "creative", "brainstorm", "ideas for", "come up with",
        "invent", "design a", "draft a", "compose",
    ],
}

FACTUAL_ANCHOR_KEYWORDS: list[str] = [
    "what is", "what are", "who is", "who are", "when did", "when was",
    "where is", "where are", "how many", "how much", "what year",
    "what was", "what were", "how old", "how long", "how far",
    "what happened", "what is happening", "what causes",
]

_FACTUAL_ANCHOR_PREFIX_CHARS = 20

# Action-intent tasks (summary, creative) take priority over topic nouns in tie-breakers
PRIORITY_ORDER = ["summary", "creative", "coding", "reasoning", "factual", "general"]

STRONG_REASONING_MARKERS = [
    "compare", "comparison", "difference between", " vs ", "versus",
    "pros and cons", "which is better", "is it better", "should i",
    "best substitute", "substitute for", "substitute of", "alternative to",
    "alternatives to", "best option", "best choice", "which one should",
    "what's the best", "what is the best", "recommend", "recommendation",
    "suggest a", "suggestions for",
]

STRONG_SUMMARY_MARKERS = [
    "summarize", "summarise", "summary", "tl;dr", "tldr",
    "in one sentence", "1 sentence", "condense", "sum up",
    "brief summary", "give me a summary", "bullet points of",
]

CODING_ACTION_VERBS = [
    "write a", "implement", "fix", "debug", "refactor", "code for",
    "how to write",
]


def _keyword_in(kw: str, text: str) -> bool:
    if re.fullmatch(r"[a-z0-9 ]+", kw):
        return re.search(rf"\b{re.escape(kw)}\b", text) is not None
    return kw in text


def analyze_task(query: str) -> str:
    """Classify a query into one of: coding, factual, reasoning, summary, creative, general."""
    if not query or not query.strip():
        return "general"

    lowered = query.lower()
    prefix  = lowered[:_FACTUAL_ANCHOR_PREFIX_CHARS]

    scores: dict[str, int] = {task: 0 for task in TASK_KEYWORDS}

    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if _keyword_in(kw, lowered):
                scores[task] += 1

    # Anchor-only factual signal
    for kw in FACTUAL_ANCHOR_KEYWORDS:
        if prefix.startswith(kw):
            scores["factual"] += 1

    # Explicit Reasoning Boost
    has_reasoning_marker = any(m in lowered for m in STRONG_REASONING_MARKERS)
    has_coding_verb      = any(v in lowered for v in CODING_ACTION_VERBS)
    if has_reasoning_marker and not has_coding_verb:
        scores["reasoning"] += 3

    # Explicit Summary Boost
    has_summary_marker = any(_keyword_in(m, lowered) for m in STRONG_SUMMARY_MARKERS)
    if has_summary_marker:
        scores["summary"] += 3

    best_score = max(scores.values())

    if best_score == 0:
        return "general"

    tied = [task for task, score in scores.items() if score == best_score]

    for task in PRIORITY_ORDER:
        if task in tied:
            return task

    return "general"


def analyze_task_detailed(query: str) -> dict:
    """Same as analyze_task() but returns full breakdown for debugging."""
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

    has_summary_marker = any(_keyword_in(m, lowered) for m in STRONG_SUMMARY_MARKERS)
    if has_summary_marker:
        scores["summary"] += 3
        matched["summary"].append("[summary-marker boost +3]")

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