"""
AQQAI — Task Analyzer (task_analyzer.py)
=========================================
Classifies a user query into one of 6 task types using keyword matching.

Task types:
  - coding
  - factual
  - reasoning
  - summary
  - creative
  - general  (fallback when nothing matches clearly)

How it works:
  Each task type has a list of keywords/phrases. The query is lowercased and
  checked for substring matches against every keyword list. The type with the
  most matches wins. Ties are broken by priority order (coding > factual >
  reasoning > summary > creative > general). If no keywords match at all,
  returns "general".

Usage:
  from task_analyzer import analyze_task

  task_type = analyze_task("Write a Python function to reverse a linked list")
  # → "coding"

  task_type = analyze_task("What is the capital of France?")
  # → "factual"
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────
# KEYWORD LISTS
# Each entry is a substring that will be searched in the
# lowercased query. Phrases work just as well as single words.
# ──────────────────────────────────────────────────────────

TASK_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code", "function", "bug", "debug", "python", "javascript", "typescript",
        "java", "c++", "c#", "rust", "golang", "script", "error", "implement",
        "algorithm", "class", "method", "array", "loop", "recursion", "api",
        "library", "framework", "database", "sql", "query", "json", "xml",
        "html", "css", "react", "django", "flask", "fastapi", "docker",
        "kubernetes", "git", "github", "compile", "runtime", "syntax",
        "exception", "stacktrace", "refactor", "unit test", "async", "thread",
        "endpoint", "request", "response", "variable", "import", "module",
        "package", "pip", "npm", "build", "deploy", "fix this", "what's wrong",
        "how to write", "write a program", "write a script",
    ],
    "factual": [
        "what is", "what are", "who is", "who are", "when did", "when was",
        "where is", "where are", "how many", "how much", "define", "definition",
        "meaning of", "tell me about", "explain what", "what does",
        "what year", "which country", "capital of", "founded", "invented",
        "discovered", "born", "died", "population", "distance", "size of",
        "height of", "speed of", "temperature", "formula", "chemical",
        "element", "planet", "history of", "origin of",
    ],
    "reasoning": [
        "why", "how does", "how do", "compare", "comparison", "difference between",
        "versus", " vs ", "analyse", "analyze", "analysis", "evaluate",
        "pros and cons", "advantages", "disadvantages", "trade-off", "tradeoff",
        "should i", "is it better", "which is better", "what would happen",
        "what if", "cause of", "reason for", "explain why", "justify",
        "impact of", "effect of", "consequence", "relationship between",
        "how would", "argue", "critique", "assess",
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

# Priority order — used to break ties (index 0 = highest priority)
PRIORITY_ORDER = ["coding", "factual", "reasoning", "summary", "creative", "general"]


# ──────────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────────

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

    # Count substring matches for each task type
    scores: dict[str, int] = {task: 0 for task in TASK_KEYWORDS}

    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in lowered:
                scores[task] += 1

    best_score = max(scores.values())

    # No keywords matched at all → general
    if best_score == 0:
        return "general"

    # Find all types tied at the best score, then pick by priority
    tied = [task for task, score in scores.items() if score == best_score]

    for task in PRIORITY_ORDER:
        if task in tied:
            return task

    return "general"


def analyze_task_detailed(query: str) -> dict:
    """
    Same as analyze_task() but returns full breakdown — useful for debugging
    and for logging in run.py.

    Returns:
        {
            "task_type": "coding",
            "scores": {"coding": 3, "factual": 0, ...},
            "matched_keywords": {"coding": ["code", "function", "python"], ...}
        }
    """
    if not query or not query.strip():
        return {
            "task_type": "general",
            "scores": {task: 0 for task in TASK_KEYWORDS},
            "matched_keywords": {task: [] for task in TASK_KEYWORDS},
        }

    lowered = query.lower()
    scores: dict[str, int] = {}
    matched: dict[str, list[str]] = {}

    for task, keywords in TASK_KEYWORDS.items():
        hits = [kw for kw in keywords if kw in lowered]
        scores[task] = len(hits)
        matched[task] = hits

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