"""
AQQAI — Week 5, Task 5: Pipeline Stress Test
==============================================
Runs 50 queries back-to-back through the LIVE /api/v1/query endpoint
(the actual 3-model orchestration + fusion pipeline, not the scorer in
isolation) and checks:

  1. Any crashes or unhandled errors        -> logged per-query, summarized
  2. Response times above 10 seconds        -> flagged individually
  3. Any model adapter failing >3/50 times  -> flagged in summary
  4. Memory usage staying stable            -> sampled every N queries

Usage:
    python stress_test.py
    python stress_test.py --url http://localhost:8000 --container aqqai_api
    python stress_test.py --queries 100 --delay 0.5

Requires the API to already be running (docker-compose up -d, or
python run.py serve / uvicorn main:app, whichever you use locally).

Output:
    - Live progress printed to terminal
    - Full results written to outputs/stress_test_<timestamp>.json
    - Human-readable summary written to outputs/stress_test_<timestamp>.md
"""

import argparse
import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests

# ──────────────────────────────────────────────────────────
# 50 test queries — spread across all 6 task types so the
# stress test exercises every code path (classifier, all 3
# Layer 2 branches, fusion of different response shapes, etc.),
# not just one category repeated 50 times.
# ──────────────────────────────────────────────────────────

QUERIES: list[str] = [
    # coding (10)
    "Write a Python function to check if a number is prime",
    "Write a function to reverse a linked list",
    "Fix this code: for i in range(10) print(i)",
    "Write a Python script that reads a CSV and prints row count",
    "Implement binary search in Python",
    "Write a function to find duplicates in a list",
    "Debug this: def add(a, b) return a + b",
    "Write a Python class for a basic stack with push and pop",
    "Write a function to check if a string is a palindrome",
    "Write a script to count word frequency in a text file",

    # factual (10)
    "What is the capital of Japan?",
    "Who invented the telephone?",
    "What is the boiling point of water in Celsius?",
    "How many continents are there?",
    "What year did World War II end?",
    "What is the population of Canada?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "How far is the Moon from Earth?",
    "What is the tallest mountain in the world?",

    # reasoning (10)
    "Why is the sky blue?",
    "Should I use React or Vue for a small project?",
    "Compare Python and JavaScript for backend development",
    "What is the best substitute for butter in baking?",
    "Why does ice float on water?",
    "Give me the pros and cons of remote work",
    "What would happen if bees went extinct?",
    "Is it better to rent or buy a house?",
    "What is the difference between TCP and UDP?",
    "Why do we dream?",

    # summary (7)
    "Summarize the plot of Romeo and Juliet in 2 sentences",
    "Give a 3-bullet TLDR of what cloud computing is",
    "Summarize the water cycle in one paragraph",
    "TLDR: what is photosynthesis",
    "Give me the key points of the theory of relativity",
    "Summarize the causes of World War I briefly",
    "In short, how does the stock market work",

    # creative (7)
    "Write a 4-line poem about the ocean",
    "Write a short story about a robot learning to paint",
    "Write an opening line for a mystery novel",
    "Describe a sunset using only sensory details",
    "Write a haiku about autumn",
    "Create a product description for a smart backpack",
    "Write a metaphor comparing time to a river",

    # general (6)
    "Hello",
    "How's it going?",
    "Thanks for the help",
    "Can you help me with something?",
    "Good morning",
    "What can you do?",
]


def get_container_memory_mb(container: str) -> float | None:
    """
    Sample the container's current memory usage via `docker stats`.
    Returns None (not an error) if not running in Docker or the
    container name doesn't match — memory sampling is best-effort,
    it shouldn't fail the whole stress test if unavailable.
    """
    try:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        # format: "123.4MiB / 2GiB"
        mem_str = out.stdout.strip().split("/")[0].strip()
        value = float(mem_str.replace("MiB", "").replace("GiB", "").strip())
        if "GiB" in mem_str:
            value *= 1024
        return round(value, 1)
    except Exception:
        return None


def run_stress_test(base_url: str, container: str, num_queries: int, delay: float) -> dict:
    results: list[dict] = []
    model_failures: dict[str, int] = {}
    memory_samples: list[dict] = []
    slow_queries: list[dict] = []
    crashes: list[dict] = []

    print(f"Starting stress test — {num_queries} queries against {base_url}")
    print(f"Memory sampling from container: {container} (best-effort)\n")

    start_mem = get_container_memory_mb(container)
    if start_mem is not None:
        print(f"Starting memory: {start_mem} MiB\n")
    else:
        print("Memory sampling unavailable (not running in Docker, or container name doesn't match) — "
              "crash/timing checks still run normally.\n")

    for i in range(num_queries):
        query = QUERIES[i % len(QUERIES)]
        row = {
            "index":      i + 1,
            "query":      query,
            "success":    False,
            "status_code": None,
            "elapsed_ms": None,
            "task_type":  None,
            "error":      None,
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{base_url}/api/v1/query",
                json={"query": query, "user_id": "stress_test"},
                timeout=30,
            )
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            row["elapsed_ms"]  = elapsed_ms
            row["status_code"] = resp.status_code

            if resp.status_code == 200:
                data = resp.json()
                row["success"]   = True
                row["task_type"] = data.get("task_type")

                for model_resp in data.get("model_responses", []):
                    mid = model_resp.get("model_id", "unknown")
                    if not model_resp.get("success", True):
                        model_failures[mid] = model_failures.get(mid, 0) + 1

                if elapsed_ms > 10_000:
                    slow_queries.append({"index": i + 1, "query": query, "elapsed_ms": elapsed_ms})
                    print(f"  [{i+1}/{num_queries}] SLOW ({elapsed_ms}ms) | {query[:60]}")
                else:
                    print(f"  [{i+1}/{num_queries}] OK ({elapsed_ms}ms) | task={row['task_type']} | {query[:60]}")
            else:
                row["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
                crashes.append(row)
                print(f"  [{i+1}/{num_queries}] FAILED — HTTP {resp.status_code} | {query[:60]}")

        except requests.exceptions.Timeout:
            row["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
            row["error"] = "Request timed out (>30s client-side limit)"
            crashes.append(row)
            print(f"  [{i+1}/{num_queries}] TIMEOUT | {query[:60]}")

        except Exception as e:
            row["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
            row["error"] = f"{type(e).__name__}: {str(e)}"
            crashes.append(row)
            print(f"  [{i+1}/{num_queries}] CRASH — {type(e).__name__}: {e} | {query[:60]}")
            traceback.print_exc()

        results.append(row)

        # sample memory every 5 queries
        if (i + 1) % 5 == 0:
            mem = get_container_memory_mb(container)
            if mem is not None:
                memory_samples.append({"after_query": i + 1, "memory_mb": mem})

        if delay > 0:
            time.sleep(delay)

    end_mem = get_container_memory_mb(container)

    # ── analysis ────────────────────────────────────────
    successful = [r for r in results if r["success"]]
    failed     = [r for r in results if not r["success"]]

    adapters_over_threshold = {m: c for m, c in model_failures.items() if c > 3}

    memory_growth = None
    memory_flagged = False
    if len(memory_samples) >= 2:
        memory_growth = round(memory_samples[-1]["memory_mb"] - memory_samples[0]["memory_mb"], 1)
        # flag if memory grew by more than 50% of its starting value —
        # a rough heuristic for "climbing indefinitely" vs normal fluctuation
        if memory_samples[0]["memory_mb"] > 0 and memory_growth > memory_samples[0]["memory_mb"] * 0.5:
            memory_flagged = True

    summary = {
        "timestamp":              datetime.now().isoformat(),
        "total_queries":          num_queries,
        "successful":             len(successful),
        "failed_or_crashed":      len(failed),
        "crash_rate_pct":         round(len(failed) / num_queries * 100, 1),
        "slow_queries_over_10s":  slow_queries,
        "model_failure_counts":   model_failures,
        "adapters_over_threshold": adapters_over_threshold,   # >3/50 failures
        "start_memory_mb":        start_mem,
        "end_memory_mb":          end_mem,
        "memory_samples":         memory_samples,
        "memory_growth_mb":       memory_growth,
        "memory_flagged":         memory_flagged,
        "avg_response_ms":        round(sum(r["elapsed_ms"] for r in successful) / len(successful), 1) if successful else None,
        "max_response_ms":        max((r["elapsed_ms"] for r in successful), default=None),
    }

    return {"summary": summary, "results": results, "crashes": crashes}


def write_report(data: dict, outdir: Path) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = outdir / f"stress_test_{ts}.json"
    md_path   = outdir / f"stress_test_{ts}.md"

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    s = data["summary"]
    lines = [
        f"# AQQAI Stress Test Report — {s['timestamp']}",
        "",
        "## Summary",
        f"- Total queries: {s['total_queries']}",
        f"- Successful: {s['successful']}",
        f"- Failed/crashed: {s['failed_or_crashed']} ({s['crash_rate_pct']}%)",
        f"- Avg response time: {s['avg_response_ms']} ms",
        f"- Max response time: {s['max_response_ms']} ms",
        "",
        "## 1. Crashes / unhandled errors",
    ]

    if data["crashes"]:
        lines.append(f"**{len(data['crashes'])} crash(es) found:**\n")
        for c in data["crashes"]:
            lines.append(f"- Query #{c['index']}: `{c['query'][:70]}` — {c['error']}")
    else:
        lines.append("None — all queries returned a valid response.")

    lines += ["", "## 2. Slow queries (>10s)"]
    if s["slow_queries_over_10s"]:
        lines.append(f"**{len(s['slow_queries_over_10s'])} flagged:**\n")
        for q in s["slow_queries_over_10s"]:
            lines.append(f"- Query #{q['index']} ({q['elapsed_ms']}ms): `{q['query'][:70]}`")
    else:
        lines.append("None — every query returned under 10 seconds.")

    lines += ["", "## 3. Model adapter reliability (>3/50 failures flagged)"]
    if s["model_failure_counts"]:
        for m, c in s["model_failure_counts"].items():
            flag = " ⚠️ OVER THRESHOLD" if m in s["adapters_over_threshold"] else ""
            lines.append(f"- {m}: {c} failure(s){flag}")
    else:
        lines.append("No model adapter failures recorded.")

    lines += ["", "## 4. Memory stability"]
    if s["start_memory_mb"] is not None:
        lines.append(f"- Start: {s['start_memory_mb']} MiB")
        lines.append(f"- End: {s['end_memory_mb']} MiB")
        lines.append(f"- Growth: {s['memory_growth_mb']} MiB")
        lines.append("- ⚠️ FLAGGED — memory grew significantly, investigate for a leak" if s["memory_flagged"]
                      else "- Stable — no significant unbounded growth detected.")
        if s["memory_samples"]:
            lines.append("\n| After query | Memory (MiB) |\n|---|---|")
            for sample in s["memory_samples"]:
                lines.append(f"| {sample['after_query']} | {sample['memory_mb']} |")
    else:
        lines.append("Memory sampling unavailable this run (not running in Docker, or "
                      "container name didn't match `--container`). Re-run with the correct "
                      "container name to get memory data, or check `docker stats` manually "
                      "during a run.")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return json_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQQAI Task 5 — pipeline stress test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--container", default="aqqai_api", help="Docker container name for memory sampling")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries to run")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between queries (seconds)")
    parser.add_argument("--outdir", default="outputs", help="Directory to write the report to")
    args = parser.parse_args()

    result = run_stress_test(args.url, args.container, args.queries, args.delay)

    json_path, md_path = write_report(result, Path(args.outdir))

    print("\n" + "=" * 60)
    print("STRESS TEST COMPLETE")
    print("=" * 60)
    s = result["summary"]
    print(f"Successful: {s['successful']}/{s['total_queries']}")
    print(f"Crashes: {s['failed_or_crashed']}")
    print(f"Slow (>10s): {len(s['slow_queries_over_10s'])}")
    print(f"Adapters over failure threshold: {list(s['adapters_over_threshold'].keys()) or 'none'}")
    print(f"Memory flagged: {s['memory_flagged']}")
    print(f"\nFull report: {json_path}")
    print(f"Summary: {md_path}")