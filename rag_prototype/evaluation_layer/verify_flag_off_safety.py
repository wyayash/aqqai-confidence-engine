"""
Safety verification for the Skill Selector integration.

Run this AFTER wiring skill_selector_client.py into main.py on the
skill-selector branch, with SKILL_INJECTION left at its default ("off").

It sends the same set of queries to the running container twice-comparable:
  1. Once against a baseline you capture from `main` BEFORE this branch's
     changes (or a saved reference file of expected responses).
  2. Once against the skill-selector branch with the flag off.

If any response differs, the flag-off safety property is violated and this
must be fixed before claiming the integration is "done" per Vineet's mail.

Usage:
    # Step 1 — on `main`, capture a baseline (run this once, before switching):
    python verify_flag_off_safety.py --capture --out baseline.json

    # Step 2 — switch to skill-selector branch, rebuild, SKILL_INJECTION=off,
    # then compare:
    python verify_flag_off_safety.py --compare --baseline baseline.json
"""

import argparse
import json
import sys
import requests

TEST_QUERIES = [
    {"query": "Give a 3-bullet TLDR of what cloud computing is."},
    {"query": "Explain how the water cycle works in two sentences."},
    {"query": "Write a haiku about the ocean."},
    {"query": "What is a vector database in simple terms?"},
    {"query": "Summarize the causes of World War 1."},
    {"query": "Write a Python function to reverse a string."},
    {"query": "What are the pros and cons of remote work?"},
    {"query": "Explain photosynthesis in one paragraph."},
    {"query": "Give me a 4-line poem about autumn."},
    {"query": "How does a neural network learn?"},
]

ENDPOINT = "http://localhost:8000/api/v1/query"


def run_queries():
    results = []
    for q in TEST_QUERIES:
        try:
            resp = requests.post(ENDPOINT, json=q, timeout=60)
            body = resp.json()
            results.append({
                "query": q["query"],
                "status_code": resp.status_code,
                "final_response": body.get("final_response"),
                "task_type": body.get("task_type"),
            })
        except Exception as e:
            results.append({"query": q["query"], "error": str(e)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", default="baseline.json")
    ap.add_argument("--baseline", default="baseline.json")
    args = ap.parse_args()

    if args.capture:
        print(f"Capturing baseline from {ENDPOINT} ...")
        results = run_queries()
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} responses to {args.out}")
        return

    if args.compare:
        with open(args.baseline) as f:
            baseline = json.load(f)

        print(f"Running the same {len(TEST_QUERIES)} queries against the "
              f"skill-selector branch (flag off) ...")
        current = run_queries()

        mismatches = []
        for b, c in zip(baseline, current):
            if b.get("query") != c.get("query"):
                mismatches.append((b.get("query"), "query order mismatch — rerun"))
                continue
            if b.get("final_response") != c.get("final_response"):
                mismatches.append((b["query"], "final_response DIFFERS"))
            if b.get("status_code") != c.get("status_code"):
                mismatches.append((b["query"], "status_code DIFFERS"))

        print()
        if not mismatches:
            print(f"PASS — all {len(TEST_QUERIES)} responses byte-identical "
                  f"to baseline. Flag-off safety property holds.")
            sys.exit(0)
        else:
            print(f"FAIL — {len(mismatches)} mismatch(es) found:")
            for query, reason in mismatches:
                print(f"  - {reason}: {query!r}")
            print()
            print("Do NOT claim this integration done until these are resolved.")
            sys.exit(1)

    ap.print_help()


if __name__ == "__main__":
    main()