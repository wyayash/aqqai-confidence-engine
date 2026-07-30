"""
AQQAI — Skill Selector Client
==============================
Pipeline-side integration for the Skill Selector / Context Injector
(aqua-skill-selector, owned by Yashveer).

SAFETY DESIGN — read before touching anything:
- SKILL_INJECTION defaults to "off". With the flag off, get_enriched_prompt()
  returns the raw query completely unchanged — behavior must be byte-identical
  to main with this module absent.
- The /enrich call has an 800ms timeout and a broad except clause. If the
  skill-selector service is slow, down, or returns anything unexpected, we
  log a warning and silently fall back to the raw query. The main pipeline
  must never break, slow down materially, or error out because of this
  integration.
- send_skill_feedback() is fire-and-forget: it must never raise into the
  caller and must never block the response being returned to the user.
"""

import os
import logging
import requests

log = logging.getLogger("main")

# ── Config ────────────────────────────────────────────────
SKILL_INJECTION = os.getenv("SKILL_INJECTION", "off") == "on"
SKILL_SELECTOR_URL = os.getenv("SKILL_SELECTOR_URL", "http://localhost:9000")

ENRICH_TIMEOUT_SECONDS = 0.8
FEEDBACK_TIMEOUT_SECONDS = 0.8


def get_enriched_prompt(query: str, task_type: str) -> str:
    """
    Call the Skill Selector's /enrich endpoint to inject the best-matching
    skill file as a system-level instruction ahead of the user's query.

    Call this immediately after analyze_task(), before the parallel
    fan-out to the 5 models.

    Returns the raw `query` unchanged if:
      - SKILL_INJECTION is off (default), or
      - the call fails, times out, or returns an unexpected shape.

    This function must never raise.
    """
    enriched, _skill_id = get_enriched_prompt_with_metadata(query, task_type)
    return enriched


def get_enriched_prompt_with_metadata(query: str, task_type: str) -> tuple[str, str | None]:
    """
    Same as get_enriched_prompt(), but also returns the skill_id that was
    used (or None if no skill was injected), so the caller can pass it to
    send_skill_feedback() later.

    NOTE: this assumes the /enrich response includes a "skill_id" field.
    This is NOT YET CONFIRMED with Yashveer — freeze the exact /enrich
    response contract with him before relying on this in a live test.
    If "skill_id" isn't present in the response, this returns None for it
    and behaves exactly like the plain get_enriched_prompt() otherwise.

    Returns (query, None) unchanged if SKILL_INJECTION is off or the call
    fails — never raises.
    """
    if not SKILL_INJECTION:
        return query, None

    try:
        resp = requests.post(
            f"{SKILL_SELECTOR_URL}/enrich",
            json={"task_type": task_type, "query": query},
            timeout=ENRICH_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        enriched = data.get("enriched_prompt")
        skill_id = data.get("skill_id")  # UNCONFIRMED field name — verify with Yashveer
        if not enriched or not isinstance(enriched, str):
            log.warning(
                "Skill injection returned an unexpected payload, "
                "falling back to raw query | payload=%r", data
            )
            return query, None
        return enriched, skill_id
    except Exception as e:
        log.warning(f"Skill injection failed, falling back to raw query: {e}")
        return query, None


def send_skill_feedback(skill_used: str, task_type: str, final_rcks_score: float) -> None:
    """
    Fire-and-forget POST to /skill-feedback so the Skill Selector can learn
    which skill files actually produce good scores per task type.

    Call this AFTER scoring is complete, alongside/after the Bayesian
    confidence update — never in the hot path before the response is
    returned to the user.

    Must never raise. Must never block. If SKILL_INJECTION is off, or
    skill_used is falsy (no skill was actually injected for this query),
    this is a no-op.
    """
    if not SKILL_INJECTION or not skill_used:
        return

    try:
        requests.post(
            f"{SKILL_SELECTOR_URL}/skill-feedback",
            json={
                "skill_used": skill_used,
                "task_type": task_type,
                "final_rcks_score": final_rcks_score,
            },
            timeout=FEEDBACK_TIMEOUT_SECONDS,
        )
    except Exception as e:
        # Deliberately swallow everything — feedback is best-effort telemetry,
        # never allowed to affect the request/response cycle.
        log.warning(f"Skill feedback POST failed (non-blocking, ignored): {e}")