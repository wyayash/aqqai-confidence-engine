"""
AQQAI — Logging (logger.py)
=============================
Centralised Loguru setup. Import this module in main.py and run.py.

Usage:
    from logger import log

    log.info("Server started")
    log.debug("Query received: {query}", query=query)
    log.warning("Model failed: {model}", model=model_id)
    log.error("Pipeline error: {err}", err=str(e))
"""

import os
import sys
from loguru import logger as log

# ── Log directory ─────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

# ── Remove default Loguru handler (we define our own below) ──
log.remove()

# ── Terminal handler ──────────────────────────────────────
# Colourised, human-readable, INFO and above
log.add(
    sys.stdout,
    level      = "INFO",
    colorize   = True,
    format     = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan> | "
        "{message}"
    ),
)

# ── File handler ──────────────────────────────────────────
# Full detail, DEBUG and above, rotates at 10MB, keeps 7 days
log.add(
    "logs/pipeline.log",
    level      = "DEBUG",
    colorize   = False,
    format     = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}",
    rotation   = "10 MB",
    retention  = "7 days",
    encoding   = "utf-8",
)

log.info("Logger initialised — writing to logs/pipeline.log")