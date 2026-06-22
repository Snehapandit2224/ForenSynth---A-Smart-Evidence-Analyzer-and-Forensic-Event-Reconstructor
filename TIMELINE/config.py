"""
ForenSynth – Timeline Agent
config.py: centralised constants and environment-driven settings.
"""
from __future__ import annotations

import os

# ── Groq / LLM ─────────────────────────────────────────────────────────────
GROK_API_KEY: str = os.environ.get("GROQ_API_KEY", "")   # reads GROQ_API_KEY from env
GROK_MODEL: str = "llama-3.1-8b-instant"                 # fast Groq-hosted model
GROK_MAX_TOKENS: int = 2048
GROK_TIMEOUT_SEC: int = 30
GROK_MAX_RETRIES: int = 3
GROK_RETRY_DELAY_SEC: float = 2.0
GROK_BATCH_CHUNK_SIZE: int = 20  # observations per LLM call

# ── Temporal reasoning ───────────────────────────────────────────────────────
CAUSAL_WINDOW_SEC: int = 600          # max gap to infer causality
SIMULTANEOUS_WINDOW_SEC: int = 30     # events within N sec are "simultaneous"
TEMPORAL_CONFIDENCE_DECAY: float = 0.05  # penalty per hour of gap uncertainty

# ── Confidence weights ───────────────────────────────────────────────────────
WEIGHT_OBS_CONFIDENCE: float = 0.40
WEIGHT_ENTITY_RESOLUTION: float = 0.30
WEIGHT_TEMPORAL_CERTAINTY: float = 0.20
WEIGHT_CONFLICT_PENALTY: float = 0.10

# ── Modality reliability (used for ordering tie-breaking) ────────────────────
MODALITY_RELIABILITY: dict[str, float] = {
    "video":   0.95,
    "audio":   0.85,
    "text":    0.75,
    "network": 0.80,
    "witness": 0.65,
    "unknown": 0.50,
}

# ── Action dependency rules (deterministic causal reasoning) ─────────────────
# Each entry: (prerequisite_keyword, dependent_keyword)
CAUSAL_ACTION_RULES: list[tuple[str, str]] = [
    ("enter",    "use"),
    ("enter",    "access"),
    ("enter",    "withdraw"),
    ("enter",    "deposit"),
    ("use",      "withdraw"),
    ("use",      "deposit"),
    ("withdraw", "exit"),
    ("deposit",  "exit"),
    ("access",   "exit"),
    ("arrive",   "enter"),
    ("arrive",   "access"),
    ("call",     "receive"),
    ("send",     "receive"),
    ("attempt",  "succeed"),
    ("attempt",  "fail"),
]

# ── Timeline versioning ───────────────────────────────────────────────────────
TIMELINE_SCHEMA_VERSION: str = "1.0.0"

# ── Output paths ─────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR: str = "."
GRAPH_EXPORT_FILENAME: str = "timeline_graph.json"