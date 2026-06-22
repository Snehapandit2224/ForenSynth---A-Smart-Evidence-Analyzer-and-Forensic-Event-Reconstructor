"""
ForenSynth – Timeline Agent
utils.py: shared utility functions.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional


# Supported timestamp formats, most-specific first
_TS_FORMATS: List[str] = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
]


def parse_epoch(timestamp: str) -> float:
    """
    Parse an ISO-8601-ish timestamp string to a POSIX epoch float.
    Returns 0.0 if parsing fails.
    """
    ts = timestamp.strip()
    if not ts:
        return 0.0
    # Strip trailing Z and normalise
    ts_clean = re.sub(r"Z$", "", ts).strip()
    for fmt in _TS_FORMATS:
        try:
            dt = datetime.strptime(ts_clean, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return 0.0


def epoch_to_iso(epoch: float) -> str:
    """Convert POSIX epoch to ISO-8601 UTC string."""
    if epoch <= 0:
        return ""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_alias(alias: str) -> str:
    """Lowercase and replace non-alphanumeric chars with underscore."""
    return re.sub(r"[^a-z0-9_]", "_", alias.strip().lower())


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def content_action_keywords(content: str) -> List[str]:
    """Extract action-like tokens from a content string."""
    tokens = re.findall(r"\b[a-z]+\b", content.lower())
    return tokens


def short_summary(content: str, max_len: int = 80) -> str:
    """Return a truncated summary of content for narrative use."""
    c = content.strip()
    if len(c) <= max_len:
        return c
    return c[:max_len].rstrip() + "…"
