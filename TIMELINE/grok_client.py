"""
ForenSynth – Timeline Agent
grok_client.py: Groq API client with batching, retries, and graceful fallback.

Uses the official `groq` Python library (same one the ER pipeline uses).
Set GROQ_API_KEY in your environment before running.

Design principles:
- NEVER hardcode secrets.
- Batch all requests; never one API call per observation.
- Graceful fallback when API key is absent or calls fail.
- Maximum 1–3 LLM calls per case.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from config import (
    GROK_API_KEY,
    GROK_BATCH_CHUNK_SIZE,
    GROK_MAX_RETRIES,
    GROK_MAX_TOKENS,
    GROK_MODEL,
    GROK_RETRY_DELAY_SEC,
)

log = logging.getLogger("forensynth.timeline.grok_client")

# ── Try importing the groq library ───────────────────────────────────────────
try:
    from groq import Groq as _GroqSDK  # type: ignore
    _GROQ_LIB_AVAILABLE = True
except ImportError:
    _GROQ_LIB_AVAILABLE = False
    log.warning("groq package not installed. Run: pip install groq")


def _strip_markdown_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _safe_json_load(text: str) -> Optional[Any]:
    cleaned = _strip_markdown_json(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


class GrokClient:
    """
    Groq API client wrapping the official `groq` Python SDK.
    Falls back silently to heuristics when the key is absent or calls fail.
    """

    def __init__(self) -> None:
        self._api_key = GROK_API_KEY
        self._client: Any = None
        self._available = False

        if not _GROQ_LIB_AVAILABLE:
            log.warning("groq library missing – LLM reasoning disabled. pip install groq")
            return
        if not self._api_key:
            log.warning("GROQ_API_KEY not set – LLM reasoning disabled; falling back to heuristics.")
            return

        try:
            self._client = _GroqSDK(api_key=self._api_key)
            self._available = True
            log.info("Groq client initialised (model: %s)", GROK_MODEL)
        except Exception as exc:
            log.warning("Groq client init failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def _chat(self, messages: List[Dict[str, str]], max_tokens: int = GROK_MAX_TOKENS) -> Optional[str]:
        """Single chat completion with retry logic. Returns response text or None."""
        if not self._available or self._client is None:
            return None

        for attempt in range(1, GROK_MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=GROK_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                return resp.choices[0].message.content
            except Exception as exc:
                log.warning("Groq attempt %d/%d failed: %s", attempt, GROK_MAX_RETRIES, exc)
                if attempt < GROK_MAX_RETRIES:
                    time.sleep(GROK_RETRY_DELAY_SEC * attempt)

        log.error("All Groq retries exhausted – falling back to heuristics.")
        self._available = False
        return None

    # ── Public batch methods ──────────────────────────────────────────────────

    def infer_temporal_relations_batch(
        self, event_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch-infer temporal relations for a list of event pairs.

        Each item in event_pairs:
          {"id": int, "a_content": str, "a_timestamp": str, "a_location": str,
           "b_content": str, "b_timestamp": str, "b_location": str}

        Returns list of:
          {"id": int, "relation": "BEFORE|AFTER|SIMULTANEOUS|UNKNOWN", "confidence": float}
        """
        if not self._available or not event_pairs:
            return []

        system = (
            "You are a forensic timeline analyst. "
            "Given pairs of events, determine their temporal relationship. "
            "Respond ONLY with a valid JSON array. No prose, no markdown. "
            'Each element: {"id": <int>, "relation": "BEFORE|AFTER|SIMULTANEOUS|UNKNOWN", '
            '"confidence": <float 0.0-1.0>}.'
        )

        results: List[Dict[str, Any]] = []
        chunks = [
            event_pairs[i: i + GROK_BATCH_CHUNK_SIZE]
            for i in range(0, len(event_pairs), GROK_BATCH_CHUNK_SIZE)
        ]

        for chunk in chunks:
            user = (
                "Determine the temporal relationship for each event pair. "
                "BEFORE means A happens before B.\n\n"
                + json.dumps(chunk, ensure_ascii=False)
            )
            raw = self._chat([
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ])
            if raw:
                parsed = _safe_json_load(raw)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "id" in item and "relation" in item:
                            results.append({
                                "id":         int(item["id"]),
                                "relation":   str(item.get("relation", "UNKNOWN")).upper(),
                                "confidence": float(item.get("confidence", 0.5)),
                            })
        return results

    def infer_causal_relations_batch(
        self, event_pairs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch-infer causal relationships for a list of event pairs.

        Each item:
          {"id": int, "a_content": str, "a_timestamp": str, "a_entity": str, "a_location": str,
           "b_content": str, "b_timestamp": str, "b_entity": str, "b_location": str}

        Returns list of:
          {"id": int, "causal": bool, "confidence": float, "explanation": str}
        """
        if not self._available or not event_pairs:
            return []

        system = (
            "You are a forensic causality analyst. "
            "Given pairs of events from a forensic case, determine if event A causally leads to event B. "
            "Consider action dependencies, shared entities, shared locations, and narrative logic. "
            "CRITICAL RULE: a witness or bystander OBSERVING or REPORTING something is never "
            "'caused by' another person's earlier action. Witness testimony corroborates an event; "
            "it is not a causal consequence of it. Only mark causal=true when A's action is a "
            "necessary precondition, trigger, or enabler of B's action, where A and B describe the "
            "SAME actor's sequence of actions, or B is a direct physical/systemic effect of A "
            "(e.g. a card swipe causing a network access log). "
            "If B is merely a different person's report or observation of the scene, set causal=false. "
            "Respond ONLY with a valid JSON array. No prose, no markdown. "
            'Each element: {"id": <int>, "causal": <bool>, "confidence": <float 0.0-1.0>, '
            '"explanation": "<one sentence>"}.'
        )

        results: List[Dict[str, Any]] = []
        chunks = [
            event_pairs[i: i + GROK_BATCH_CHUNK_SIZE]
            for i in range(0, len(event_pairs), GROK_BATCH_CHUNK_SIZE)
        ]

        for chunk in chunks:
            user = (
                "For each pair, determine if A causally leads to B.\n\n"
                + json.dumps(chunk, ensure_ascii=False)
            )
            raw = self._chat([
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ])
            if raw:
                parsed = _safe_json_load(raw)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "id" in item:
                            results.append({
                                "id":          int(item["id"]),
                                "causal":      bool(item.get("causal", False)),
                                "confidence":  float(item.get("confidence", 0.4)),
                                "explanation": str(item.get("explanation", "")),
                            })
        return results