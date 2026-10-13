"""
ForenSynth - shared LLM extraction helpers for the text-based field agents
(fir_agent.py, text_agent.py).

Two things live here so both agents behave identically:

1. GroqJSON - one budgeted, rate-limit-aware Groq client that always asks
   for a JSON object back and never raises: any failure (no key, network,
   rate limit, unparseable reply) returns None and the caller falls back to
   its own non-LLM path.

2. verify_quote - the hallucination guard. Every value the LLM extracts
   must come with a verbatim `evidence` quote from the source text; this
   checks the quote really is in the text (exact after whitespace/case
   normalisation, else a high fuzzy partial match to tolerate the model
   fixing a typo or dropping a comma). Anything that fails is dropped or
   marked unverified by the caller - an LLM is allowed to *read* the
   evidence for us, never to *add* to it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from rapidfuzz import fuzz

try:
    from groq import Groq as _GroqClient  # type: ignore
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClient = None  # type: ignore
    _GROQ_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

log = logging.getLogger("forensynth.llm_extract")

# Same model the ER pipeline's ContextScoringAgent uses (agents/entity_resolution.py).
GROQ_MODEL = "openai/gpt-oss-20b"

# Fuzzy partial-match threshold for an evidence quote that isn't an exact
# substring. 90 tolerates small LLM edits (punctuation, a fixed typo) but
# rejects a paraphrase.
QUOTE_MATCH_THRESHOLD = 90

# Groq free-tier tokens-per-minute for gpt-oss-20b. Override with GROQ_TPM_LIMIT on a paid tier.
TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", "8000"))


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def verify_quote(quote: str | None, source: str) -> bool:
    """True if `quote` actually appears in `source` (see module docstring).
    A quote stitched from several passages with "..." passes only if every
    part does."""
    if not quote or not str(quote).strip():
        return False
    s = _norm(source)
    parts = [p.strip(" .,;:") for p in re.split(r"\.{3,}|…", _norm(str(quote)))]
    parts = [p for p in parts if p]
    if not parts or all(len(p) < 3 for p in parts):
        return False
    return all(p in s or fuzz.partial_ratio(p, s) >= QUOTE_MATCH_THRESHOLD
               for p in parts if len(p) >= 3)


_STOPWORDS = {
    "the", "and", "with", "was", "were", "who", "that", "this", "from", "into", "onto", "about",
    "wearing", "carrying", "person", "man", "men", "woman", "unknown", "described", "approx",
    "approximately", "years", "year", "aged", "having", "their", "they", "his", "her", "one", "two",
    # role vocabulary the model legitimately uses to describe people
    "informant", "complainant", "victim", "witness", "suspect", "accused", "relative", "of",
}


def ungrounded_words(fragment: str | None, source: str) -> list[str]:
    """Content words in an LLM-written fragment (e.g. a person's
    description) that never appear in the source - the cheap check for
    invented detail ('tall', 'bearded') in text that isn't a verbatim quote."""
    src = set(re.findall(r"[a-z0-9]+", _norm(source)))
    words = re.findall(r"[a-z0-9]+", _norm(fragment or ""))

    def present(w: str) -> bool:
        if w in src or w.rstrip("s") in src or f"{w}s" in src:
            return True
        # spelling variants (neighbor/neighbour, grey/gray)
        return any(fuzz.ratio(w, x) >= 85 for x in src if x[:1] == w[:1] and abs(len(x) - len(w)) <= 2)

    return [w for w in words if len(w) > 2 and w not in _STOPWORDS and not present(w)]


def parse_json_object(text: str) -> dict | None:
    """Tolerant JSON-object parser - strips code fences and falls back to the
    outermost {...} span if the model wrapped its answer in prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def chunk_text(text: str, max_chars: int = 5000, overlap: int = 300) -> list[str]:
    """Split long documents on paragraph boundaries so each LLM call sees a
    coherent passage. Short documents (the common case) come back whole."""
    if len(text) <= max_chars:
        return [text]
    paras = re.split(r"\n\s*\n", text)
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p   # carry a little context across the cut
        else:
            cur = f"{cur}\n\n{p}" if cur else p
        while len(cur) > max_chars:             # a single giant paragraph
            chunks.append(cur[:max_chars])
            cur = cur[max_chars - overlap:]
    if cur.strip():
        chunks.append(cur)
    return chunks


class GroqJSON:
    """Budgeted Groq JSON-mode caller. `available` is False when there is no
    key / no groq package / --no-llm, and callers must handle that."""

    def __init__(self, model: str = GROQ_MODEL, enabled: bool = True,
                 max_calls: int = 20, min_interval_sec: float = 3.0):
        self.model = model
        self.max_calls = max_calls
        self.min_interval_sec = min_interval_sec
        self.calls_made = 0
        self._last_call = 0.0
        self._client = None
        if enabled and _GROQ_AVAILABLE:
            key = os.environ.get("GROQ_API_KEY", "")
            if key:
                self._client = _GroqClient(api_key=key)
            else:
                log.warning("GROQ_API_KEY not set - LLM extraction disabled, using fallback.")

    @property
    def available(self) -> bool:
        return self._client is not None

    def ask(self, system: str, user: str, max_tokens: int = 4096) -> dict | None:
        if not self.available:
            return None
        if self.calls_made >= self.max_calls:
            log.warning("LLM call budget (%d) exhausted - using fallback.", self.max_calls)
            return None
        # Groq counts prompt + max_tokens against the per-minute token limit
        # (8000 on the free tier), so a large max_tokens gets a request
        # rejected outright (HTTP 413) even when the real answer is short.
        prompt_est = int((len(system) + len(user)) / 3.2)
        max_tokens = max(1024, min(max_tokens, TPM_LIMIT - prompt_est - 200))
        for attempt in range(3):
            wait = self.min_interval_sec - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self.calls_made += 1
            self._last_call = time.time()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=max_tokens,
                    # Retries think less: long reasoning is what eats the token
                    # budget and truncates the JSON in the first place.
                    reasoning_effort="medium" if attempt == 0 else "low",
                )
                parsed = parse_json_object(resp.choices[0].message.content or "")
                if parsed is not None:
                    return parsed
                log.warning("LLM returned unparseable JSON (attempt %d).", attempt + 1)
            except Exception as exc:  # rate limit, network, invalid JSON-mode output...
                msg = str(exc)
                # Groq rejects JSON-mode output it can't validate but returns it
                # as failed_generation - often salvageable (e.g. trailing prose).
                body = getattr(exc, "body", None)
                failed = (body.get("error", body) if isinstance(body, dict) else {}).get("failed_generation")
                if failed:
                    parsed = parse_json_object(failed)
                    if parsed is not None:
                        log.info("Recovered JSON from failed_generation (attempt %d).", attempt + 1)
                        return parsed
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, msg[:200])
                m = re.search(r"Limit (\d+), Requested (\d+)", msg)
                if m:   # request too large for the TPM window - shrink the answer budget
                    max_tokens = max(1024, max_tokens - (int(m.group(2)) - int(m.group(1))) - 300)
                time.sleep(15 if "rate" in msg.lower() else 2)
            if self.calls_made >= self.max_calls:
                break
        return None


_HONORIFIC = re.compile(r"^(?:smt|sri|shri|shrimati|kum|kumari|mr|mrs|ms|miss|dr|master)\.?\s+", re.IGNORECASE)


def strip_honorific(name: str | None) -> str | None:
    """'Smt. Kavitha Raghunath' -> 'Kavitha Raghunath' - ER's alias
    normalisation is literal, so titles would otherwise split one person
    into two entities across documents."""
    if not name:
        return name
    prev, name = None, str(name).strip()
    while prev != name:
        prev, name = name, _HONORIFIC.sub("", name).strip()
    return name or prev


def clamp_conf(value, default: float = 0.5) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 3)
    except (TypeError, ValueError):
        return default
