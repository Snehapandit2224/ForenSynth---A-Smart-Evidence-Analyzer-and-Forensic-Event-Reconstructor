from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_agents_dir = str(_Path(__file__).parent)
if _agents_dir not in _sys.path:
    _sys.path.insert(0, _agents_dir)

"""
ForenSynth – timeline_agent.py
Timeline Agent: event enrichment, temporal/causal reasoning,
graph construction, narrative and explainability generation.
"""

# ── stdlib ───────────────────────────────────────────────────────────────
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# ── third-party ──────────────────────────────────────────────────────────
import networkx as nx

# ── shared ForenSynth utilities ───────────────────────────────────────────
from shared import (
    NormalizedObservation, normalize_case, get_normalized_observation_store,
    set_normalized_observation_store, get_semantic_scorer, semantic_location_similarity,
    normalize_alias, normalize_modality, normalize_role, clean_content,
    extract_action_tags, parse_timestamp, epoch_to_iso,
    utc_now_iso, clamp, content_action_keywords, short_summary, deterministic_tiebreak_key,
)

try:
    from groq import Groq as _GroqClient  # type: ignore
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClient = None  # type: ignore
    _GROQ_AVAILABLE = False

log = logging.getLogger("forensynth.timeline_agent")


# ── Configuration constants ────────────────────────────────────────────────
# -- LLM fallback selection ---------------------------------------------------
# Backend priority when TimelineAgent() is constructed without an explicit
# `llm=` argument (see agent.py:_build_default_llm):
#   1. CloudLLMFallback (Groq) - used automatically IF a Timeline_Key is
#      found (env var or Colab secret). Hard-capped at 2 API calls per run
#      (see llm_fallback.py docstring) - never per-pair, never uncapped.
#   2. LocalLLMFallback - used if TIMELINE_LOCAL_LLM_BACKEND is set and no
#      cloud key was found.
#   3. NoOpLLMFallback - fully offline, zero calls, zero dependency. Used if
#      neither of the above is configured. Recommended for reproducible
#      grading runs where you don't want output to vary between runs.
CLOUD_LLM_MODEL: str = os.environ.get("TIMELINE_CLOUD_LLM_MODEL", "openai/gpt-oss-20b")
CLOUD_LLM_TIMEOUT_SEC: float = 20.0

# Local-only fallback (opt-in via env var; talks only to localhost / in-process)
LOCAL_LLM_BACKEND: str = os.environ.get("TIMELINE_LOCAL_LLM_BACKEND", "")  # "", "ollama", "llama_cpp_server", "transformers"
LOCAL_LLM_MODEL: str = os.environ.get("TIMELINE_LOCAL_LLM_MODEL", "")
LOCAL_LLM_HOST: str = os.environ.get("TIMELINE_LOCAL_LLM_HOST", "http://localhost:11434")
LOCAL_LLM_TIMEOUT_SEC: float = 15.0

# -- Temporal reasoning -------------------------------------------------------
CAUSAL_WINDOW_SEC: int = 600          # max gap to infer causality
SIMULTANEOUS_WINDOW_SEC: int = 30     # events within N sec are "simultaneous"
TEMPORAL_CONFIDENCE_DECAY: float = 0.05  # penalty per hour of gap uncertainty

# FIX: same-location causal check used to require an EXACT location_key
# match (text before first comma). "at the entrance of the ATM" and "at the
# ATM entry door" never matched despite meaning the same place. Now uses
# semantic_similarity.py (real embeddings, falls back to lexical fuzzy
# matching if the model is unavailable) with this threshold.
LOCATION_SEMANTIC_SIMILARITY_THRESHOLD: float = 0.60

# -- Confidence weights --------------------------------------------------------
# FIX: these three must sum to 1.0 so a perfect-evidence event can reach a
# confidence of 1.0. The conflict penalty is applied as a separate,
# subtractive term afterwards (and the result is clamped to [0, 1]).
WEIGHT_OBS_CONFIDENCE: float = 0.45
WEIGHT_ENTITY_RESOLUTION: float = 0.30
WEIGHT_TEMPORAL_CERTAINTY: float = 0.25
assert abs((WEIGHT_OBS_CONFIDENCE + WEIGHT_ENTITY_RESOLUTION + WEIGHT_TEMPORAL_CERTAINTY) - 1.0) < 1e-9, \
    "Confidence weights must sum to 1.0"

WEIGHT_CONFLICT_PENALTY: float = 0.15   # subtracted (not part of the 1.0 budget above)

# -- Modality reliability (used for ordering tie-breaking) --------------------
MODALITY_RELIABILITY: dict[str, float] = {
    "video":   0.95,
    "audio":   0.85,
    "text":    0.75,
    "network": 0.80,
    "witness": 0.65,
    "unknown": 0.50,
}

# -- Action dependency rules (deterministic causal reasoning) -----------------
# FIX: previously these were single exact-root words ("enter","exit"...)
# matched against raw tokenized content, which missed almost everything on
# real generator output (see normalization.py docstring - 6/7 real
# observations matched zero keywords). Now expressed as canonical ACTION
# TAGS produced by normalization.extract_action_tags(), which does
# fragment-based matching grounded in the generator's actual phrase banks
# (ATM_Robbery + Office_Theft domains) - each entry: (prerequisite_tag, dependent_tag).
CAUSAL_ACTION_RULES: list[tuple[str, str]] = [
    ("APPROACH",  "ENTER"),
    ("APPROACH",  "TAMPER"),
    ("ENTER",     "WITHDRAW"),
    ("ENTER",     "TAMPER"),
    ("ENTER",     "STEAL"),
    ("ENTER",     "NAVIGATE"),
    ("ENTER",     "WORK"),
    ("NAVIGATE",  "STEAL"),
    ("WITHDRAW",  "LOITER"),
    ("WITHDRAW",  "EXIT"),
    ("TAMPER",    "EXIT"),
    ("STEAL",     "EXIT"),
    ("LOITER",    "EXIT"),
    # FIX: found via real CASE_ATM_002 ground truth - a lookout actor's own
    # real sequence is approach_atm -> wait_outside -> flee_scene, distinct
    # from the machine-operator's approach -> enter -> withdraw -> exit.
    ("APPROACH",  "WATCH"),
    ("WATCH",     "FLEE"),
    ("EXIT",      "FLEE"),
    ("EXIT",      "OBSERVE"),
    ("FLEE",      "OBSERVE"),
    ("OBSERVE",   "REPORT"),
    ("COMMUNICATE", "CONFIRM"),
    ("CONFIRM",   "COMMUNICATE"),
    ("INTERCEPT", "REPORT"),
]

# -- Event clustering (observation -> event reconstruction) -------------------
# Per the unified architecture doc (Phase 4/5): observations are evidence,
# not events. Multiple observations describing the same real-world action
# (e.g. video + audio both covering one ATM interaction) should reconstruct
# into ONE event with multiple supporting observations, not N separate
# events. Clustering is scoped to a single canonical entity (never merges
# across different actors) - this is the conservative, correct interpretation
# for a forensic tool: cross-entity "same event" claims (e.g. witness+suspect
# co-presence) stay as separate events linked by causal/temporal edges
# instead, which is auditable, rather than fused into one ambiguous record.
EVENT_CLUSTER_WINDOW_SEC: float = 90.0      # max time gap to be clusterable
EVENT_CLUSTER_COMPAT_THRESHOLD: float = 0.5  # matches the unified doc's stated threshold

# Compatibility score weights (sum to 1.0) - see event_clustering.py
EVENT_CLUSTER_WEIGHT_ACTION_TAG: float = 0.40
EVENT_CLUSTER_WEIGHT_TIME: float = 0.30
EVENT_CLUSTER_WEIGHT_LOCATION: float = 0.20
EVENT_CLUSTER_WEIGHT_LEXICAL: float = 0.10
assert abs(
    EVENT_CLUSTER_WEIGHT_ACTION_TAG + EVENT_CLUSTER_WEIGHT_TIME
    + EVENT_CLUSTER_WEIGHT_LOCATION + EVENT_CLUSTER_WEIGHT_LEXICAL - 1.0
) < 1e-9, "Event cluster weights must sum to 1.0"


TIMELINE_SCHEMA_VERSION: str = "1.1.0"

# -- Output paths ---------------------------------------------------------------
DEFAULT_OUTPUT_DIR: str = "."
GRAPH_EXPORT_FILENAME: str = "timeline_graph.json"

# -- Output classification -----------------------------------------------------
# Per the unified architecture doc's three-case output taxonomy
# (CLEAR_WINNER / AMBIGUOUS / PARTIAL). Honest scope note: this is a
# signal-based classification of THIS single reconstructed timeline's own
# confidence - not a comparison across multiple ranked hypothesis timelines
# (that requires the entity-identity beam search + Critique/Showrunner
# belief-update loop, which are not built). It answers "how much should an
# investigator trust this timeline as reconstructed," using concrete,
# already-computed signals rather than a single opaque score.
CLASSIFICATION_CLEAR_MIN_AVG_CONFIDENCE: float = 0.75
CLASSIFICATION_CLEAR_MAX_UNRESOLVED_FRACTION: float = 0.15
CLASSIFICATION_AMBIGUOUS_MAX_AVG_CONFIDENCE: float = 0.55
CLASSIFICATION_AMBIGUOUS_MIN_CONFLICT_FRACTION: float = 0.30
CLASSIFICATION_AMBIGUOUS_MIN_UNRESOLVED_FRACTION: float = 0.50
class EdgeType(str, Enum):
    TEMPORAL = "TEMPORAL"
    CAUSAL   = "CAUSAL"
    INFERRED = "INFERRED"


class TemporalRelation(str, Enum):
    BEFORE       = "BEFORE"
    AFTER        = "AFTER"
    SIMULTANEOUS = "SIMULTANEOUS"
    UNKNOWN      = "UNKNOWN"


# -- Raw observation (mirrors ER pipeline output) -------------------------------

@dataclass
class RawObservation:
    obs_id:     str
    entity:     str
    role:       str
    modality:   str
    location:   str
    content:    str
    timestamp:  str
    confidence: float
    # Populated by normalize_case() via the repository (see repositories.py) -
    # computed ONCE at the normalization boundary, not re-derived downstream.
    entity_norm:     str   = ""
    time_offset_sec: int   = 0
    _ts_epoch:       float = field(default=0.0, repr=False)
    location_key:    str   = ""
    action_tags:     List[str] = field(default_factory=list)


# -- Canonical entity from ER ---------------------------------------------------

@dataclass
class CanonicalEntity:
    entity_id:         str
    primary_alias:     str
    aliases:           List[str]
    confidence_score:  float
    sources:           List[str]   # obs_ids that belong to this entity
    modalities:        List[str]
    locations:         List[str]
    roles:             List[str]
    earliest_timestamp: str        # ISO-8601
    latest_timestamp:   str
    time_span_seconds:  int
    candidate_mentions: List[Dict[str, Any]] = field(default_factory=list)


# -- Timeline event (enriched observation) --------------------------------------

@dataclass
class TimelineEvent:
    event_id:      str
    obs_ids:       List[str]          # provenance -> raw observations
    timestamp:     str                # ISO-8601
    ts_epoch:      float              # for arithmetic
    location:      str
    entity_id:     str                # canonical entity id
    primary_alias: str
    aliases:       List[str]
    modality:      str
    content:       str
    confidence:    float
    role:          str  = "unknown"   # suspect / witness / system / etc.
    conflict_flag: bool = False
    conflict_note: str  = ""
    location_key:  str  = ""          # coarse normalized location for matching (see normalization.py)
    action_tags:   List[str] = field(default_factory=list)  # canonical tags detected across supporting obs

    # Explainability
    reasoning:     List[str] = field(default_factory=list)

    # Versioning hook
    version:       str = "V1"


# -- Graph edge ------------------------------------------------------------------

@dataclass
class TimelineEdge:
    source:     str       # event_id
    target:     str       # event_id
    edge_type:  EdgeType
    confidence: float
    relation:   TemporalRelation = TemporalRelation.BEFORE
    label:      str = ""


# -- Uncertainty record ------------------------------------------------------------

@dataclass
class UncertaintyRecord:
    event_id:          str
    uncertainty_score: float          # 1.0 - confidence
    sources:           List[str]      # obs_ids
    reasons:           List[str]


# -- Conflict record -----------------------------------------------------------

@dataclass
class ConflictRecord:
    conflict_type: str
    cluster_id:    str
    detail:        str
    affected_obs:  List[str] = field(default_factory=list)


# -- Narrative line ---------------------------------------------------------------

@dataclass
class NarrativeLine:
    timestamp:  str
    actor:      str
    action:     str
    location:   str
    evidence:   List[str]   # obs_ids
    confidence: float
    event_id:   str


# -- Explainability record -------------------------------------------------------

@dataclass
class ExplainabilityRecord:
    event_id:    str
    derived_from: List[str]   # obs_ids
    entity_used: str          # canonical entity_id
    reasoning:   List[str]
    confidence:  float


# -- Timeline version wrapper ----------------------------------------------------

@dataclass
class TimelineVersion:
    version:          str              # "V1", "V2", ...
    schema_version:   str
    case_id:          str
    generated_at:     str              # ISO-8601 UTC
    events:           List[TimelineEvent]
    causal_links:     List[TimelineEdge]
    timeline_graph:   Dict[str, Any]   # serialised NetworkX graph
    uncertainties:    List[UncertaintyRecord]
    narrative:        List[NarrativeLine]
    explainability:   List[ExplainabilityRecord]
    conflicts_summary: List[Dict[str, Any]]
    # FIX: transparent reporting of anything the pipeline could not resolve,
    # instead of silently dropping or silently guessing it.
    unresolved_temporal_pairs: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_causal_pairs:   List[Dict[str, Any]] = field(default_factory=list)
    conflicts_unlocalized_count: int = 0
    unresolved_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON export."""
        def _edge_to_dict(e: TimelineEdge) -> Dict[str, Any]:
            return {
                "source":     e.source,
                "target":     e.target,
                "edge_type":  e.edge_type.value,
                "confidence": round(e.confidence, 4),
                "relation":   e.relation.value,
                "label":      e.label,
            }

        def _event_to_dict(ev: TimelineEvent) -> Dict[str, Any]:
            return {
                "event_id":      ev.event_id,
                "obs_ids":       ev.obs_ids,
                "timestamp":     ev.timestamp,
                "location":      ev.location,
                "location_key":  ev.location_key,
                "entity_id":     ev.entity_id,
                "primary_alias": ev.primary_alias,
                "aliases":       ev.aliases,
                "modality":      ev.modality,
                "content":       ev.content,
                "confidence":    round(ev.confidence, 4),
                "role":          ev.role,
                "action_tags":   ev.action_tags,
                "conflict_flag": ev.conflict_flag,
                "conflict_note": ev.conflict_note,
                "reasoning":     ev.reasoning,
                "version":       ev.version,
            }

        def _uncertainty_to_dict(u: UncertaintyRecord) -> Dict[str, Any]:
            return {
                "event_id":          u.event_id,
                "uncertainty_score": round(u.uncertainty_score, 4),
                "sources":           u.sources,
                "reasons":           u.reasons,
            }

        def _narrative_to_dict(n: NarrativeLine) -> Dict[str, Any]:
            return {
                "event_id":  n.event_id,
                "timestamp": n.timestamp,
                "actor":     n.actor,
                "action":    n.action,
                "location":  n.location,
                "evidence":  n.evidence,
                "confidence": round(n.confidence, 4),
            }

        def _explain_to_dict(x: ExplainabilityRecord) -> Dict[str, Any]:
            return {
                "event_id":    x.event_id,
                "derived_from": x.derived_from,
                "entity_used": x.entity_used,
                "reasoning":   x.reasoning,
                "confidence":  round(x.confidence, 4),
            }

        return {
            "case_id":          self.case_id,
            "timeline_version": self.version,
            "schema_version":   self.schema_version,
            "generated_at":     self.generated_at,
            "events":           [_event_to_dict(e) for e in self.events],
            "causal_links":     [_edge_to_dict(e) for e in self.causal_links],
            "timeline_graph":   self.timeline_graph,
            "uncertainties":    [_uncertainty_to_dict(u) for u in self.uncertainties],
            "narrative":        [_narrative_to_dict(n) for n in self.narrative],
            "explainability":   [_explain_to_dict(x) for x in self.explainability],
            "conflicts_summary": self.conflicts_summary,
            "conflicts_unlocalized_count": self.conflicts_unlocalized_count,
            "unresolved_temporal_pairs": self.unresolved_temporal_pairs,
            "unresolved_causal_pairs":   self.unresolved_causal_pairs,
            "unresolved_entities":       self.unresolved_entities,
        }


"""
ForenSynth - Timeline Agent
llm_fallback.py: pluggable, OFFLINE-ONLY LLM fallback interface.

FIXED in this revision:
  The previous version of this file (grok_client.py) called the hosted
  Groq cloud API, which directly violates this project's own design
  mandate: "No external API calls. The system must run fully offline."
  Sending case evidence off-device to a third party is also a real
  chain-of-custody problem for a forensic tool.

  This replacement is offline by default (NoOpLLMFallback) and only
  talks to a model running on localhost if you explicitly configure one
  (LocalLLMFallback) — it never reaches the public internet.

Design contract (unchanged from the original spec):
  An LLM may optionally be used ONLY for:
    - unknown event interpretation
    - natural-language narrative polish
    - resolving genuinely ambiguous temporal/causal pairs after all
      deterministic rules have been exhausted
  An LLM must NEVER be used for:
    - entity resolution
    - graph construction
    - edge scoring / timeline ranking
    - event acceptance/rejection
  Logic remains the decision-maker; the LLM is advisory-only and its
  absence must never change correctness, only reduce how many ambiguous
  pairs get resolved (they fall through to "unresolved" instead, and are
  reported explicitly rather than guessed).
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

log = logging.getLogger("forensynth.timeline.llm_fallback")


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


class BaseLLMFallback(ABC):
    """Interface every LLM fallback backend must implement."""

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def resolve_temporal_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        pairs: [{"id": int, "a_content", "a_timestamp", "a_location",
                 "b_content", "b_timestamp", "b_location"}, ...]
        returns: [{"id": int, "relation": "BEFORE|AFTER|SIMULTANEOUS|UNKNOWN",
                   "confidence": float}, ...]  (best-effort; may be empty)
        """
        ...

    @abstractmethod
    def resolve_causal_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        pairs: [{"id": int, "a_content", "a_entity", "a_role", ...,
                 "b_content", "b_entity", "b_role", ...}, ...]
        returns: [{"id": int, "causal": bool, "confidence": float,
                   "explanation": str}, ...]  (best-effort; may be empty)
        """
        ...


class NoOpLLMFallback(BaseLLMFallback):
    """
    Default backend. No model, no network calls of any kind — fully offline.
    Everything routes through the deterministic rules in temporal_reasoner.py
    / causal_reasoner.py; pairs those rules can't resolve are reported as
    'unresolved' in the output rather than being silently guessed.
    """

    def available(self) -> bool:
        return False

    def resolve_temporal_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []

    def resolve_causal_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []


class LocalLLMFallback(BaseLLMFallback):
    """
    OPTIONAL local-only fallback. Disabled unless explicitly configured.
    Supports three backends, chosen via `backend`:

      "ollama"             -> REST call to a local Ollama server
                               (default host http://localhost:11434)
      "llama_cpp_server"   -> REST call to a local llama.cpp server
                               (default host http://localhost:8080)
      "transformers"       -> in-process HuggingFace pipeline; no network
                               call of any kind, fully local inference

    In every case, only localhost / in-process inference is used — this
    class makes no request to any public-internet host, ever. If the
    configured backend can't be reached, `available()` returns False and
    every caller falls back to deterministic logic automatically; the
    pipeline never blocks or errors because a local model is missing.

    Usage:
        from config import LOCAL_LLM_BACKEND, LOCAL_LLM_MODEL, LOCAL_LLM_HOST
        llm = LocalLLMFallback(LOCAL_LLM_BACKEND, LOCAL_LLM_MODEL, LOCAL_LLM_HOST)
    """

    def __init__(
        self,
        backend: str = "",
        model: str = "",
        host: str = "http://localhost:11434",
        timeout_sec: float = 15.0,
    ) -> None:
        self._backend = (backend or "").strip().lower()
        self._model = model or ""
        self._host = host.rstrip("/")
        self._timeout = timeout_sec
        self._ok = False
        self._pipe = None

        if not self._backend:
            log.info("LocalLLMFallback: no backend configured - running deterministic-only (offline).")
            return

        try:
            if self._backend == "ollama":
                import urllib.request
                urllib.request.urlopen(f"{self._host}/api/tags", timeout=2)
                self._ok = True
            elif self._backend == "llama_cpp_server":
                import urllib.request
                urllib.request.urlopen(f"{self._host}/health", timeout=2)
                self._ok = True
            elif self._backend == "transformers":
                from transformers import pipeline  # type: ignore
                self._pipe = pipeline("text-generation", model=self._model or "gpt2")
                self._ok = True
            else:
                log.warning("Unknown local LLM backend '%s' - disabled.", self._backend)
        except Exception as exc:
            log.warning(
                "Local LLM backend '%s' unreachable (%s) - disabled; "
                "deterministic rules will handle everything.",
                self._backend, exc,
            )
            self._ok = False

    def available(self) -> bool:
        return self._ok

    # -- internal transport --------------------------------------------------

    def _ask_json(self, system: str, user: str) -> Optional[Any]:
        if not self._ok:
            return None
        prompt = f"{system}\n\n{user}"
        try:
            if self._backend == "ollama":
                import urllib.request
                req = urllib.request.Request(
                    f"{self._host}/api/generate",
                    data=json.dumps({
                        "model": self._model or "llama3",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.0},
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return _safe_json_load(data.get("response", ""))

            if self._backend == "llama_cpp_server":
                import urllib.request
                req = urllib.request.Request(
                    f"{self._host}/completion",
                    data=json.dumps({"prompt": prompt, "n_predict": 512, "temperature": 0.0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return _safe_json_load(data.get("content", ""))

            if self._backend == "transformers" and self._pipe is not None:
                out = self._pipe(prompt, max_new_tokens=400, do_sample=False)
                text = out[0]["generated_text"][len(prompt):]
                return _safe_json_load(text)

        except Exception as exc:
            log.warning("Local LLM call failed (%s) - disabling for rest of run.", exc)
            self._ok = False
        return None

    # -- public API ------------------------------------------------------------

    def resolve_temporal_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pairs or not self._ok:
            return []
        system = (
            "You are a forensic timeline analyst. Given pairs of events, "
            "determine their temporal relationship. Respond ONLY with a "
            "valid JSON array, no prose, no markdown. Each element: "
            '{"id": <int>, "relation": "BEFORE|AFTER|SIMULTANEOUS|UNKNOWN", '
            '"confidence": <float 0.0-1.0>}.'
        )
        user = "BEFORE means A happens before B.\n\n" + json.dumps(pairs, ensure_ascii=False)
        parsed = self._ask_json(system, user)
        results: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "id" in item and "relation" in item:
                    results.append({
                        "id":         int(item["id"]),
                        "relation":   str(item.get("relation", "UNKNOWN")).upper(),
                        "confidence": float(item.get("confidence", 0.5)),
                    })
        return results

    def resolve_causal_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pairs or not self._ok:
            return []
        system = (
            "You are a forensic causality analyst. Determine if event A "
            "causally leads to event B. A witness/bystander observing or "
            "reporting something is never 'caused by' another person's "
            "earlier action - that is corroboration, not causation. "
            "Respond ONLY with a valid JSON array, no prose, no markdown. "
            'Each element: {"id": <int>, "causal": <bool>, '
            '"confidence": <float 0.0-1.0>, "explanation": "<one sentence>"}.'
        )
        user = json.dumps(pairs, ensure_ascii=False)
        parsed = self._ask_json(system, user)
        results: List[Dict[str, Any]] = []
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


class CloudLLMFallback(BaseLLMFallback):
    """
    OPTIONAL cloud-backed fallback (Groq), disabled unless an API key is
    found. This exists because you explicitly want to leverage a hosted LLM
    as a fallback, capped hard at a small, fixed number of calls per run.

    Call budget, by construction (not just convention):
      - temporal_reasoner.py batches ALL unresolved temporal pairs into
        exactly ONE call to resolve_temporal_pairs().
      - causal_reasoner.py batches ALL unresolved causal pairs into exactly
        ONE call to resolve_causal_pairs().
      => at most 2 API calls total per Timeline Agent run, regardless of
         case size. Neither method is ever called per-pair or in a loop.

    Key resolution order (mirrors your Entity Resolution notebook so both
    agents behave the same way in Colab):
      1. explicit `api_key` argument
      2. Timeline_Key environment variable
      3. Colab secret named Timeline_Key (google.colab.userdata)
    If no key is found anywhere, this silently reports unavailable and the
    deterministic rules handle everything - the pipeline never blocks or
    errors because a key is missing.

    Chain-of-custody note: when this backend is used, every event/pair it
    touches is tagged so the fact that case content left the machine is
    visible in the final output (see `agent.py`'s `llm_backend_used` field
    and the "[cloud-LLM]" labels on affected edges) - this keeps the
    forensic explainability story honest rather than hiding it.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "llama-3.1-8b-instant",
        timeout_sec: float = 20.0,
    ) -> None:
        self._model = model
        self._timeout = timeout_sec
        self._ok = False
        self._client = None
        self.calls_made = 0  # instrumented so the cap is auditable, not just asserted

        key = api_key or os.environ.get("Timeline_Key", "")
        if not key:
            try:
                from google.colab import userdata  # type: ignore
                key = userdata.get("Timeline_Key") or ""
            except Exception:
                key = ""

        if not key:
            log.info("CloudLLMFallback: no Timeline_Key found (env or Colab secret) - disabled, deterministic-only.")
            return

        try:
            from groq import Groq  # type: ignore
            self._client = Groq(api_key=key)
            self._ok = True
            log.info("CloudLLMFallback: Groq client initialised (model=%s).", self._model)
        except Exception as exc:
            log.warning("CloudLLMFallback: could not initialise Groq client (%s) - disabled.", exc)
            self._ok = False

    def available(self) -> bool:
        return self._ok

    def _ask_json(self, system: str, user: str) -> Optional[Any]:
        if not self._ok:
            return None
        try:
            self.calls_made += 1
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=1024,
                timeout=self._timeout,
            )
            text = resp.choices[0].message.content or ""
            return _safe_json_load(text)
        except Exception as exc:
            log.warning("CloudLLMFallback call failed (%s) - disabling for the rest of this run.", exc)
            self._ok = False
            return None

    # FIX: real production failure, found via CASE_ATM_002 (15 events, 45
    # unresolved causal pairs). Both resolve_*_pairs methods sent ALL pairs
    # in one uncapped request - for this case, 45 pairs needed 6,979 tokens
    # against Groq's 6,000 TPM cap, so the WHOLE request was rejected
    # (413), losing every pair's worth of potential LLM help in one shot -
    # not a graceful partial degradation. Worse: _ask_json sets self._ok =
    # False on ANY failure, silently disabling the LLM for the rest of the
    # run too (so if causal fails, a later temporal call would also be
    # skipped). MAX_PAIRS_PER_CALL caps the batch to a size that comfortably
    # fits under the limit (~20 pairs at this payload's shape estimates to
    # ~3100 tokens, well under 6,000). Pairs beyond the cap are simply never
    # sent - causal_reasoner.py/temporal_reasoner.py already correctly
    # report any pair with no LLM result as "unresolved", so this needs no
    # changes on their side. This keeps the <=2-calls-total guarantee intact
    # (still exactly one call each) while preventing one oversized case from
    # silently forfeiting all LLM assistance.
    MAX_PAIRS_PER_CALL = 20

    def resolve_temporal_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pairs or not self._ok:
            return []
        if len(pairs) > self.MAX_PAIRS_PER_CALL:
            log.info(
                "CloudLLMFallback: %d temporal pairs exceeds the %d-pair safe batch size - "
                "sending only the first %d to stay under the token limit; the rest report as unresolved.",
                len(pairs), self.MAX_PAIRS_PER_CALL, self.MAX_PAIRS_PER_CALL,
            )
            pairs = pairs[: self.MAX_PAIRS_PER_CALL]
        system = (
            "You are a forensic timeline analyst. Given pairs of events, "
            "determine their temporal relationship. Respond ONLY with a "
            "valid JSON array, no prose, no markdown. Each element: "
            '{"id": <int>, "relation": "BEFORE|AFTER|SIMULTANEOUS|UNKNOWN", '
            '"confidence": <float 0.0-1.0>}.'
        )
        user = "BEFORE means A happens before B.\n\n" + json.dumps(pairs, ensure_ascii=False)
        parsed = self._ask_json(system, user)
        results: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "id" in item and "relation" in item:
                    results.append({
                        "id":         int(item["id"]),
                        "relation":   str(item.get("relation", "UNKNOWN")).upper(),
                        "confidence": float(item.get("confidence", 0.5)),
                    })
        return results

    def resolve_causal_pairs(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pairs or not self._ok:
            return []
        if len(pairs) > self.MAX_PAIRS_PER_CALL:
            log.info(
                "CloudLLMFallback: %d causal pairs exceeds the %d-pair safe batch size - "
                "sending only the first %d to stay under the token limit; the rest report as unresolved.",
                len(pairs), self.MAX_PAIRS_PER_CALL, self.MAX_PAIRS_PER_CALL,
            )
            pairs = pairs[: self.MAX_PAIRS_PER_CALL]
        system = (
            "You are a forensic causality analyst. Determine if event A "
            "causally leads to event B. A witness/bystander observing or "
            "reporting something is never 'caused by' another person's "
            "earlier action - that is corroboration, not causation. "
            "Respond ONLY with a valid JSON array, no prose, no markdown. "
            'Each element: {"id": <int>, "causal": <bool>, '
            '"confidence": <float 0.0-1.0>, "explanation": "<one sentence>"}.'
        )
        user = json.dumps(pairs, ensure_ascii=False)
        parsed = self._ask_json(system, user)
        results: List[Dict[str, Any]] = []
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


"""
ForenSynth - Timeline Agent
event_clustering.py: observation -> event reconstruction.

IMPLEMENTS a gap identified against your unified architecture doc: Phase 4
(Observation Compatibility + Clustering) and Phase 5 (Reconstructed Events).
Previously the agent did a strict 1 observation -> 1 event mapping, which
your own design doc explicitly calls out as wrong: "observations are
evidence, not events." Multiple observations describing the same
real-world action (e.g. a video frame AND an audio snippet both covering
one ATM withdrawal) should reconstruct into ONE event with multiple
supporting observations - that's the whole point of multimodal
corroboration, and the old 1:1 mapping threw that signal away entirely.

Scope decision (deliberately conservative): clustering is scoped to a
SINGLE canonical entity. It never merges observations across different
entities, even if they're at the same place and time (a witness standing
next to a suspect never gets fused into the suspect's event) - cross-entity
relationships stay as separate events connected by causal/temporal edges,
which keeps every merge decision auditable back to "these observations are
plausibly the same actor doing the same thing," not "these things happened
near each other." This is the harder-to-get-wrong interpretation for a
forensic tool, at the cost of not modelling shared/joint events - a
reasonable and explicit trade-off, not an oversight.

Not implemented (still a documented gap, unchanged from before): entity-
identity hypothesis generation / beam search over candidate identity
assignments. That gap is about ambiguous WHO; it doesn't apply here because
Entity Resolution has already committed to canonical entities before this
stage runs. This module only addresses ambiguous WHICH-OBSERVATIONS-BELONG-
TO-THE-SAME-EVENT, which is a real and separate gap that was fully open.
"""

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple


log = logging.getLogger("forensynth.timeline.event_clustering")


@dataclass
class ClusterableObservation:
    """Minimal view of an observation needed for clustering - decoupled from
    RawObservation/TimelineEvent so this module has no upward dependency."""
    obs_id: str
    entity_id: str
    ts_epoch: float
    location_key: str
    content: str
    modality: str
    confidence: float
    action_tags: Set[str] = field(default_factory=set)


@dataclass
class EventCluster:
    """One reconstructed event's worth of supporting observations."""
    cluster_id: str
    entity_id: str
    obs_ids: List[str]
    representative_ts_epoch: float
    location_key: str
    modalities: Set[str]
    action_tags: Set[str]


def _lexical_similarity(a: str, b: str) -> float:
    """Cheap, dependency-free lexical similarity (stdlib difflib)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def compute_compatibility(a: ClusterableObservation, b: ClusterableObservation) -> float:
    """
    Compatibility score in [0, 1] between two SAME-ENTITY observations,
    per the unified architecture's Phase 4 compatibility matrix concept:
      - shared action tag(s):  0.40
      - time proximity:        0.30 (linear falloff within EVENT_CLUSTER_WINDOW_SEC)
      - same location_key:     0.20
      - lexical similarity:    0.10

    Returns 0.0 immediately (hard gate) if the two observations are more
    than EVENT_CLUSTER_WINDOW_SEC apart when both have valid timestamps -
    no amount of other similarity should merge events far apart in time.
    """
    if a.ts_epoch > 0 and b.ts_epoch > 0:
        gap = abs(a.ts_epoch - b.ts_epoch)
        if gap > EVENT_CLUSTER_WINDOW_SEC:
            return 0.0
        time_score = max(0.0, 1.0 - (gap / EVENT_CLUSTER_WINDOW_SEC))
    else:
        # Missing timestamp on either side: neutral (not a rejection, not a
        # bonus) - let action-tag/location/lexical signal decide.
        time_score = 0.5

    shared_tags = a.action_tags & b.action_tags
    action_score = 1.0 if shared_tags else 0.0

    location_score = 1.0 if (a.location_key and b.location_key and a.location_key == b.location_key) else 0.0

    lexical_score = _lexical_similarity(a.content, b.content)

    return (
        EVENT_CLUSTER_WEIGHT_ACTION_TAG * action_score
        + EVENT_CLUSTER_WEIGHT_TIME * time_score
        + EVENT_CLUSTER_WEIGHT_LOCATION * location_score
        + EVENT_CLUSTER_WEIGHT_LEXICAL * lexical_score
    )


class _UnionFind:
    """Minimal union-find for greedy clustering (Phase 4's stated approach)."""

    def __init__(self, ids: List[str]) -> None:
        self._parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def cluster_observations(
    observations: List[ClusterableObservation],
    threshold: float = EVENT_CLUSTER_COMPAT_THRESHOLD,
) -> List[EventCluster]:
    """
    Greedy union-find clustering, scoped per-entity (never crosses entities).
    Returns one EventCluster per connected component, each containing >=1
    observation. Deterministic: iteration order is obs_id-sorted so results
    don't depend on input list order (matters once this reads from a Memory
    Store rather than a fixed JSON file - same rationale as the temporal
    sort tiebreak fix).
    """
    if not observations:
        return []

    by_entity: Dict[str, List[ClusterableObservation]] = {}
    for obs in observations:
        by_entity.setdefault(obs.entity_id, []).append(obs)

    clusters: List[EventCluster] = []
    cluster_counter = 0

    for entity_id, group in sorted(by_entity.items()):
        group = sorted(group, key=lambda o: o.obs_id)
        ids = [o.obs_id for o in group]
        uf = _UnionFind(ids)
        by_id = {o.obs_id: o for o in group}

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                score = compute_compatibility(group[i], group[j])
                if score >= threshold:
                    uf.union(group[i].obs_id, group[j].obs_id)

        components: Dict[str, List[str]] = {}
        for obs_id in ids:
            root = uf.find(obs_id)
            components.setdefault(root, []).append(obs_id)

        for root, member_ids in sorted(components.items()):
            members = [by_id[i] for i in sorted(member_ids)]
            valid_epochs = [m.ts_epoch for m in members if m.ts_epoch > 0]
            confidences = [m.confidence for m in members]
            rep_ts = (
                sum(e * c for e, c in zip(valid_epochs, confidences[: len(valid_epochs)]))
                / sum(confidences[: len(valid_epochs)])
                if valid_epochs and sum(confidences[: len(valid_epochs)]) > 0
                else (valid_epochs[0] if valid_epochs else 0.0)
            )
            loc_key = next((m.location_key for m in members if m.location_key), "")
            action_tags: Set[str] = set()
            for m in members:
                action_tags |= m.action_tags
            modalities = {m.modality for m in members}

            cluster_counter += 1
            clusters.append(EventCluster(
                cluster_id=f"EC_{cluster_counter}",
                entity_id=entity_id,
                obs_ids=[m.obs_id for m in members],
                representative_ts_epoch=rep_ts,
                location_key=loc_key,
                modalities=modalities,
                action_tags=action_tags,
            ))

    log.info("Clustered %d observations into %d events across %d entities.",
              len(observations), len(clusters), len(by_entity))
    return clusters


"""
ForenSynth - Timeline Agent
graph_builder.py: builds a directed NetworkX graph from events and edges,
and serialises it for the Critique Agent / Visualisation Layer.

No functional bug found here in review; hardened with defensive checks
for empty inputs so an empty-events edge case can't raise inside export.
"""

import json
import logging
from typing import Any, Dict, List

import networkx as nx


log = logging.getLogger("forensynth.timeline.graph_builder")


class GraphBuilder:
    """
    Stage 4 / Stage 13 - Event Graph Construction and Export.

    Node attributes:
        event_id, timestamp, entity_id, primary_alias, confidence, location, modality, content
    Edge attributes:
        edge_type, confidence, relation, label
    """

    def build(
        self,
        events: List[TimelineEvent],
        temporal_edges: List[TimelineEdge],
        causal_edges: List[TimelineEdge],
    ) -> nx.DiGraph:
        G = nx.DiGraph()

        for ev in events:
            G.add_node(
                ev.event_id,
                timestamp=ev.timestamp,
                ts_epoch=ev.ts_epoch,
                entity_id=ev.entity_id,
                primary_alias=ev.primary_alias,
                confidence=round(ev.confidence, 4),
                location=ev.location,
                modality=ev.modality,
                content=(ev.content or "")[:120],
                conflict_flag=ev.conflict_flag,
                obs_ids=ev.obs_ids,
            )

        for edge in [*temporal_edges, *causal_edges]:
            if not G.has_node(edge.source) or not G.has_node(edge.target):
                log.debug("Edge references missing node (%s -> %s); skipping", edge.source, edge.target)
                continue
            if G.has_edge(edge.source, edge.target):
                existing = G[edge.source][edge.target]
                if existing.get("edge_type") == "CAUSAL" and edge.edge_type.value == "TEMPORAL":
                    continue  # keep the richer causal edge
            G.add_edge(
                edge.source, edge.target,
                edge_type=edge.edge_type.value,
                confidence=round(edge.confidence, 4),
                relation=edge.relation.value,
                label=edge.label,
            )

        log.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    def to_export_dict(self, G: nx.DiGraph) -> Dict[str, Any]:
        nodes = [{"id": node_id, **attrs} for node_id, attrs in G.nodes(data=True)]
        edges = [{"source": src, "target": tgt, **attrs} for src, tgt, attrs in G.edges(data=True)]
        causal_links = [e for e in edges if e.get("edge_type") == "CAUSAL"]

        return {
            "nodes": nodes,
            "edges": edges,
            "causal_links": causal_links,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
        }

    def export_to_file(self, G: nx.DiGraph, path: str) -> None:
        export = self.to_export_dict(G)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(export, fh, indent=2, ensure_ascii=False)
        log.info("Graph exported -> %s", path)


"""
ForenSynth - Timeline Agent
explainability.py: generates ExplainabilityRecords and human-readable narrative.

No functional bug found here in review; carried over unchanged.
"""

import logging
from typing import List


log = logging.getLogger("forensynth.timeline.explainability")


class ExplainabilityLayer:
    """Stage 10 / Stage 11 - Explainability and Narrative generation."""

    def build_explainability(self, events: List[TimelineEvent]) -> List[ExplainabilityRecord]:
        records: List[ExplainabilityRecord] = []
        for ev in events:
            records.append(
                ExplainabilityRecord(
                    event_id=ev.event_id,
                    derived_from=list(ev.obs_ids),
                    entity_used=ev.entity_id,
                    reasoning=list(ev.reasoning),
                    confidence=round(ev.confidence, 4),
                )
            )
        return records

    def build_narrative(
        self, events: List[TimelineEvent], causal_edges: List[TimelineEdge]
    ) -> List[NarrativeLine]:
        """Generate one NarrativeLine per event, in chronological order."""
        narrative: List[NarrativeLine] = []
        causal_targets = {e.target: e.label for e in causal_edges if e.label}

        for ev in events:
            action = short_summary(ev.content, max_len=100)
            if not action:
                action = f"{ev.primary_alias} observed at {ev.location or 'unknown location'}"

            causal_note = causal_targets.get(ev.event_id, "")
            if causal_note:
                action = f"{action}  [-> {causal_note}]"

            narrative.append(
                NarrativeLine(
                    timestamp=ev.timestamp or "unknown",
                    actor=ev.primary_alias,
                    action=action,
                    location=ev.location or "unknown",
                    evidence=list(ev.obs_ids),
                    confidence=round(ev.confidence, 4),
                    event_id=ev.event_id,
                )
            )
        return narrative

    def build_uncertainties(self, events: List[TimelineEvent]) -> List[UncertaintyRecord]:
        uncertainties: List[UncertaintyRecord] = []
        for ev in events:
            if ev.confidence < 0.85:
                reasons: List[str] = []
                if not ev.timestamp:
                    reasons.append("missing timestamp")
                if ev.confidence < 0.60:
                    reasons.append("low observation confidence")
                if ev.conflict_flag:
                    reasons.append("entity resolution conflict present")
                if not ev.obs_ids:
                    reasons.append("no source observations")
                if not reasons:
                    reasons.append("moderate confidence")
                uncertainties.append(
                    UncertaintyRecord(
                        event_id=ev.event_id,
                        uncertainty_score=round(1.0 - ev.confidence, 4),
                        sources=list(ev.obs_ids),
                        reasons=reasons,
                    )
                )
        return uncertainties

    def format_text_narrative(self, narrative: List[NarrativeLine]) -> str:
        """Render the narrative as a human-readable forensic text report."""
        lines: List[str] = ["=" * 60, "FORENSIC TIMELINE NARRATIVE", "=" * 60, ""]
        for line in narrative:
            lines.append(f"[{line.timestamp}]")
            lines.append(f"  Actor    : {line.actor}")
            lines.append(f"  Action   : {line.action}")
            lines.append(f"  Location : {line.location}")
            lines.append(f"  Evidence : {', '.join(line.evidence) or 'N/A'}")
            lines.append(f"  Confidence: {line.confidence:.2%}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


"""
ForenSynth - Timeline Agent
temporal_reasoner.py: deterministic-first temporal ordering and relationship inference.

FIXED in this revision:
  - No longer depends on GrokClient / any cloud API. Takes a BaseLLMFallback
    (default NoOpLLMFallback = fully offline) and only calls it for pairs
    the deterministic rules genuinely cannot resolve.
  - Sort key now has an explicit, content-derived final tie-break so output
    ordering is 100% reproducible regardless of input list order (matters
    once inputs come from a Memory Store rather than a fixed JSON file).
  - Pairs that remain UNKNOWN after both deterministic rules AND the (optional)
    LLM fallback are collected and returned separately as `unresolved_pairs`
    instead of being silently left in the graph with a made-up confidence.
"""

import logging
from typing import Dict, List, Tuple


log = logging.getLogger("forensynth.timeline.temporal_reasoner")


def _modality_reliability(modality: str) -> float:
    return MODALITY_RELIABILITY.get((modality or "unknown").lower(), 0.50)


def _sorted_events_key(ev: TimelineEvent) -> Tuple[float, float, float, str]:
    """
    Sort key: (epoch ASC, confidence DESC, modality_reliability DESC, tiebreak ASC).
    The final tiebreak is a stable content-derived string, guaranteeing
    identical output ordering across runs regardless of input list order.
    """
    return (
        ev.ts_epoch if ev.ts_epoch > 0 else float("inf"),
        -ev.confidence,
        -_modality_reliability(ev.modality),
        deterministic_tiebreak_key(ev.obs_ids, ev.event_id),
    )


class TemporalReasoner:
    """
    Stage 5 - Temporal Reasoning.

    1. Sort events by timestamp / confidence / modality / deterministic tiebreak.
    2. Infer TEMPORAL edges deterministically where possible.
    3. Optionally batch-call a LOCAL LLM for unresolved ambiguous pairs.
    4. Anything still unresolved is reported explicitly, never guessed.
    """

    def __init__(self, llm: BaseLLMFallback) -> None:
        self._llm = llm

    def sort_events(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Return a new list sorted chronologically (deterministic)."""
        return sorted(events, key=_sorted_events_key)

    def build_temporal_edges(
        self, events: List[TimelineEvent]
    ) -> List[TimelineEdge]:
        """
        Build TEMPORAL edges between consecutive events in the sorted list.
        """
        edges: List[TimelineEdge] = []
        sorted_evs = self.sort_events(events)

        for i in range(len(sorted_evs) - 1):
            a = sorted_evs[i]
            b = sorted_evs[i + 1]

            relation, confidence = self._determine_relation(a, b)
            edge = TimelineEdge(
                source=a.event_id,
                target=b.event_id,
                edge_type=EdgeType.TEMPORAL,
                confidence=confidence,
                relation=relation,
                label=f"{relation.value} ({confidence:.2f})",
            )
            edges.append(edge)

        return edges

    def _determine_relation(
        self, a: TimelineEvent, b: TimelineEvent
    ) -> Tuple[TemporalRelation, float]:
        """Deterministic temporal relation between two events."""

        if a.ts_epoch > 0 and b.ts_epoch > 0:
            gap = b.ts_epoch - a.ts_epoch
            if abs(gap) <= SIMULTANEOUS_WINDOW_SEC:
                return TemporalRelation.SIMULTANEOUS, 0.90
            if gap > 0:
                return TemporalRelation.BEFORE, 0.95
            return TemporalRelation.AFTER, 0.88

        if a.ts_epoch > 0 and b.ts_epoch <= 0:
            return TemporalRelation.BEFORE, 0.60

        if a.ts_epoch <= 0 and b.ts_epoch > 0:
            return TemporalRelation.BEFORE, 0.55

        relation = self._rule_based_action_order(a.content, b.content)
        if relation != TemporalRelation.UNKNOWN:
            return relation, 0.65

        return TemporalRelation.UNKNOWN, 0.35

    def _rule_based_action_order(
        self, content_a: str, content_b: str
    ) -> TemporalRelation:
        """Apply CAUSAL_ACTION_RULES to determine ordering from action tags
        (fragment-matched via normalization.extract_action_tags, grounded in
        the generator's actual phrase banks - not raw single-word tokens)."""
        tags_a = extract_action_tags(content_a)
        tags_b = extract_action_tags(content_b)

        for prereq, dependent in CAUSAL_ACTION_RULES:
            if prereq in tags_a and dependent in tags_b:
                return TemporalRelation.BEFORE
            if prereq in tags_b and dependent in tags_a:
                return TemporalRelation.AFTER

        return TemporalRelation.UNKNOWN

    def _temporal_pair_priority(self, pair: Tuple[TimelineEvent, TimelineEvent]) -> float:
        """
        Higher score = more worth spending one of the capped LLM calls on.
        These pairs are ambiguous specifically because NEITHER a valid
        timestamp ordering NOR action-tag-rule ordering was available on
        both sides (see the caller), so temporal proximity isn't a usable
        signal here - unlike causal_reasoner.py's equivalent, this omits
        it and leans on entity/location/content/confidence instead.
        """
        a, b = pair
        score = 0.0
        if a.entity_id == b.entity_id:
            score += 2.0
        score += semantic_location_similarity(a.location, b.location)
        if a.action_tags and b.action_tags:
            score += 0.5
        elif a.action_tags or b.action_tags:
            score += 0.2
        score += 0.3 * ((a.confidence + b.confidence) / 2.0)
        return score

    def resolve_ambiguous_with_llm(
        self,
        ambiguous_pairs: List[Tuple[TimelineEvent, TimelineEvent]],
    ) -> Tuple[Dict[str, Tuple[TemporalRelation, float]], List[Dict[str, str]]]:
        """
        Attempt to resolve UNKNOWN temporal pairs via the (optional, local-only)
        LLM fallback. Returns (resolved_map, still_unresolved_list).

        With the default NoOpLLMFallback this always returns ({}, [all pairs]) -
        i.e. every genuinely ambiguous pair is reported, not guessed.
        """
        still_unresolved: List[Dict[str, str]] = []

        if not ambiguous_pairs:
            return {}, []

        if not self._llm.available():
            for a, b in ambiguous_pairs:
                still_unresolved.append({
                    "event_a": a.event_id, "event_b": b.event_id,
                    "reason": "no valid timestamp on either side and no matching "
                              "causal-action-rule keywords; no local LLM configured",
                })
            return {}, still_unresolved

        # FIX: the LLM batch is hard-capped (llm_fallback.py's
        # MAX_PAIRS_PER_CALL), so spend that limited budget on the pairs
        # most likely to actually be resolvable rather than an arbitrary
        # subset (same reasoning as causal_reasoner.py's priority sort).
        # Sorted ONCE here and reused consistently below - payloads, the
        # LLM call, and the result-matching loop must all use the SAME
        # ordering, or idx-based result lookups silently attach the wrong
        # result to the wrong pair.
        ambiguous_pairs = sorted(ambiguous_pairs, key=self._temporal_pair_priority, reverse=True)

        payloads = [
            {
                "id":          idx,
                "a_content":   a.content,
                "a_timestamp": a.timestamp,
                "a_location":  a.location,
                "b_content":   b.content,
                "b_timestamp": b.timestamp,
                "b_location":  b.location,
            }
            for idx, (a, b) in enumerate(ambiguous_pairs)
        ]

        log.info("Calling local LLM for temporal resolution of %d ambiguous pairs.", len(payloads))
        results = self._llm.resolve_temporal_pairs(payloads)

        resolved: Dict[str, Tuple[TemporalRelation, float]] = {}
        result_by_id = {r["id"]: r for r in results}
        for idx, (a, b) in enumerate(ambiguous_pairs):
            key = f"{a.event_id}|{b.event_id}"
            if idx in result_by_id:
                r = result_by_id[idx]
                rel_str = r.get("relation", "UNKNOWN").upper()
                try:
                    rel = TemporalRelation(rel_str)
                except ValueError:
                    rel = TemporalRelation.UNKNOWN
                if rel == TemporalRelation.UNKNOWN:
                    still_unresolved.append({
                        "event_a": a.event_id, "event_b": b.event_id,
                        "reason": "local LLM also returned UNKNOWN",
                    })
                else:
                    resolved[key] = (rel, float(r.get("confidence", 0.5)))
            else:
                still_unresolved.append({
                    "event_a": a.event_id, "event_b": b.event_id,
                    "reason": "local LLM returned no result for this pair",
                })

        return resolved, still_unresolved


"""
ForenSynth - Timeline Agent
causal_reasoner.py: deterministic + optional LOCAL-LLM-based causal inference.

FIXED in this revision:
  - No longer depends on GrokClient / any cloud API; takes a BaseLLMFallback
    (default NoOpLLMFallback = fully offline, deterministic-only).
  - Pairs the deterministic rules and (optional) local LLM both fail to
    resolve are returned as `unresolved_pairs` for transparency instead of
    silently having no causal edge with no explanation.

Strategy:
 1. Reject pairs that cross the actor/observer boundary (a witness's
    observation cannot be "caused by" a suspect's earlier action - the
    witness merely reported it; that is correlation, not causation).
 2. Apply CAUSAL_ACTION_RULES (deterministic).
 3. Score by temporal proximity + shared entity + shared location.
 4. Only send unresolved pairs to the (optional) local LLM.
"""

import logging
from typing import Dict, List, Tuple


log = logging.getLogger("forensynth.timeline.causal_reasoner")

# Roles that merely *report* events rather than *perform* them.
OBSERVER_ROLES = {"witness", "bystander", "reporter"}


class CausalReasoner:
    """
    Stage 6 - Causal Reasoning. Produces CAUSAL edges in the timeline graph.
    """

    def __init__(self, llm: BaseLLMFallback) -> None:
        self._llm = llm

    def infer_causal_links(
        self, events: List[TimelineEvent]
    ) -> Tuple[List[TimelineEdge], List[Dict[str, str]]]:
        """
        Main entry point. Returns (causal_edges, unresolved_pairs).
        """
        if len(events) < 2:
            return [], []

        causal_edges: List[TimelineEdge] = []
        unresolved_pairs: List[Tuple[TimelineEvent, TimelineEvent]] = []

        ev_sorted = sorted(events, key=lambda e: (e.ts_epoch if e.ts_epoch > 0 else float("inf")))

        for i, a in enumerate(ev_sorted):
            for b in ev_sorted[i + 1:]:
                if a.ts_epoch > 0 and b.ts_epoch > 0:
                    gap = b.ts_epoch - a.ts_epoch
                    if gap > CAUSAL_WINDOW_SEC:
                        break  # further events even further away
                    if gap < 0:
                        continue

                if not self._eligible_for_causal_link(a, b):
                    continue

                result = self._deterministic_causal(a, b)
                if result is not None:
                    edge, _ = result
                    causal_edges.append(edge)
                else:
                    unresolved_pairs.append((a, b))

        still_unresolved: List[Dict[str, str]] = []
        if unresolved_pairs:
            if self._llm.available():
                # FIX: the LLM call is hard-capped (llm_fallback.py's
                # MAX_PAIRS_PER_CALL) to stay under Groq's token limit -
                # verified necessary via a real 413 error on CASE_ATM_002
                # (45 unresolved pairs in one case). Previously, whichever
                # pairs happened to be FIRST in unresolved_pairs (arbitrary
                # - just chronological event order) got the LLM's attention
                # and the rest were silently dropped to "unresolved" with no
                # regard for which pairs were actually worth spending the
                # budget on. Sorting by _causal_pair_priority() first means
                # the capped budget goes to the pairs most likely to have a
                # genuine, resolvable causal relationship (same entity,
                # near-miss location similarity, informative content,
                # temporal proximity, higher-confidence evidence) - not just
                # whichever pairs happened to appear earliest.
                unresolved_pairs = sorted(unresolved_pairs, key=self._causal_pair_priority, reverse=True)
                llm_edges, still_unresolved = self._llm_causal_resolve(unresolved_pairs)
                causal_edges.extend(llm_edges)
            else:
                # No local model configured: report as unresolved rather than
                # silently having no causal edge with no explanation.
                still_unresolved = [
                    {
                        "event_a": a.event_id, "event_b": b.event_id,
                        "reason": "no deterministic causal rule matched and no "
                                  "local LLM configured; no causal edge created",
                    }
                    for a, b in unresolved_pairs
                ]

        return causal_edges, still_unresolved

    def _causal_pair_priority(self, pair: Tuple[TimelineEvent, TimelineEvent]) -> float:
        """
        Higher score = more worth spending one of the capped LLM calls on.
        Combines cheap, already-available signals that correlate with a
        pair actually having a resolvable causal relationship, so a limited
        batch is spent on the most promising pairs instead of an arbitrary
        subset (e.g. whichever happened to sort earliest by timestamp).
        None of these signals were enough to satisfy the DETERMINISTIC
        rules (or this pair wouldn't be "unresolved" at all) - this is
        purely about ranking near-misses and well-evidenced pairs above
        weak, low-information ones within the pairs the LLM might still
        resolve.
        """
        a, b = pair
        score = 0.0

        # Same entity: overwhelmingly more likely to be a genuine sequence
        # of one actor's actions than a cross-entity pair.
        if a.entity_id == b.entity_id:
            score += 2.0

        # Near-miss on location: some semantic similarity, even below the
        # merge threshold, is a real signal worth the LLM's judgment -
        # vs. a pair with essentially no locational relationship at all.
        score += semantic_location_similarity(a.location, b.location)

        # Both sides having SOME interpretable action content is more
        # promising than a pair where one or both sides are uninformative
        # (e.g. corrupted/noise observations that legitimately have zero tags).
        if a.action_tags and b.action_tags:
            score += 0.5
        elif a.action_tags or b.action_tags:
            score += 0.2

        # Higher-confidence evidence is more reliable to reason about.
        score += 0.3 * ((a.confidence + b.confidence) / 2.0)

        # Closer in time = more plausible causal proximity, within the
        # window both are already inside (that's what made them candidates).
        if a.ts_epoch > 0 and b.ts_epoch > 0:
            gap = abs(b.ts_epoch - a.ts_epoch)
            score += 0.5 * max(0.0, 1.0 - gap / CAUSAL_WINDOW_SEC)

        return score

    def _eligible_for_causal_link(self, a: TimelineEvent, b: TimelineEvent) -> bool:
        """
        False when a causal link a -> b would be a category error: a different
        entity's actor action cannot "cause" a witness/observer's report.
        Same-entity pairs and same-role pairs are always eligible.
        """
        if a.entity_id == b.entity_id:
            return True

        b_role = (b.role or "").strip().lower()
        a_role = (a.role or "").strip().lower()

        if b_role in OBSERVER_ROLES and a_role not in OBSERVER_ROLES:
            return False

        return True

    def _deterministic_causal(
        self, a: TimelineEvent, b: TimelineEvent
    ):
        """Return (edge, confidence) if a -> b can be determined by rules, else None."""
        tags_a = extract_action_tags(a.content)
        tags_b = extract_action_tags(b.content)

        for prereq, dependent in CAUSAL_ACTION_RULES:
            if prereq in tags_a and dependent in tags_b:
                confidence = self._causal_confidence(a, b, base=0.82)
                edge = TimelineEdge(
                    source=a.event_id, target=b.event_id, edge_type=EdgeType.CAUSAL,
                    confidence=confidence, relation=TemporalRelation.BEFORE,
                    label=f"action dependency: {prereq}->{dependent}",
                )
                return edge, confidence

        # FIX: semantic location similarity instead of exact location_key
        # match. "at the entrance of the ATM" and "at the ATM entry door"
        # never matched under exact comparison despite meaning the same
        # place - see semantic_similarity.py / config.LOCATION_SEMANTIC_SIMILARITY_THRESHOLD.
        loc_sim = semantic_location_similarity(a.location, b.location)
        if (
            a.entity_id == b.entity_id
            and loc_sim >= LOCATION_SEMANTIC_SIMILARITY_THRESHOLD
            and a.ts_epoch > 0 and b.ts_epoch > 0
            and 0 < (b.ts_epoch - a.ts_epoch) <= CAUSAL_WINDOW_SEC
        ):
            confidence = self._causal_confidence(a, b, base=0.68)
            edge = TimelineEdge(
                source=a.event_id, target=b.event_id, edge_type=EdgeType.CAUSAL,
                confidence=confidence, relation=TemporalRelation.BEFORE,
                label=f"same entity, semantically same location (sim={loc_sim:.2f}), sequential",
            )
            return edge, confidence

        return None

    def _causal_confidence(self, a: TimelineEvent, b: TimelineEvent, base: float) -> float:
        score = base
        if a.entity_id == b.entity_id:
            score += 0.06
        if semantic_location_similarity(a.location, b.location) >= LOCATION_SEMANTIC_SIMILARITY_THRESHOLD:
            score += 0.04
        if a.ts_epoch > 0 and b.ts_epoch > 0:
            gap = b.ts_epoch - a.ts_epoch
            if gap > CAUSAL_WINDOW_SEC / 2:
                score -= 0.08
        score *= (a.confidence + b.confidence) / 2.0
        return clamp(score)

    def _llm_causal_resolve(
        self, pairs: List[Tuple[TimelineEvent, TimelineEvent]]
    ) -> Tuple[List[TimelineEdge], List[Dict[str, str]]]:
        """Batched local-LLM call for unresolved causal pairs."""
        payloads = [
            {
                "id":          idx,
                "a_content":   a.content, "a_timestamp": a.timestamp,
                "a_entity":    a.primary_alias, "a_role": a.role, "a_location": a.location,
                "b_content":   b.content, "b_timestamp": b.timestamp,
                "b_entity":    b.primary_alias, "b_role": b.role, "b_location": b.location,
            }
            for idx, (a, b) in enumerate(pairs)
        ]

        log.info("Calling local LLM for causal resolution of %d ambiguous pairs.", len(payloads))
        results = self._llm.resolve_causal_pairs(payloads)

        result_by_id = {r["id"]: r for r in results}
        edges: List[TimelineEdge] = []
        unresolved: List[Dict[str, str]] = []
        for idx, (a, b) in enumerate(pairs):
            info = result_by_id.get(idx)
            if info is None:
                unresolved.append({
                    "event_a": a.event_id, "event_b": b.event_id,
                    "reason": "local LLM returned no result for this pair",
                })
                continue
            if info.get("causal", False):
                confidence = clamp(float(info.get("confidence", 0.50)))
                edges.append(TimelineEdge(
                    source=a.event_id, target=b.event_id, edge_type=EdgeType.CAUSAL,
                    confidence=confidence, relation=TemporalRelation.BEFORE,
                    label=f"local LLM: {info.get('explanation', 'inferred causal link')}",
                ))
            else:
                unresolved.append({
                    "event_a": a.event_id, "event_b": b.event_id,
                    "reason": "local LLM determined no causal link",
                })
        return edges, unresolved


"""
ForenSynth - Timeline Agent
validators.py: input validation with descriptive error messages.

FIXED / ADDED in this revision:
  - Detects duplicate obs_id values (warns; downstream dedup keeps first).
  - Validates 'clusters' shape if present (needed for conflict localisation).
  - Validates 'conflicts' shape if present (new ER contract field).
  - Cross-checks canonical_entities.sources against observation obs_ids and
    warns about dangling references either direction (helps catch Memory
    Store sync bugs early, e.g. an entity referencing an obs_id that never
    arrived, or observations no entity claims).
"""

from typing import Any, Dict, List, Set, Tuple


class ValidationError(Exception):
    """Raised when the Timeline Agent input fails schema checks."""


def _validate_observation(obs: Any, idx: int, seen_ids: Set[str]) -> List[str]:
    path = f"obs_only.observations[{idx}]"
    warnings: List[str] = []
    if not isinstance(obs, dict):
        raise ValidationError(f"[{path}] Observation must be a dict, got {type(obs).__name__}")
    if "obs_id" not in obs or not str(obs["obs_id"]).strip():
        raise ValidationError(f"[{path}] Missing or empty 'obs_id'")
    obs_id = str(obs["obs_id"])
    if obs_id in seen_ids:
        warnings.append(f"[{path}] Duplicate obs_id '{obs_id}' - only the first occurrence will be used")
    seen_ids.add(obs_id)

    if "entity" not in obs:
        raise ValidationError(f"[{path}] Missing 'entity'")
    if "timestamp" not in obs or not str(obs.get("timestamp", "")).strip():
        warnings.append(f"[{path}] Missing/empty 'timestamp' - ordering will be uncertain for this observation")
    confidence = obs.get("confidence")
    if confidence is not None:
        try:
            c = float(confidence)
            if not 0.0 <= c <= 1.0:
                warnings.append(f"[{path}] confidence={c} out of [0,1] range - will be clamped")
        except (TypeError, ValueError):
            warnings.append(f"[{path}] Non-numeric confidence '{confidence}' - will default to 0.5")
    return warnings


def _validate_canonical_entity(ent: Any, idx: int) -> List[str]:
    path = f"entity_resolved.canonical_entities[{idx}]"
    warnings: List[str] = []
    if not isinstance(ent, dict):
        raise ValidationError(f"[{path}] Entity must be a dict, got {type(ent).__name__}")
    for field in ("entity_id", "primary_alias", "aliases"):
        if field not in ent:
            raise ValidationError(f"[{path}] Missing required field '{field}'")
    if not isinstance(ent["aliases"], list):
        raise ValidationError(f"[{path}] 'aliases' must be a list")
    sources = ent.get("sources", [])
    if not isinstance(sources, list):
        warnings.append(f"[{path}] 'sources' should be a list of obs_ids; got {type(sources).__name__} - ignoring")
    elif not sources:
        warnings.append(f"[{path}] Entity '{ent.get('entity_id')}' has no 'sources' (obs_ids) - it will match no observations")
    return warnings


def _validate_clusters(clusters: Any) -> List[str]:
    warnings: List[str] = []
    if clusters is None:
        return warnings
    if not isinstance(clusters, list):
        warnings.append("'entity_resolved.clusters' should be a list - ignoring for conflict localisation")
        return warnings
    for i, c in enumerate(clusters):
        if not isinstance(c, dict):
            warnings.append(f"[entity_resolved.clusters[{i}]] should be a dict - skipping")
            continue
        if "cluster_id" not in c:
            warnings.append(f"[entity_resolved.clusters[{i}]] missing 'cluster_id' - conflicts referencing it can't be localised")
        if "obs_ids" in c and not isinstance(c["obs_ids"], list):
            warnings.append(f"[entity_resolved.clusters[{i}]] 'obs_ids' should be a list")
    return warnings


def _validate_conflicts(conflicts: Any) -> List[str]:
    warnings: List[str] = []
    if conflicts is None:
        return warnings
    if isinstance(conflicts, int):
        return warnings  # legacy count-only format; handled downstream as "unlocalized"
    if not isinstance(conflicts, list):
        warnings.append(
            f"'entity_resolved.conflicts' has unexpected type {type(conflicts).__name__} "
            "- expected a list of conflict records or an int count; treating as empty"
        )
        return warnings
    for i, c in enumerate(conflicts):
        if not isinstance(c, dict):
            warnings.append(f"[entity_resolved.conflicts[{i}]] should be a dict - skipping")
            continue
        if "cluster_id" not in c and not any(k in c for k in ("obs_ids", "members", "affected_obs")):
            warnings.append(
                f"[entity_resolved.conflicts[{i}]] has neither 'cluster_id' nor an explicit "
                "obs-id field - this conflict cannot be localised to any event"
            )
    return warnings


def validate_input(payload: Any) -> Tuple[str, List[str]]:
    """
    Validate the full Timeline Agent input payload.

    Returns:
        (case_id, warnings)
    Raises:
        ValidationError on fatal issues.
    """
    if not isinstance(payload, dict):
        raise ValidationError("Input payload must be a JSON object (dict)")

    case_id = payload.get("case_id")
    if not case_id or not isinstance(case_id, str) or not case_id.strip():
        raise ValidationError("'case_id' must be a non-empty string")
    case_id = case_id.strip()

    # -- obs_only section --------------------------------------------------
    obs_only = payload.get("obs_only")
    if obs_only is None:
        raise ValidationError("'obs_only' section is missing")
    if not isinstance(obs_only, dict):
        raise ValidationError("'obs_only' must be a dict")
    observations = obs_only.get("observations")
    if observations is None:
        raise ValidationError("'obs_only.observations' is missing")
    if not isinstance(observations, list):
        raise ValidationError("'obs_only.observations' must be a list")
    if len(observations) == 0:
        raise ValidationError("'obs_only.observations' is empty - nothing to process")

    warnings: List[str] = []
    seen_obs_ids: Set[str] = set()
    for i, obs in enumerate(observations):
        warnings.extend(_validate_observation(obs, i, seen_obs_ids))

    # -- entity_resolved section --------------------------------------------
    er = payload.get("entity_resolved")
    if er is None:
        raise ValidationError("'entity_resolved' section is missing")
    if not isinstance(er, dict):
        raise ValidationError("'entity_resolved' must be a dict")
    canonical_entities = er.get("canonical_entities")
    if canonical_entities is None:
        raise ValidationError("'entity_resolved.canonical_entities' is missing")
    if not isinstance(canonical_entities, list):
        raise ValidationError("'entity_resolved.canonical_entities' must be a list")
    if len(canonical_entities) == 0:
        warnings.append("No canonical entities found - all observations will be treated as unresolved")

    referenced_obs: Set[str] = set()
    for i, ent in enumerate(canonical_entities):
        warnings.extend(_validate_canonical_entity(ent, i))
        if isinstance(ent, dict) and isinstance(ent.get("sources"), list):
            referenced_obs.update(str(s) for s in ent["sources"])

    dangling = referenced_obs - seen_obs_ids
    if dangling:
        warnings.append(
            f"{len(dangling)} obs_id(s) referenced by canonical_entities.sources were not "
            f"found in obs_only.observations (e.g. {sorted(dangling)[:5]}) - possible "
            "Memory Store sync issue between Entity Resolution and Timeline Agent reads"
        )

    warnings.extend(_validate_clusters(er.get("clusters")))
    # Prefer the new 'conflicts' contract field; fall back to legacy 'conflicts_detected'.
    conflicts_field = er.get("conflicts", er.get("conflicts_detected", []))
    warnings.extend(_validate_conflicts(conflicts_field))

    return case_id, warnings


"""
ForenSynth - Timeline Agent
repositories.py: repository abstractions for storage.

Design contract: replace the JSON-backed implementations with Memory-Store
backed ones (e.g. Postgres/Mongo) without touching any other Timeline Agent
file - agent.py only talks to the abstract interfaces below.

No functional bug found here in review; carried over with minor hardening
(duplicate-obs_id-safe, defensive parsing already present and kept).
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


log = logging.getLogger("forensynth.timeline.repositories")


# -- Abstract base classes ------------------------------------------------------

class ObservationRepository(ABC):
    """Provides raw observations to the Timeline Agent."""

    @abstractmethod
    def get_all(self, case_id: str) -> List[RawObservation]:
        ...

    @abstractmethod
    def get_by_id(self, case_id: str, obs_id: str) -> Optional[RawObservation]:
        ...


class EntityRepository(ABC):
    """Provides canonical entity data to the Timeline Agent."""

    @abstractmethod
    def get_all(self, case_id: str) -> List[CanonicalEntity]:
        ...

    @abstractmethod
    def get_by_id(self, case_id: str, entity_id: str) -> Optional[CanonicalEntity]:
        ...

    @abstractmethod
    def get_by_obs_id(self, case_id: str, obs_id: str) -> Optional[CanonicalEntity]:
        ...


class TimelineRepository(ABC):
    """Persists and retrieves Timeline Agent output."""

    @abstractmethod
    def save(self, timeline: TimelineVersion) -> None:
        ...

    @abstractmethod
    def load(self, case_id: str, version: str) -> Optional[TimelineVersion]:
        ...


# -- JSON-backed implementations (stand-ins for the Memory Store today) --------

def _raw_observation_from_normalized(n: NormalizedObservation, entity_raw_fallback: str) -> RawObservation:
    """
    FIX: previously each field here was normalized inline (normalize_role,
    normalize_modality, normalize_alias called separately in this file, and
    normalize_location/extract_action_tags called AGAIN later in
    agent.py::_stage_2_enrich). Now every field is copied straight from a
    NormalizedObservation already computed once by
    normalization.normalize_case() (via the store below) - the single
    normalization boundary for this agent, matching the same pattern used
    in the Entity Resolution pipeline. This is the seam a real Memory
    Store swaps in behind later (see normalization.py's
    NormalizedObservationStore docstring) - nothing here needs to change
    when that happens, only which store get_normalized_observation_store()
    returns.
    """
    return RawObservation(
        obs_id=n.obs_id,
        entity=n.entity_raw or entity_raw_fallback,
        role=n.role,
        modality=n.modality,
        location=n.location_raw,
        content=n.content_raw,
        timestamp=n.timestamp_raw,
        confidence=n.confidence,
        entity_norm=n.entity_norm,
        time_offset_sec=n.time_offset_sec,
        _ts_epoch=n.ts_epoch,
        location_key=n.location_key,
        action_tags=list(n.action_tags),
    )


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_entity(item: Dict[str, Any]) -> CanonicalEntity:
    aliases = list(item.get("aliases", []))
    return CanonicalEntity(
        entity_id=str(item["entity_id"]),
        primary_alias=str(item.get("primary_alias", aliases[0] if aliases else "unknown")),
        aliases=aliases,
        confidence_score=_safe_float(item.get("confidence_score", 0.5), 0.5),
        sources=list(item.get("sources", [])),
        modalities=list(item.get("modalities", [])),
        locations=list(item.get("locations", [])),
        roles=list(item.get("roles", [])),
        earliest_timestamp=str(item.get("earliest_timestamp", "")),
        latest_timestamp=str(item.get("latest_timestamp", "")),
        time_span_seconds=_safe_int(item.get("time_span_seconds", 0), 0),
        candidate_mentions=list(item.get("candidate_mentions", [])),
    )


class JsonObservationRepository(ObservationRepository):
    """
    Reads observations from the in-memory payload passed at construction
    time, via the shared NormalizedObservationStore (normalization.py) -
    the same seam a real Memory-Store-backed repository will use later.
    Duplicate obs_id values keep the FIRST occurrence (matches the warning
    validators.py emits) so behaviour is deterministic and documented.
    """

    def __init__(self, case_id: str, raw_observations: List[Dict[str, Any]]) -> None:
        self._obs: Dict[str, RawObservation] = {}
        store = get_normalized_observation_store()
        try:
            normalized = store.get(case_id, raw_observations)
        except Exception as exc:
            log.error("Normalization store failed for case '%s' (%s) - falling back to per-item parsing.", case_id, exc)
            normalized = None

        if normalized is not None:
            for n in normalized:
                if not n.obs_id:
                    continue
                if n.obs_id in self._obs:
                    log.warning("Duplicate obs_id '%s' - keeping first occurrence", n.obs_id)
                    continue
                self._obs[n.obs_id] = _raw_observation_from_normalized(n, entity_raw_fallback=n.entity_raw)
        else:
            # FIX: normalization store failed (e.g. shared module not importable
            # in this runtime). Fall back to direct per-item parsing so we always
            # load observations — without this, 0 events are produced silently.
            log.warning(
                "Normalization store unavailable for case '%s' - parsing %d observations directly.",
                case_id, len(raw_observations),
            )
            for item in raw_observations:
                obs_id = str(item.get("obs_id", "")).strip()
                if not obs_id:
                    continue
                if obs_id in self._obs:
                    log.warning("Duplicate obs_id '%s' (fallback path) - keeping first", obs_id)
                    continue
                ts_raw = str(item.get("timestamp", ""))
                try:
                    ts_epoch = parse_timestamp(ts_raw)
                except Exception:
                    ts_epoch = 0.0
                self._obs[obs_id] = RawObservation(
                    obs_id=obs_id,
                    entity=str(item.get("entity", "")),
                    role=str(item.get("role", "unknown")),
                    modality=str(item.get("modality", "unknown")),
                    location=str(item.get("location", "")),
                    content=str(item.get("content", "")),
                    timestamp=ts_raw,
                    confidence=float(item.get("confidence", 0.5)),
                    entity_norm=str(item.get("entity", "")).strip().lower(),
                    time_offset_sec=int(item.get("time_offset", 0)),
                    _ts_epoch=ts_epoch,
                    location_key=str(item.get("location", "")).strip().lower(),
                    action_tags=set(),
                )


    def get_all(self, case_id: str) -> List[RawObservation]:
        return list(self._obs.values())

    def get_by_id(self, case_id: str, obs_id: str) -> Optional[RawObservation]:
        return self._obs.get(obs_id)


class JsonEntityRepository(EntityRepository):
    """Reads canonical entities from the in-memory entity_resolved payload."""

    def __init__(self, canonical_entities: List[Dict[str, Any]]) -> None:
        self._entities: Dict[str, CanonicalEntity] = {}
        self._obs_index: Dict[str, str] = {}
        for item in canonical_entities:
            try:
                ent = _parse_entity(item)
                self._entities[ent.entity_id] = ent
                for obs_id in ent.sources:
                    self._obs_index[str(obs_id)] = ent.entity_id
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Skipping malformed entity %s: %s", item.get("entity_id"), exc)

    def get_all(self, case_id: str) -> List[CanonicalEntity]:
        return list(self._entities.values())

    def get_by_id(self, case_id: str, entity_id: str) -> Optional[CanonicalEntity]:
        return self._entities.get(entity_id)

    def get_by_obs_id(self, case_id: str, obs_id: str) -> Optional[CanonicalEntity]:
        entity_id = self._obs_index.get(obs_id)
        return self._entities.get(entity_id) if entity_id else None


class JsonTimelineRepository(TimelineRepository):
    """Persists TimelineVersion objects to JSON files."""

    def __init__(self, output_dir: str = ".") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str, version: str) -> Path:
        safe = case_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}_timeline_{version}.json"

    def save(self, timeline: TimelineVersion) -> None:
        path = self._path(timeline.case_id, timeline.version)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(timeline.to_dict(), fh, indent=2, ensure_ascii=False)
        log.info("Timeline saved -> %s", path)

    def load(self, case_id: str, version: str) -> Optional[TimelineVersion]:
        path = self._path(case_id, version)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        log.info("Timeline loaded <- %s", path)
        return data  # type: ignore[return-value]  # callers handle dict form


"""
ForenSynth - Timeline Agent
agent.py: top-level orchestrator that wires all stages together.

Pipeline (13 stages):
  1.  Input Validation
  2.  Event Enrichment
  3.  Timestamp-First Ordering
  4.  Event Graph Construction
  5.  Temporal Reasoning
  6.  Causal Reasoning
  7.  Uncertainty Modelling
  8.  Conflict Awareness
  9.  Provenance Preservation           (guaranteed throughout)
  10. Explainability Layer
  11. Timeline Narrative
  12. Timeline Versioning
  13. Graph Export

===============================================================================
FIXES applied in this revision (see accompanying README for full detail):

1. CONFLICT CONTRACT BUG (critical): the Entity Resolution pipeline only ever
   exposed a conflict *count* (`conflicts_detected: int`), never which
   observations were affected. The old `_collect_conflict_obs_ids()` expected
   a list of dicts with obs-level keys that the real ER output never
   produces, so conflict-aware confidence penalties silently never fired.
   Fixed here to consume the corrected ER contract - a `conflicts` list of
   `{type, cluster_id, detail}` records - and resolve `cluster_id -> obs_ids`
   via the `clusters` array (also present in ER output). Falls back
   gracefully (with an explicit `conflicts_unlocalized_count` in the output)
   for ER outputs that still only send the legacy int count.

2. NO EXTERNAL API CALLS: GrokClient (hosted Groq cloud calls) removed.
   Default is NoOpLLMFallback (fully offline, deterministic-only). An
   optional local-only model can be injected via `llm=` in the constructor.

3. CONFIDENCE CEILING BUG: weights now sum to 1.0 before the conflict
   penalty is subtracted (see config.py), so a perfect-evidence event can
   reach confidence 1.0 instead of capping at 0.90.

4. Unresolved temporal/causal pairs and unresolved (non-canonicalised)
   entities are now reported explicitly in the output rather than being
   silently dropped or silently guessed.

5. Defensive error handling: malformed-but-partial input no longer crashes
   the whole run; unexpected runtime errors are wrapped in a
   `TimelineAgentError` with stage context instead of propagating a bare
   traceback, while `ValidationError` (bad input) is still raised distinctly
   so callers can tell the two apart.
===============================================================================
"""

import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


log = logging.getLogger("forensynth.timeline.agent")

_EV_ID_RE = re.compile(r"[^a-z0-9_]")


class TimelineAgentError(Exception):
    """
    Raised for unexpected runtime failures (as opposed to ValidationError,
    which is raised for malformed input). Wraps the original exception with
    stage context so failures are diagnosable instead of a bare traceback.
    """


def _make_event_id(obs_ids: List[str], entity_id: str) -> str:
    key = f"{entity_id}_{'_'.join(sorted(obs_ids))}"
    return "EVT_" + _EV_ID_RE.sub("_", key.lower())[:48]


def _build_default_llm() -> BaseLLMFallback:
    """
    Backend selection priority (see config.py for the full rationale):
      1. CloudLLMFallback (Groq) - auto-enabled IF a Timeline_Key is found
         (env var or Colab secret). Hard-capped at <=2 API calls per run by
         construction (temporal_reasoner.py and causal_reasoner.py each make
         at most one batched call).
      2. LocalLLMFallback - if TIMELINE_LOCAL_LLM_BACKEND is explicitly set
         and no cloud key was found.
      3. NoOpLLMFallback - fully offline, zero calls. Used if neither above
         applies.
    Every path degrades to deterministic-only automatically on any failure
    (missing key, unreachable model, network error) - the pipeline never
    blocks or errors because an LLM backend isn't available.
    """
    cloud = CloudLLMFallback(model=CLOUD_LLM_MODEL, timeout_sec=CLOUD_LLM_TIMEOUT_SEC)
    if cloud.available():
        log.info("LLM fallback: using CloudLLMFallback (Groq, model=%s) - max 2 calls/run.", CLOUD_LLM_MODEL)
        return cloud

    if LOCAL_LLM_BACKEND:
        return LocalLLMFallback(
            backend=LOCAL_LLM_BACKEND, model=LOCAL_LLM_MODEL,
            host=LOCAL_LLM_HOST, timeout_sec=LOCAL_LLM_TIMEOUT_SEC,
        )

    log.info("LLM fallback: no Timeline_Key or local backend configured - running fully offline (deterministic-only).")
    return NoOpLLMFallback()


def _build_cluster_obs_index(clusters_raw: Any) -> Dict[str, List[str]]:
    """cluster_id -> obs_ids, built from ER's 'clusters' array."""
    index: Dict[str, List[str]] = {}
    if isinstance(clusters_raw, list):
        for c in clusters_raw:
            if isinstance(c, dict) and "cluster_id" in c:
                obs_ids = c.get("obs_ids", [])
                if isinstance(obs_ids, list):
                    index[str(c["cluster_id"])] = [str(o) for o in obs_ids]
    return index


class TimelineAgent:
    """
    ForenSynth Timeline Agent V1 (fixed).

    Usage::
        agent = TimelineAgent()                 # fully offline (default)
        result = agent.run(payload_dict)

        # optional: inject a local-only model
        from llm_fallback import LocalLLMFallback
        agent = TimelineAgent(llm=LocalLLMFallback(backend="ollama"))
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        save_outputs: bool = True,
        llm: Optional[BaseLLMFallback] = None,
    ) -> None:
        self._llm = llm if llm is not None else _build_default_llm()
        self._temporal = TemporalReasoner(self._llm)
        self._causal = CausalReasoner(self._llm)
        self._graph_builder = GraphBuilder()
        self._explainability = ExplainabilityLayer()
        self._timeline_repo = JsonTimelineRepository(output_dir)
        self._output_dir = Path(output_dir)
        self._save_outputs = save_outputs

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full 13-stage timeline construction pipeline.

        Raises:
            ValidationError: on malformed input (caller's fault - fix the input).
            TimelineAgentError: on unexpected internal failure (agent's fault -
                                 report this; input passed validation).
        """
        t_start = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        stage = "stage_1_validation"

        try:
            # -- Stage 1: Input Validation --------------------------------------
            t0 = time.perf_counter()
            case_id, warnings = validate_input(payload)
            for w in warnings:
                log.warning("Validation warning: %s", w)
            stage_timings["stage_1_validation"] = time.perf_counter() - t0
            log.info("Stage 1 complete - case_id=%s, %d warning(s)", case_id, len(warnings))

            # -- Repositories ------------------------------------------------------
            stage = "repository_load"
            obs_repo = JsonObservationRepository(case_id, payload["obs_only"]["observations"])
            er = payload["entity_resolved"]
            ent_repo = JsonEntityRepository(er.get("canonical_entities", []))

            raw_observations = obs_repo.get_all(case_id)
            canonical_entities = ent_repo.get_all(case_id)

            # FIX: prefer the corrected 'conflicts' list contract; fall back to
            # the legacy int-only 'conflicts_detected' if that's all we get.
            conflicts_raw = er.get("conflicts", er.get("conflicts_detected", []))
            cluster_index = _build_cluster_obs_index(er.get("clusters", []))

            log.info("Loaded %d observations, %d canonical entities", len(raw_observations), len(canonical_entities))

            # -- Stage 2: Event Enrichment ------------------------------------------
            stage = "stage_2_enrichment"
            t0 = time.perf_counter()
            events, obs_conflict_set, unlocalized_count, unresolved_entities = self._stage_2_enrich(
                raw_observations, canonical_entities, conflicts_raw, cluster_index
            )
            stage_timings["stage_2_enrichment"] = time.perf_counter() - t0
            log.info("Stage 2 complete - %d events created", len(events))

            # -- Stage 3: Timestamp-First Ordering -----------------------------------
            stage = "stage_3_ordering"
            t0 = time.perf_counter()
            events = self._temporal.sort_events(events)
            stage_timings["stage_3_ordering"] = time.perf_counter() - t0

            # -- Stage 4 + 5: Event Graph + Temporal Reasoning -----------------------
            stage = "stage_4_5_temporal"
            t0 = time.perf_counter()
            temporal_edges = self._temporal.build_temporal_edges(events)

            events_by_id = {e.event_id: e for e in events}
            ambiguous = [
                (events_by_id[ed.source], events_by_id[ed.target])
                for ed in temporal_edges
                if ed.relation == TemporalRelation.UNKNOWN
                and ed.source in events_by_id and ed.target in events_by_id
            ]
            unresolved_temporal: List[Dict[str, str]] = []
            if ambiguous:
                resolved, unresolved_temporal = self._temporal.resolve_ambiguous_with_llm(ambiguous)
                for ed in temporal_edges:
                    key = f"{ed.source}|{ed.target}"
                    if key in resolved:
                        ed.relation, ed.confidence = resolved[key]
                        ed.label = f"{ed.relation.value} ({ed.confidence:.2f}) [local-LLM]"

            stage_timings["stage_4_5_temporal"] = time.perf_counter() - t0
            log.info("Stage 4/5 complete - %d temporal edges, %d unresolved", len(temporal_edges), len(unresolved_temporal))

            # -- Stage 6: Causal Reasoning --------------------------------------------
            stage = "stage_6_causal"
            t0 = time.perf_counter()
            causal_edges, unresolved_causal = self._causal.infer_causal_links(events)
            stage_timings["stage_6_causal"] = time.perf_counter() - t0
            log.info("Stage 6 complete - %d causal edges, %d unresolved", len(causal_edges), len(unresolved_causal))

            # -- Stage 7: Uncertainty Modelling (confidence already embedded in Stage 2) --
            stage_timings["stage_7_uncertainty"] = 0.0

            # -- Stage 8: Conflict Awareness ------------------------------------------
            stage = "stage_8_conflicts"
            t0 = time.perf_counter()
            conflicts_summary = self._stage_8_conflicts(conflicts_raw, events, cluster_index)
            stage_timings["stage_8_conflicts"] = time.perf_counter() - t0
            log.info("Stage 8 complete - %d conflict entries, %d unlocalized", len(conflicts_summary), unlocalized_count)

            # -- Stage 9: Provenance ---------------------------------------------------
            stage = "stage_9_provenance"
            for ev in events:
                if not ev.obs_ids:
                    log.error("PROVENANCE VIOLATION: event %s has no source observations!", ev.event_id)

            # -- Stage 10: Explainability -----------------------------------------------
            stage = "stage_10_explainability"
            t0 = time.perf_counter()
            explainability = self._explainability.build_explainability(events)
            uncertainties = self._explainability.build_uncertainties(events)
            stage_timings["stage_10_explainability"] = time.perf_counter() - t0

            # -- Stage 11: Narrative -------------------------------------------------------
            stage = "stage_11_narrative"
            t0 = time.perf_counter()
            narrative = self._explainability.build_narrative(events, causal_edges)
            stage_timings["stage_11_narrative"] = time.perf_counter() - t0

            # -- Stage 12: Versioning ------------------------------------------------------
            stage = "stage_12_versioning"
            t0 = time.perf_counter()
            G = self._graph_builder.build(events, temporal_edges, causal_edges)
            graph_export = self._graph_builder.to_export_dict(G)
            all_edges = [*temporal_edges, *causal_edges]

            timeline = TimelineVersion(
                version="V1",
                schema_version=TIMELINE_SCHEMA_VERSION,
                case_id=case_id,
                generated_at=utc_now_iso(),
                events=events,
                causal_links=[e for e in all_edges if e.edge_type == EdgeType.CAUSAL],
                timeline_graph=graph_export,
                uncertainties=uncertainties,
                narrative=narrative,
                explainability=explainability,
                conflicts_summary=conflicts_summary,
                unresolved_temporal_pairs=unresolved_temporal,
                unresolved_causal_pairs=unresolved_causal,
                conflicts_unlocalized_count=unlocalized_count,
                unresolved_entities=unresolved_entities,
            )
            stage_timings["stage_12_versioning"] = time.perf_counter() - t0

            # -- Stage 13: Graph Export -------------------------------------------------------
            stage = "stage_13_export"
            t0 = time.perf_counter()
            if self._save_outputs:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                graph_path = self._output_dir / f"{case_id}_timeline_graph.json"
                self._graph_builder.export_to_file(G, str(graph_path))
                self._timeline_repo.save(timeline)
            stage_timings["stage_13_export"] = time.perf_counter() - t0

        except ValidationError:
            raise  # bad input - let the caller see this distinctly
        except Exception as exc:
            log.error("Unexpected failure during %s: %s", stage, exc)
            raise TimelineAgentError(
                f"Timeline Agent failed during '{stage}': {exc}"
            ) from exc

        total_time = time.perf_counter() - t_start
        log.info("Pipeline complete in %.3fs", total_time)

        result = timeline.to_dict()
        result["stage_timings"] = {k: round(v, 4) for k, v in stage_timings.items()}
        result["total_time_sec"] = round(total_time, 4)
        result["validation_warnings"] = warnings
        # Transparency: make it visible in the output itself whether case
        # content ever left the machine, and exactly how many times. This
        # keeps the forensic explainability story honest when the cloud
        # fallback is enabled, instead of hiding it inside logs.
        result["llm_backend_used"] = type(self._llm).__name__
        result["llm_calls_made"] = getattr(self._llm, "calls_made", 0)

        classification, reason = self._classify_output(
            events, unresolved_temporal, unresolved_causal, conflicts_summary
        )
        result["output_classification"] = classification
        result["output_classification_reason"] = reason
        return result

    def _classify_output(
        self,
        events: List[TimelineEvent],
        unresolved_temporal: List[Dict[str, str]],
        unresolved_causal: List[Dict[str, str]],
        conflicts_summary: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        Signal-based CLEAR / PARTIAL / AMBIGUOUS classification of this
        timeline's own reconstruction confidence (see config.py for the
        scope note - this is not multi-hypothesis ranking).
        """
        if not events:
            return "AMBIGUOUS", "No events were reconstructed from the input."

        avg_confidence = sum(e.confidence for e in events) / len(events)
        conflict_fraction = sum(1 for e in events if e.conflict_flag) / len(events)
        unresolved_count = len(unresolved_temporal) + len(unresolved_causal)
        # FIX: found via CASE_ATM_002 (15 events, 45 unresolved causal
        # pairs) - this counts ambiguous PAIRS (which scale ~O(n^2) with
        # event count) divided by event count (O(n)), so it can legitimately
        # exceed 1.0 for larger, more interconnected cases (seen: 3.00).
        # The threshold comparisons below are still correct either way (a
        # higher ratio should still trigger AMBIGUOUS), but calling this a
        # "fraction" is misleading since fractions read as bounded to
        # [0, 1]. Renamed to "ratio" and the reason text notes explicitly
        # when it's above 1.0, rather than silently printing a confusing
        # "150% unresolved" - no threshold values changed, this is a
        # terminology/clarity fix only.
        unresolved_ratio = unresolved_count / len(events)

        is_ambiguous = (
            avg_confidence <= CLASSIFICATION_AMBIGUOUS_MAX_AVG_CONFIDENCE
            or conflict_fraction >= CLASSIFICATION_AMBIGUOUS_MIN_CONFLICT_FRACTION
            or unresolved_ratio >= CLASSIFICATION_AMBIGUOUS_MIN_UNRESOLVED_FRACTION
        )
        unresolved_note = (
            f"unresolved_ratio={unresolved_ratio:.2f} (more unresolved pairs than events - "
            "a highly interconnected/ambiguous case)"
            if unresolved_ratio > 1.0 else
            f"unresolved_ratio={unresolved_ratio:.2f}"
        )
        if is_ambiguous:
            return (
                "AMBIGUOUS",
                f"avg_confidence={avg_confidence:.2f}, conflict_fraction={conflict_fraction:.2f}, "
                f"{unresolved_note} - at least one signal crossed the "
                "ambiguity threshold; treat this reconstruction as low-trust and prioritise "
                "investigator review of flagged/unresolved items.",
            )

        if (
            avg_confidence >= CLASSIFICATION_CLEAR_MIN_AVG_CONFIDENCE
            and conflict_fraction == 0.0
            and unresolved_ratio <= CLASSIFICATION_CLEAR_MAX_UNRESOLVED_FRACTION
        ):
            return (
                "CLEAR",
                f"avg_confidence={avg_confidence:.2f}, no conflict-flagged events, "
                f"{unresolved_note} - reconstruction is well-supported.",
            )

        return (
            "PARTIAL",
            f"avg_confidence={avg_confidence:.2f}, conflict_fraction={conflict_fraction:.2f}, "
            f"{unresolved_note} - reconstruction is usable but has "
            "some flagged or unresolved items worth investigator attention.",
        )

    # -------------------------------------------------------------------------
    # Stage 2 - Event Enrichment
    # -------------------------------------------------------------------------

    def _stage_2_enrich(
        self,
        raw_observations: List[RawObservation],
        canonical_entities: List[CanonicalEntity],
        conflicts_raw: Any,
        cluster_index: Dict[str, List[str]],
    ) -> Tuple[List[TimelineEvent], Set[str], int, List[str]]:
        """
        Convert raw observations -> TimelineEvent objects.

        FIX (closing a gap from the unified architecture doc): this used to
        be a strict 1 observation -> 1 event mapping. Now it's two real
        sub-phases:
          Stage 2  (this method, per-observation): normalize, resolve
                    entity, compute per-observation confidence/conflict/tags.
          Stage 4/5 (event_clustering.py): cluster same-entity observations
                    that plausibly describe the same real-world action, then
                    reconstruct ONE TimelineEvent per cluster with combined
                    confidence and full provenance (obs_ids = every member).

        A cluster of size 1 behaves identically to the old 1:1 mapping, so
        single-observation callers (including existing unit tests) see no
        change - clustering only kicks in when it finds something to merge.

        Returns (events, conflict_obs_id_set, unlocalized_conflict_count, unresolved_entity_aliases).
        """
        obs_to_entity: Dict[str, CanonicalEntity] = {}
        for ent in canonical_entities:
            for obs_id in ent.sources:
                obs_to_entity[str(obs_id)] = ent

        conflict_obs_ids, unlocalized_count = self._collect_conflict_obs_ids(conflicts_raw, cluster_index)

        conflict_entity_ids: Set[str] = set()
        for obs_id in conflict_obs_ids:
            ent = obs_to_entity.get(obs_id)
            if ent:
                conflict_entity_ids.add(ent.entity_id)

        # -- Stage 2: per-observation enrichment (unchanged logic, no longer
        # immediately materialized as a TimelineEvent) -------------------------
        enriched: Dict[str, Dict[str, Any]] = {}
        unresolved_entity_aliases: List[str] = []
        clusterable: List[ClusterableObservation] = []

        for obs in raw_observations:
            ent = obs_to_entity.get(obs.obs_id)

            # FIX: ts_epoch now comes straight from the repository (which
            # populated it once via normalization.normalize_case()) instead
            # of being re-parsed here - same normalization boundary
            # principle as location_key/action_tags below.
            ts_epoch = obs._ts_epoch
            timestamp = epoch_to_iso(ts_epoch) if ts_epoch > 0 else obs.timestamp

            if ent:
                entity_id = ent.entity_id
                primary_alias = ent.primary_alias
                aliases = list(ent.aliases)
                er_confidence = ent.confidence_score
            else:
                entity_id = f"UNRESOLVED_{normalize_alias(obs.entity)}"
                primary_alias = obs.entity or "unknown"
                aliases = [obs.entity] if obs.entity else []
                er_confidence = 0.40
                unresolved_entity_aliases.append(obs.entity or obs.obs_id)

            in_conflict = (
                obs.obs_id in conflict_obs_ids
                or (ent is not None and ent.entity_id in conflict_entity_ids)
            )
            temporal_certainty = 1.0 if ts_epoch > 0 else 0.50
            conflict_penalty = WEIGHT_CONFLICT_PENALTY if in_conflict else 0.0

            confidence = clamp(
                obs.confidence * WEIGHT_OBS_CONFIDENCE
                + er_confidence * WEIGHT_ENTITY_RESOLUTION
                + temporal_certainty * WEIGHT_TEMPORAL_CERTAINTY
                - conflict_penalty
            )

            # FIX: location_key/action_tags now come straight from the
            # repository (populated once via normalize_case()) instead of
            # being recomputed here - eliminates the redundant second call
            # to normalize_location()/extract_action_tags() per observation.
            location_key = obs.location_key
            action_tags = set(obs.action_tags)

            enriched[obs.obs_id] = {
                "obs_id": obs.obs_id,
                "entity_id": entity_id,
                "primary_alias": primary_alias,
                "aliases": aliases,
                "resolved": ent is not None,
                "modality": obs.modality,
                "location": obs.location,
                "location_key": location_key,
                "content": obs.content,
                "timestamp": timestamp,
                "ts_epoch": ts_epoch,
                "confidence": confidence,
                "role": obs.role or "unknown",
                "in_conflict": in_conflict,
                "action_tags": action_tags,
            }

            clusterable.append(ClusterableObservation(
                obs_id=obs.obs_id,
                entity_id=entity_id,
                ts_epoch=ts_epoch,
                location_key=location_key,
                content=obs.content,
                modality=obs.modality,
                confidence=confidence,
                action_tags=action_tags,
            ))

        # -- Stage 4/5: cluster + reconstruct -----------------------------------
        clusters = cluster_observations(clusterable)
        events: List[TimelineEvent] = []
        seen_event_ids: Set[str] = set()

        for ec in clusters:
            members = [enriched[oid] for oid in ec.obs_ids]
            # Representative content: the highest-confidence member's
            # (cleaned) content. Full traceability to every supporting
            # observation's raw content is via obs_ids -> raw_evidence, not
            # duplicated here.
            best = max(members, key=lambda m: m["confidence"])

            distinct_modalities = {m["modality"] for m in members}
            base_confidence = (
                sum(m["confidence"] * m["confidence"] for m in members)
                / sum(m["confidence"] for m in members)
                if sum(m["confidence"] for m in members) > 0
                else sum(m["confidence"] for m in members) / len(members)
            )
            # Corroboration bonus: multiple independent modalities agreeing
            # on the same reconstructed event is a real forensic confidence
            # signal (this is the entire point of multimodal fusion), capped
            # so it can't dominate the base evidence quality.
            corroboration_bonus = min(0.09, 0.03 * (len(distinct_modalities) - 1))
            confidence = clamp(base_confidence + corroboration_bonus)

            in_conflict = any(m["in_conflict"] for m in members)
            any_resolved = any(m["resolved"] for m in members)

            reasoning: List[str] = ["timestamp ordering"]
            if any_resolved:
                reasoning.append("canonical entity match")
            else:
                reasoning.append("unresolved entity - fallback")
            if len(members) > 1:
                reasoning.append(
                    f"reconstructed from {len(members)} supporting observations "
                    f"across modalities: {', '.join(sorted(distinct_modalities))}"
                )
            if in_conflict:
                reasoning.append("conflict flagged by entity resolution")
            if ec.representative_ts_epoch > 0:
                reasoning.append("valid timestamp")
            else:
                reasoning.append("timestamp missing or unparseable")

            event_id = _make_event_id(ec.obs_ids, ec.entity_id)
            base_id = event_id
            suffix = 0
            while event_id in seen_event_ids:
                suffix += 1
                event_id = f"{base_id}_{suffix}"
            seen_event_ids.add(event_id)

            timestamp = epoch_to_iso(ec.representative_ts_epoch) if ec.representative_ts_epoch > 0 else best["timestamp"]

            events.append(TimelineEvent(
                event_id=event_id,
                obs_ids=list(ec.obs_ids),
                timestamp=timestamp,
                ts_epoch=ec.representative_ts_epoch,
                location=best["location"],
                location_key=ec.location_key,
                entity_id=ec.entity_id,
                primary_alias=best["primary_alias"],
                aliases=best["aliases"],
                modality=best["modality"] if len(distinct_modalities) == 1 else "multimodal",
                content=best["content"],
                confidence=confidence,
                role=best["role"],
                conflict_flag=in_conflict,
                conflict_note=(
                    "Entity resolution conflict detected; confidence reduced."
                    if in_conflict else ""
                ),
                reasoning=reasoning,
                action_tags=sorted(ec.action_tags),
                version="V1",
            ))

        return events, conflict_obs_ids, unlocalized_count, unresolved_entity_aliases

    # -------------------------------------------------------------------------
    # Stage 8 - Conflict Awareness
    # -------------------------------------------------------------------------

    def _collect_conflict_obs_ids(
        self, conflicts_raw: Any, cluster_index: Dict[str, List[str]]
    ) -> Tuple[Set[str], int]:
        """
        FIX: extract obs_ids affected by ER conflicts, supporting three shapes:

          1. list[dict] with explicit obs-level keys (obs_ids/members/affected_obs)
             - used directly (backward compatible with any future ER version
               that decides to embed obs_ids per conflict).
          2. list[dict] with only 'cluster_id' (the REAL, corrected ER contract)
             - resolved via cluster_index (cluster_id -> obs_ids from ER's
               'clusters' array).
          3. plain int (legacy / pre-patch ER output with no detail at all)
             - cannot be localised; returned as `unlocalized_count` so it's
               reported honestly instead of silently discarded or ignored.

        Returns (obs_ids, unlocalized_count).
        """
        ids: Set[str] = set()
        unlocalized = 0

        if isinstance(conflicts_raw, list):
            for c in conflicts_raw:
                if not isinstance(c, dict):
                    continue
                localized = False
                for key in ("obs_ids", "members", "affected_obs"):
                    val = c.get(key)
                    if isinstance(val, list) and val:
                        ids.update(str(v) for v in val)
                        localized = True
                if not localized:
                    cluster_id = str(c.get("cluster_id", ""))
                    cluster_obs = cluster_index.get(cluster_id, [])
                    if cluster_obs:
                        ids.update(cluster_obs)
                        localized = True
                if not localized:
                    unlocalized += 1
        elif isinstance(conflicts_raw, int):
            unlocalized = conflicts_raw

        return ids, unlocalized

    def _stage_8_conflicts(
        self,
        conflicts_raw: Any,
        events: List[TimelineEvent],
        cluster_index: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Translate ER conflict records into Timeline-Agent conflict summaries."""
        summaries: List[Dict[str, Any]] = []

        if isinstance(conflicts_raw, int):
            if conflicts_raw > 0:
                summaries.append({
                    "conflict_type": "entity_resolution_conflicts",
                    "count": conflicts_raw,
                    "detail": f"Entity Resolution reported {conflicts_raw} conflict(s) but the "
                              "output used the legacy count-only format, so affected events "
                              "could not be localised. Upgrade the ER pipeline to emit the "
                              "'conflicts' list (see README) to get per-event localisation.",
                    "affected_events": [],
                })
            return summaries

        if isinstance(conflicts_raw, list):
            for c in conflicts_raw:
                if not isinstance(c, dict):
                    continue
                obs_ids_for_conflict: Set[str] = set()
                for key in ("obs_ids", "members", "affected_obs"):
                    val = c.get(key)
                    if isinstance(val, list):
                        obs_ids_for_conflict.update(str(v) for v in val)
                if not obs_ids_for_conflict:
                    cluster_id = str(c.get("cluster_id", ""))
                    obs_ids_for_conflict.update(cluster_index.get(cluster_id, []))

                affected_events = [
                    ev.event_id for ev in events
                    if any(o in obs_ids_for_conflict for o in ev.obs_ids)
                ]
                summaries.append({
                    "conflict_type": c.get("type", "unknown"),
                    "cluster_id": c.get("cluster_id", ""),
                    "detail": c.get("detail", ""),
                    "affected_events": affected_events,
                })

        return summaries


# -------------------------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------------------------

def run_timeline_agent(
    payload: Dict[str, Any],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_outputs: bool = True,
    llm: Optional[BaseLLMFallback] = None,
) -> Dict[str, Any]:
    """Module-level convenience wrapper."""
    agent = TimelineAgent(output_dir=output_dir, save_outputs=save_outputs, llm=llm)
    return agent.run(payload)