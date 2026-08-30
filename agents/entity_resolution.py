from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_agents_dir = str(_Path(__file__).parent)
if _agents_dir not in _sys.path:
    _sys.path.insert(0, _agents_dir)

"""
ForenSynth – entity_resolution.py
Entity Resolution pipeline: blocking, scoring, clustering, labelling.
"""

# ── stdlib ───────────────────────────────────────────────────────────────
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# ── third-party ──────────────────────────────────────────────────────────
import networkx as nx

# ── shared ForenSynth utilities ───────────────────────────────────────────
from shared import (
    NormalizedObservation, normalize_case, get_normalized_observation_store,
    set_normalized_observation_store, get_semantic_scorer, semantic_location_similarity,
    normalize_alias, normalize_modality, normalize_role, clean_content,
    extract_action_tags, parse_timestamp, epoch_to_iso,
)

try:
    from rapidfuzz import fuzz  # type: ignore
except ImportError:
    fuzz = None  # type: ignore

try:
    from groq import Groq as _GroqClient  # type: ignore
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClient = None  # type: ignore
    _GROQ_AVAILABLE = False

log = logging.getLogger("forensynth.entity_resolution")

ACTION_TOKENS: Set[str] = {
    "enter", "exit", "withdraw", "deposit", "call", "message",
    "send", "receive", "transfer", "arrive", "depart", "access",
    "attempt", "fail", "succeed", "heading", "headed", "inside", "clear",
}

CONFLICTING_ACTION_PAIRS: List[Tuple[str, str]] = [
    ("enter", "exit"),
    ("withdraw", "deposit"),
    ("arrive", "depart"),
    ("send", "receive"),
    ("succeed", "fail"),
    ("heading", "exited"),
]

WEIGHT_MAP: Dict[str, float] = {
    "entity_coreference":  0.280,
    "mention_consistency": 0.120,
    "temporal":            0.150,
    "location":            0.100,
    "context":             0.180,
    "lexical":             0.050,
    "interaction":         0.070,
    "modality":            0.050,
}
assert abs(sum(WEIGHT_MAP.values()) - 1.0) < 1e-9

CONFIRMED_THRESHOLD:      float = 0.80
CANDIDATE_THRESHOLD_HIGH: float = 0.65
CANDIDATE_THRESHOLD_LOW:  float = 0.50
ATTACHMENT_THRESHOLD:     float = 0.70
CLUSTER_CONFIDENCE_FLOOR: float = 0.55
CROSS_MODAL_MERGE_MIN:    float = 0.58
MERGE_COMPOSITE_MIN:      float = 0.55

# FIX: when LLM and embeddings are unavailable, entity_coreference heuristic
# returns ~0.78 for almost any cross-modal same-role pair near the same
# location — it cannot distinguish different suspects. In heuristic-only
# mode, raise the cross-alias merge bar to near-certainty (0.88) so a
# generic estimate alone never merges two distinct suspects.
# Only real LLM/embedding scores (which discriminate content semantics)
# should be trusted at the standard 0.58 threshold.
CROSS_MODAL_MERGE_MIN_HEURISTIC: float = 0.88

OVERSIZED_CLUSTER_FACTOR: float = 3.0
TEMPORAL_WINDOW_SEC:      int   = 300
MAX_TEMPORAL_GAP_SEC:     int   = 3600
MAX_PAIRS:                int   = 500
# FIX: reduced from 40. The chunk that failed needed 6586 tokens against
# a 6000 TPM free-tier cap - i.e. the old size sat right at the edge of
# the limit by design, not as an edge case. 20 pairs/chunk estimates to
# ~3400 tokens (~57% of the cap), giving real margin instead of routinely
# grazing the ceiling.
LLM_BATCH_CHUNK_SIZE:     int   = 5   # FIX: was 20 — at ~300 tokens/pair that hit the 6000 token/min rate limit
                                    # 10 pairs * 300 tokens = 3000 tokens, safe margin

# FIX: previously each scoring agent chunked ALL candidate pairs at
# 40-per-call with NO ceiling on total API calls - call count scaled
# unboundedly with case size. MAX_LLM_CALLS_PER_RUN is a SHARED budget
# across BOTH agents within one resolve_entities() call (see LLMCallBudget
# below). Once exhausted, remaining chunks silently use the heuristic/
# fuzzy fallback - never blocks, never errors.
# FIX: reduced from 6 to 2 - a hard "at most 1-2 API calls per case
# file" ceiling, shared across BOTH scoring agents combined (not 2 each).
# In practice this typically means the context-scoring agent (which runs
# first in stage 4) gets one real LLM call and the entity-coreference
# agent gets the other; anything beyond that falls back to the heuristic
# scorer. This intentionally trades "most pairs get a real LLM score" for
# "usage is small and predictable" - see the accuracy note below for why
# this doesn't meaningfully cost resolution quality here.
MAX_LLM_CALLS_PER_RUN: int = 14  # FIX: 131 pairs / 20 per batch = 7 calls per agent needed to score all pairs
                                   # was 2 — only first 20 pairs got real LLM scores, rest fell back to heuristic

GROQ_MODEL: str = "openai/gpt-oss-20b"

FEATURE_NAMES: Tuple[str, ...] = (
    "entity_coreference", "mention_consistency", "temporal", "location",
    "context", "lexical", "interaction", "modality",
)

# -- Output classification (CLEAR / PARTIAL / AMBIGUOUS) -----------------------
# Mirrors the Timeline Agent notebook's equivalent classification for
# consistency across both agents.
CLASSIFICATION_CLEAR_MIN_AVG_CONFIDENCE: float = 0.75
CLASSIFICATION_CLEAR_MAX_CONFLICT_FRACTION: float = 0.0
CLASSIFICATION_AMBIGUOUS_MAX_AVG_CONFIDENCE: float = 0.55
CLASSIFICATION_AMBIGUOUS_MIN_CONFLICT_FRACTION: float = 0.30
CLASSIFICATION_AMBIGUOUS_MIN_LOW_CONFIDENCE_FRACTION: float = 0.50

# FIX: constrained re-splitting threshold. When role_count_mismatch fires,
# Stage 10b attempts single-linkage splitting (max spanning tree, cut
# weakest bridging edges). A cut only ever happens if EVERY edge being cut
# scores below this - i.e. there must be a genuinely weak bridge, not just
# the least-strong of several confidently strong bonds. Stricter than
# MERGE_COMPOSITE_MIN on purpose: merging is permissive, splitting is
# destructive.
SPLIT_CUT_MAX_WEIGHT: float = 0.60


@dataclass
class Observation:
    obs_id:     str
    entity:     str
    role:       str
    modality:   str
    location:   str
    content:    str
    timestamp:  str
    confidence: float
    entity_norm:     str   = ""
    time_offset_sec: int   = 0
    _ts_epoch:       float = field(default=0.0, repr=False)


@dataclass
class HumanConstraints:
    must_merge:     List[Tuple[str, str]]        = field(default_factory=list)
    must_not_merge: List[Tuple[str, str]]        = field(default_factory=list)
    soft_hints:     Dict[Tuple[str, str], float] = field(default_factory=dict)


@dataclass
class PairFeatures:
    obs_a: Observation
    obs_b: Observation
    entity_coreference:  float = 0.0
    mention_consistency: float = 0.0
    temporal:            float = 0.0
    location:            float = 0.0
    context:             float = 0.0
    lexical:             float = 0.0
    interaction:         float = 0.0
    modality:            float = 0.0
    composite:           float = 0.0
    reasons:             List[str] = field(default_factory=list)
    hard_negative:       bool = False

    def compute_composite(self, soft_hint: float = 0.0) -> float:
        raw = sum(getattr(self, name) * WEIGHT_MAP[name] for name in FEATURE_NAMES)
        self.composite = min(1.0, raw + soft_hint)
        return self.composite


@dataclass
class EdgeRecord:
    alias_1:        str
    alias_2:        str
    obs_id_1:       str
    obs_id_2:       str
    weight:         float
    support:        int = 1
    classification: str = "rejected"
    features:       Optional[PairFeatures] = None
    reasons:        List[str] = field(default_factory=list)
    hard_negative:  bool = False


# NOTE: normalize_alias() is now defined once in the shared normalization
# cell above (## 2b) and used here and throughout the pipeline - no longer
# duplicated in this cell.


def _parse_llm_json_map(text: str) -> Dict[str, float]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    return {str(k): float(v) for k, v in parsed.items() if re.match(r"^\d+$", str(k))}


def _mention_payload(obs: Observation) -> Dict[str, str]:
    return {
        "entity":    obs.entity,
        "role":      obs.role,
        "modality":  obs.modality,
        "location":  obs.location[:120],
        "content":   obs.content[:220],
        "timestamp": obs.timestamp,
    }


class LLMCallBudget:
    """
    Shared, mutable call budget across BOTH scoring agents in one pipeline
    run. Enforces a hard ceiling on total LLM API calls regardless of case
    size - once exhausted, remaining chunks silently use the heuristic
    fallback. Instantiate ONE of these per resolve_entities() call and pass
    it to both ContextScoringAgent and EntityCoreferenceAgent.
    """
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls_made = 0

    def try_consume(self) -> bool:
        if self.calls_made >= self.max_calls:
            return False
        self.calls_made += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.calls_made)

    @property
    def exhausted(self) -> bool:
        return self.calls_made >= self.max_calls


class _BatchLLMScorer:
    """Shared Groq batch caller with chunking, a shared call budget, and local fallback."""

    def __init__(self, model: str, enabled: bool, budget: Optional["LLMCallBudget"] = None):
        self.model = model
        self.enabled = enabled and _GROQ_AVAILABLE
        self._client: Any = None
        self._budget = budget
        if self.enabled:
            api_key = os.environ.get("GROQ_API_KEY", "")
            if api_key:
                self._client = _GroqClient(api_key=api_key)
            else:
                log.warning("GROQ_API_KEY not set – using heuristic/fuzzy fallbacks.")
                self.enabled = False

    def score_batch(
        self,
        payload_items: List[Dict[str, Any]],
        system_prompt: str,
        user_intro: str,
        fallback_fn,
    ) -> List[float]:
        """Return one score per payload item (index-aligned)."""
        n = len(payload_items)
        if n == 0:
            return []

        scores: List[Optional[float]] = [None] * n
        if not self.enabled or self._client is None:
            return [float(fallback_fn(i)) for i in range(n)]

        for chunk_start in range(0, n, LLM_BATCH_CHUNK_SIZE):
            chunk = payload_items[chunk_start : chunk_start + LLM_BATCH_CHUNK_SIZE]

            # FIX: hard budget check before spending an API call. Once the
            # shared budget is exhausted, every remaining chunk (across BOTH
            # agents, for the rest of this pipeline run) uses the fallback
            # scorer instead.
            if self._budget is not None and not self._budget.try_consume():
                log.info("LLM call budget exhausted (%d/%d used) - falling back to heuristic scoring for remaining %d pair(s).",
                          self._budget.calls_made, self._budget.max_calls, len(chunk))
                for local_i in range(len(chunk)):
                    scores[chunk_start + local_i] = float(fallback_fn(chunk_start + local_i))
                continue

            # FIX: add delay between calls to respect Groq free tier rate limit
            # (6000 tokens/min). Without delay, 3 rapid calls exhaust the limit
            # in ~2 seconds, causing 429 errors on subsequent calls.
            if chunk_start > 0:
                import time as _time
                _time.sleep(12)  # 12s gap = ~5 calls/min, well under 6000 token limit

            prompt = (
                f"{system_prompt}\n\n{user_intro}\n"
                + json.dumps(chunk, ensure_ascii=False)
                + '\n\nReturn ONLY JSON: {"0": 0.85, "1": 0.12, ...} with one float 0.0-1.0 per id.'
            )
            # FIX: was len(chunk)*16+128 which gave only 288 tokens for 10 pairs
            # — not enough for the JSON response, causing truncation and parse errors.
            # 10 pairs need ~200 output tokens; 300 gives safe headroom.
            max_tok = min(4096, max(300, len(chunk) * 30))
            raw_scores: Dict[str, float] = {}
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tok,
                    temperature=0.0,
                )
                raw_scores = _parse_llm_json_map(resp.choices[0].message.content or "")
            except Exception as exc:
                log.warning("LLM batch chunk failed (%s); using fallback for %d pairs.", exc, len(chunk))

            for local_i in range(len(chunk)):
                global_i = chunk_start + local_i
                if str(local_i) in raw_scores:
                    scores[global_i] = max(0.0, min(1.0, raw_scores[str(local_i)]))
                else:
                    scores[global_i] = float(fallback_fn(global_i))

        return [s if s is not None else float(fallback_fn(i)) for i, s in enumerate(scores)]


class ContextScoringAgent:
  def __init__(self, model: str = GROQ_MODEL, enabled: bool = True, budget: Optional["LLMCallBudget"] = None):
    self._llm = _BatchLLMScorer(model, enabled, budget=budget)
    self._cache: Dict[Tuple[str, str], float] = {}

  @staticmethod
  def _cache_key(a: str, b: str) -> Tuple[str, str]:
    ka, kb = a[:120], b[:120]
    return (ka, kb) if ka <= kb else (kb, ka)

  def _detect_conflicting_actions(self, text_a: str, text_b: str) -> float:
    def _tokens(t: str) -> Set[str]:
      return ACTION_TOKENS & set(re.findall(r"\b\w+\b", t.lower()))
    ta, tb = _tokens(text_a), _tokens(text_b)
    for act_a, act_b in CONFLICTING_ACTION_PAIRS:
      if (act_a in ta and act_b in tb) or (act_b in ta and act_a in tb):
        return 0.5
    return 1.0

  def _content_fallback(self, pair: Tuple[str, str]) -> float:
    ca, cb = pair
    penalty = self._detect_conflicting_actions(ca, cb)
    return min(1.0, max(0.0, fuzz.token_set_ratio(ca, cb) / 100.0 * penalty))

  def precompute_batch_scores(
    self,
    candidate_pairs: List[Tuple[str, str]],
    obs_by_id: Dict[str, Observation],
  ) -> None:
    unique: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for id_a, id_b in candidate_pairs:
      oa, ob = obs_by_id.get(id_a), obs_by_id.get(id_b)
      if not oa or not ob:
        continue
      ck = self._cache_key(oa.content, ob.content)
      if ck in self._cache or ck in seen:
        continue
      seen.add(ck)
      unique.append(ck)

    if not unique:
      return

    # FIX: use heuristic fallback for ALL context pairs.
    # Context scoring (text similarity) is not the primary merge signal —
    # entity coreference is. Sending 100+ pairs to the LLM for context
    # scoring burns the token budget before entity coreference gets its turn,
    # causing 429 rate limit errors on the more important entity calls.
    # Heuristic lexical similarity is sufficient for context weighting.
    for ck in unique:
      self._cache[ck] = self._content_fallback(ck)
    log.info("Context precompute: %d unique pairs, all decided by heuristic (LLM reserved for entity coreference).",
             len(unique))
    return

    unique = unique  # unreachable — kept for diff clarity
    system = (
      "You are a forensic semantic analyst. Score how likely two observation texts "
      "describe the same underlying event or the same actor's actions."
    )
    intro = "Rate each text pair from 0.0 (unrelated) to 1.0 (same event/entity)."
    payload = [
      {"id": str(i), "text_a": a[:300], "text_b": b[:300]}
      for i, (a, b) in enumerate(unique)
    ]

    def fallback(i: int) -> float:
      a, b = unique[i]
      return self._content_fallback((a, b))

    scored = self._llm.score_batch(payload, system, intro, fallback)
    for pair_key, raw, (a, b) in zip(unique, scored, unique):
      penalty = self._detect_conflicting_actions(a, b)
      self._cache[pair_key] = min(1.0, max(0.0, raw * penalty))

  def score(self, content_a: str, content_b: str) -> float:
    ck = self._cache_key(content_a, content_b)
    if ck not in self._cache:
      self._cache[ck] = self._content_fallback(ck)
    return self._cache[ck]


class EntityCoreferenceAgent:
  """LLM scores whether two forensic mentions refer to the same real-world entity."""

  def __init__(self, model: str = GROQ_MODEL, enabled: bool = True, budget: Optional["LLMCallBudget"] = None):
    self._llm = _BatchLLMScorer(model, enabled, budget=budget)
    self._cache: Dict[Tuple[str, str], float] = {}

  @staticmethod
  def _pair_key(id_a: str, id_b: str) -> Tuple[str, str]:
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)

  @staticmethod
  def heuristic_coreference(oa: Observation, ob: Observation, temporal_window: int, max_gap: int) -> float:
    if oa.entity_norm == ob.entity_norm:
      return 1.0
    if oa.role.strip().lower() != ob.role.strip().lower():
      return 0.12

    dt = abs(oa.time_offset_sec - ob.time_offset_sec)
    if dt > max_gap:
      return 0.05

    cross_modal = oa.modality.lower() != ob.modality.lower()
    score = 0.45 if cross_modal else 0.32

    if dt <= temporal_window:
      score += 0.25
    elif dt <= max_gap:
      score += 0.10

    la, lb = oa.location.strip().lower(), ob.location.strip().lower()
    if la and lb:
      loc_sim = fuzz.token_set_ratio(la, lb) / 100.0
      if loc_sim >= 0.45:
        score += 0.10
      shared_tokens = set(re.findall(r"[a-z0-9]+", la)) & set(re.findall(r"[a-z0-9]+", lb))
      if shared_tokens & {"atm", "server", "booth", "room", "network", "ssh", "login"}:
        score += 0.08

    content_sim = fuzz.token_set_ratio(oa.content, ob.content) / 100.0
    if content_sim >= 0.35:
      score += 0.08

    topical = {"enter", "entered", "inside", "heading", "headed", "starting", "exit", "exited",
               "coming", "leaving", "depart", "server", "ssh", "login", "access", "atm", "booth", "clear"}
    ca = set(re.findall(r"[a-z0-9]+", oa.content.lower()))
    cb = set(re.findall(r"[a-z0-9]+", ob.content.lower()))
    overlap = ca & cb & topical
    if overlap:
      score += min(0.14, 0.05 * len(overlap))

    enter_tokens = {"enter", "entered", "inside", "heading", "headed", "starting"}
    exit_tokens = {"exit", "exited", "coming", "leaving", "depart"}
    a_enter = bool(ca & enter_tokens)
    b_enter = bool(cb & enter_tokens)
    a_exit = bool(ca & exit_tokens)
    b_exit = bool(cb & exit_tokens)
    if (a_enter and b_enter) or (a_exit and b_exit):
      score += 0.10

    # Narrative phase bucketing: approach / inside / exit within one event window
    def _phase(tokens: Set[str]) -> str:
      if tokens & exit_tokens:
        return "exit"
      if tokens & enter_tokens:
        return "inside"
      if tokens & {"heading", "headed", "towards", "see", "activity"}:
        return "approach"
      return "other"

    if dt <= temporal_window and _phase(ca) == _phase(cb) != "other":
      score += 0.08

    return min(1.0, score)

  def precompute_batch_scores(
    self,
    candidate_pairs: List[Tuple[str, str]],
    obs_by_id: Dict[str, Observation],
    temporal_window: int,
    max_gap: int,
  ) -> None:
    unique_keys: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for id_a, id_b in candidate_pairs:
      key = self._pair_key(id_a, id_b)
      if key in self._cache or key in seen:
        continue
      seen.add(key)
      unique_keys.append(key)

    if not unique_keys:
      return

    # FIX: pre-filter pairs using hard rules before sending to LLM.
    # Only genuinely ambiguous cross-modal same-role pairs go to Groq.
    # This reduces 131 pairs to ~10-20, requiring only 1 LLM call.
    obs_pairs_all = [(obs_by_id[a], obs_by_id[b]) for a, b in unique_keys]
    needs_llm_keys: List[Tuple[str, str]] = []
    needs_llm_pairs: List[Tuple] = []

    for key, (oa, ob) in zip(unique_keys, obs_pairs_all):
      # Hard rule 1: same alias = definitely same entity
      if oa.entity_norm == ob.entity_norm:
        self._cache[key] = 1.0
        continue
      # Hard rule 2: different role = definitely different entity
      if oa.role.strip().lower() != ob.role.strip().lower():
        self._cache[key] = 0.05
        continue
      # Hard rule 3: same modality, different alias = hard negative (stage_5 blocks merge)
      if oa.modality.lower() == ob.modality.lower():
        self._cache[key] = 0.10
        continue
      # Genuinely ambiguous: cross-modal, same role, different alias — needs LLM
      needs_llm_keys.append(key)
      needs_llm_pairs.append((oa, ob))

    ambiguous = len(needs_llm_keys)
    decided = len(unique_keys) - ambiguous
    log.info("Entity coreference precompute: %d unique pairs, %d ambiguous (LLM), %d decided by rules.",
             len(unique_keys), ambiguous, decided)

    if not needs_llm_keys:
      return

    # Only score the ambiguous subset with LLM
    unique_keys = needs_llm_keys
    obs_pairs = needs_llm_pairs
    log.info("Entity coreference precompute: %d unique mention pairs.", len(unique_keys))
    payload = [
      {"id": str(i), "mention_a": _mention_payload(oa), "mention_b": _mention_payload(ob)}
      for i, (oa, ob) in enumerate(obs_pairs)
    ]

    system = (
      "You are a digital forensics entity-resolution expert. "
      "Decide if mention A and mention B refer to the SAME real-world person/device/account. "
      "CRITICAL RULES: "
      "(1) Different roles (suspect vs witness) must score below 0.2. "
      "(2) COORDINATING SUSPECTS ARE DIFFERENT PEOPLE: in multi-actor crimes, suspects "
      "communicate with each other ('I am close', 'keep going', 'security nearby'). "
      "Two observations showing DIFFERENT ACTORS COORDINATING must score below 0.3, "
      "even if they are at the same location and time. "
      "Ask: are these the SAME INDIVIDUAL or TWO PEOPLE TALKING TO EACH OTHER? "
      "(3) Same alias string in same modality = same entity (score 0.95+). "
      "(4) Cross-modal same-suspect evidence (e.g. video Person_X + audio Speaker_Y "
      "both describing ONE person's actions) may score high IF content describes "
      "one person's actions, not a conversation between two people. "
      "Use role, modality, location, timestamp, and content together."
    )
    intro = "Score each mention pair from 0.0 (definitely different) to 1.0 (same entity)."

    def fallback(i: int) -> float:
      oa, ob = obs_pairs[i]
      return self.heuristic_coreference(oa, ob, temporal_window, max_gap)

    scored = self._llm.score_batch(payload, system, intro, fallback)
    for i, (key, raw) in enumerate(zip(unique_keys, scored)):
      # FIX (real production regression): the LLM path used to fully
      # REPLACE the heuristic score whenever available, instead of
      # supplementing it. Observed effect on real CASE_ATM_001 data:
      # three cross-modal same-witness pairs heuristic_coreference
      # correctly scored 0.65-0.69 (merged) dropped to 0.42-0.48
      # (rejected) once the LLM was live - the LLM read one
      # observations literal content ("...leave the office" - a
      # wording artifact, not the ATM) and was more conservative than
      # the heuristics role/temporal/topical-keyword logic. Entity
      # count went from the correct 2 to a fragmented 4. Blending via
      # max() means the LLM can still ADD confidence beyond the
      # heuristic, but can never subtract from it.
      heuristic_score = fallback(i)
      self._cache[key] = max(0.0, min(1.0, max(raw, heuristic_score)))

  def score(self, oa: Observation, ob: Observation, temporal_window: int, max_gap: int) -> float:
    key = self._pair_key(oa.obs_id, ob.obs_id)
    if key not in self._cache:
      self._cache[key] = self.heuristic_coreference(oa, ob, temporal_window, max_gap)
    return self._cache[key]


class UnionFind:
  def __init__(self, elements: List[str]):
    self.parent = {e: e for e in elements}
    self.rank = {e: 0 for e in elements}

  def find(self, x: str) -> str:
    while self.parent[x] != x:
      self.parent[x] = self.parent[self.parent[x]]
      x = self.parent[x]
    return x

  def union(self, x: str, y: str) -> bool:
    px, py = self.find(x), self.find(y)
    if px == py:
      return False
    if self.rank[px] < self.rank[py]:
      px, py = py, px
    self.parent[py] = px
    if self.rank[px] == self.rank[py]:
      self.rank[px] += 1
    return True

  def groups(self) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = defaultdict(list)
    for e in self.parent:
      d[self.find(e)].append(e)
    return dict(d)


def classify_resolution_output(canonical_entities, conflicts):
    """
    Signal-based CLEAR / PARTIAL / AMBIGUOUS classification of this
    resolution run's own confidence, mirroring the Timeline Agent
    notebook's equivalent classification for consistency across both
    agents. Honest scope note: classifies THIS run's confidence using
    concrete already-computed signals - not a comparison across
    alternative resolution hypotheses.
    """
    if not canonical_entities:
        return "AMBIGUOUS", "No canonical entities were resolved."

    confidences = [e.get("confidence_score", 0.0) for e in canonical_entities]
    avg_confidence = sum(confidences) / len(confidences)
    low_confidence_fraction = sum(1 for c in confidences if c < CLUSTER_CONFIDENCE_FLOOR) / len(confidences)
    conflict_fraction = len(conflicts) / len(canonical_entities)

    is_ambiguous = (
        avg_confidence <= CLASSIFICATION_AMBIGUOUS_MAX_AVG_CONFIDENCE
        or conflict_fraction >= CLASSIFICATION_AMBIGUOUS_MIN_CONFLICT_FRACTION
        or low_confidence_fraction >= CLASSIFICATION_AMBIGUOUS_MIN_LOW_CONFIDENCE_FRACTION
    )
    if is_ambiguous:
        return (
            "AMBIGUOUS",
            f"avg_confidence={avg_confidence:.2f}, conflict_fraction={conflict_fraction:.2f}, "
            f"low_confidence_fraction={low_confidence_fraction:.2f} - at least one signal crossed "
            "the ambiguity threshold; treat these entity mappings as low-trust and prioritise "
            "investigator review before downstream use.",
        )

    if avg_confidence >= CLASSIFICATION_CLEAR_MIN_AVG_CONFIDENCE and conflict_fraction <= CLASSIFICATION_CLEAR_MAX_CONFLICT_FRACTION:
        return (
            "CLEAR",
            f"avg_confidence={avg_confidence:.2f}, no conflicts detected - entity resolution is well-supported.",
        )

    return (
        "PARTIAL",
        f"avg_confidence={avg_confidence:.2f}, conflict_fraction={conflict_fraction:.2f}, "
        f"low_confidence_fraction={low_confidence_fraction:.2f} - resolution is usable but has "
        "some flagged or low-confidence clusters worth investigator attention.",
    )


class EntityResolutionPipeline:
  def __init__(
    self,
    config: Optional[Dict[str, Any]] = None,
    human_constraints: Optional[HumanConstraints] = None,
    llm_enabled: bool = True,
    max_llm_calls: int = MAX_LLM_CALLS_PER_RUN,
  ):
    cfg = config or {}
    self.check_duplicates = cfg.get("check_duplicates", True)
    self.case_base_time = cfg.get("case_base_time", None)
    self.temporal_window_sec = cfg.get("temporal_window_sec", TEMPORAL_WINDOW_SEC)
    self.max_temporal_gap_sec = cfg.get("max_temporal_gap_sec", MAX_TEMPORAL_GAP_SEC)
    self.max_pairs = cfg.get("max_pairs", MAX_PAIRS)
    self.confirmed_threshold = cfg.get("confirmed_threshold", CONFIRMED_THRESHOLD)
    self.candidate_threshold_low = cfg.get("candidate_threshold_low", CANDIDATE_THRESHOLD_LOW)
    self.candidate_threshold_high = cfg.get("candidate_threshold_high", CANDIDATE_THRESHOLD_HIGH)
    self.llm_enabled = llm_enabled
    # Track whether real LLM is active so Stage 8 can use the conservative
    # merge threshold when only heuristics run.
    self._llm_actually_active = (
        llm_enabled
        and bool(os.environ.get('GROQ_API_KEY', ''))
    )

    if isinstance(human_constraints, dict):
      # Callers (e.g. run_case.py, loading constraints back out of the DB via
      # mem.load_er_constraints()) pass a plain dict, not a HumanConstraints
      # instance -- normalize it here so every attribute access below works
      # regardless of which shape the caller handed in.
      human_constraints = HumanConstraints(
        must_merge=human_constraints.get("must_merge", []),
        must_not_merge=human_constraints.get("must_not_merge", []),
        soft_hints=human_constraints.get("soft_hints", {}),
      )
    self.constraints = human_constraints or HumanConstraints()
    # FIX: previously one shared LLMCallBudget across both agents meant
    # whichever agent ran first (context-scoring) could consume the ENTIRE
    # budget, leaving entity-coreference - arguably the more consequential
    # signal for merge decisions - with zero real LLM calls on anything
    # but the smallest cases. Each agent now gets its own guaranteed
    # share (ceil/floor split of max_llm_calls) instead of competing for
    # one pool, so both signal sources get at least one real LLM-scored
    # chunk whenever max_llm_calls >= 2.
    context_budget_size = (max_llm_calls + 1) // 2
    entity_budget_size = max_llm_calls // 2
    self._context_budget = LLMCallBudget(context_budget_size)
    self._entity_budget = LLMCallBudget(entity_budget_size)
    self.context_agent = ContextScoringAgent(model=GROQ_MODEL, enabled=self.llm_enabled, budget=self._context_budget)
    self.entity_agent = EntityCoreferenceAgent(model=GROQ_MODEL, enabled=self.llm_enabled, budget=self._entity_budget)

    self.case_id = "UNKNOWN_CASE"
    self.observations: List[Observation] = []
    self.base_epoch = 0.0
    self.candidate_pairs: List[Tuple[str, str]] = []
    self.edges: List[EdgeRecord] = []
    self.graph = nx.Graph()
    self.clusters: Dict[str, List[str]] = {}
    self.conflicts: List[Dict[str, Any]] = []
    self.fir_role_counts: Dict[str, int] = {}
    self._resplit_log: List[Dict[str, Any]] = []
    self.status = "success"
    self.error_message = ""
    self.stage_timings: Dict[str, float] = {}
    self._obs_by_id: Dict[str, Observation] = {}
    self._pair_features: Dict[Tuple[str, str], PairFeatures] = {}
    self._candidate_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    self._canonical_entities: List[Dict[str, Any]] = []

  def run(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
    pipeline_start = time.perf_counter()
    try:
      self._stage(1, self._stage_1_intake, raw_input)
      self._stage(2, self._stage_2_normalization)
      self._stage(3, self._stage_3_blocking)
      self._stage(4, self._stage_4_feature_computation)
      self._stage(5, self._stage_5_scoring)
      self._stage(6, self._stage_6_classification)
      self._stage(7, self._stage_7_graph_construction)
      self._stage(8, self._stage_8_clustering)
      self._stage(9, self._stage_9_guarded_attachment)
      self._stage(10, self._stage_10_conflict_detection)
      self._stage("10b", self._stage_10b_constrained_resplit)
      self._stage(11, self._stage_11_entity_labeling)
    except Exception as exc:
      log.exception("Pipeline failed: %s", exc)
      self.status = "failed"
      self.error_message = str(exc)
    return self._stage_12_package(time.perf_counter() - pipeline_start)

  def _stage(self, n: Any, fn: Any, *args: Any) -> None:
    t0 = time.perf_counter()
    fn(*args)
    suffix = fn.__name__.split("_stage_")[1]
    self.stage_timings[f"stage_{suffix}"] = time.perf_counter() - t0
    log.info("Stage %s (%s) done in %.4fs", n, fn.__name__, self.stage_timings[f"stage_{suffix}"])

  @staticmethod
  def _canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return (min(a, b), max(a, b))

  @staticmethod
  def _parse_ts(ts_str: str) -> float:
    # FIX: delegates to the shared parse_timestamp() (## 2b) instead of
    # maintaining its own separate 4-format list - this is what previously
    # let ER and the Timeline Agent silently disagree on which timestamps
    # were parseable.
    return parse_timestamp(ts_str)

  def _stage_1_intake(self, raw_input: Dict[str, Any]) -> None:
    self.case_id = str(raw_input.get("case_id", "") or "UNKNOWN_CASE")
    # FIR-declared role counts are legitimate input data (the complaint
    # itself), not derived from ground truth - captured so Stage 10 can
    # flag a mismatch against resolved entity counts.
    self.fir_role_counts: Dict[str, int] = dict(raw_input.get("fir", {}).get("roles", {}) or {})
    raw_obs = raw_input.get("observations", [])

    # FIX: normalization (role/modality canonicalization, content cleaning,
    # timestamp parsing, entity_norm, time_offset_sec) now goes through the
    # single shared batch entry point (normalize_case(), via the store - see
    # ## 2b) instead of being computed inline here AND again in
    # _stage_2_normalization. This is the same seam the Timeline Agent
    # notebook now uses - the swap point for the real Memory Store later:
    # once the team's store is ready, call set_normalized_observation_store(
    # MemoryStoreNormalizedObservationStore(...)) once and this intake code
    # does not need to change.
    store = get_normalized_observation_store()
    try:
      normalized_list = store.get(self.case_id, raw_obs)
    except Exception as exc:
      log.error("Normalization store failed for case '%s' (%s) - falling back to empty observation set.", self.case_id, exc)
      normalized_list = [None] * len(raw_obs)

    seen_hashes: Dict[str, Observation] = {}
    for item, n in zip(raw_obs, normalized_list):
      obs_id = str(item.get("obs_id", "")).strip()
      entity = str(item.get("entity", "")).strip()
      if not obs_id or not entity:
        log.warning("Skipping obs %s: missing obs_id or entity", item.get("obs_id"))
        continue
      if n is not None:
        obs = Observation(
          obs_id=n.obs_id,
          entity=n.entity_raw,
          role=n.role,
          modality=n.modality,
          location=n.location_raw,
          content=n.content_clean,
          timestamp=n.timestamp_raw,
          confidence=n.confidence,
          entity_norm=n.entity_norm,
          time_offset_sec=n.time_offset_sec,
          _ts_epoch=n.ts_epoch,
        )
      else:
        # FIX: normalization failed — build Observation directly from raw dict
        ts_raw = str(item.get("timestamp", ""))
        try:
            from shared import parse_timestamp as _pts
            ts_epoch = _pts(ts_raw)
        except Exception:
            ts_epoch = 0.0
        obs = Observation(
          obs_id=obs_id,
          entity=entity,
          role=normalize_role(str(item.get("role", "unknown"))),
          modality=normalize_modality(str(item.get("modality", "unknown"))),
          location=str(item.get("location", "")),
          content=str(item.get("content", "")),
          timestamp=ts_raw,
          confidence=float(item.get("confidence", 0.5)),
          entity_norm=normalize_alias(entity),
          time_offset_sec=int(item.get("time_offset", 0)),
          _ts_epoch=ts_epoch,
        )
      if self.check_duplicates:
        fp = f"{obs.entity}|{obs.modality}|{obs.location}|{obs.content}|{obs.timestamp}"
        digest = hashlib.sha256(fp.encode()).hexdigest()
        if digest not in seen_hashes or obs.confidence > seen_hashes[digest].confidence:
          seen_hashes[digest] = obs
      else:
        seen_hashes[obs.obs_id] = obs
    self.observations = list(seen_hashes.values())
    self._obs_by_id = {o.obs_id: o for o in self.observations}

  def _stage_2_normalization(self) -> None:
    # FIX: ts_epoch / time_offset_sec / entity_norm are now already
    # populated by normalize_case() in Stage 1 (single normalization
    # boundary, computed once) - this stage is intentionally a no-op, kept
    # only so stage numbering and stage_timings output stay stable.
    pass

  def _build_obs_blocked_set(self) -> Set[Tuple[str, str]]:
    blocked: Set[Tuple[str, str]] = set()
    for a_alias, b_alias in self.constraints.must_not_merge:
      a_norm, b_norm = normalize_alias(a_alias), normalize_alias(b_alias)
      for oa in self.observations:
        for ob in self.observations:
          if oa.obs_id == ob.obs_id:
            continue
          if {oa.entity_norm, ob.entity_norm} == {a_norm, b_norm}:
            blocked.add(self._canonical_pair(oa.obs_id, ob.obs_id))
    return blocked

  def _stage_3_blocking(self) -> None:
    blocked = self._build_obs_blocked_set()
    forced: Set[Tuple[str, str]] = set()
    for a_alias, b_alias in self.constraints.must_merge:
      a_norm, b_norm = normalize_alias(a_alias), normalize_alias(b_alias)
      ids_a = [o.obs_id for o in self.observations if o.entity_norm == a_norm]
      ids_b = [o.obs_id for o in self.observations if o.entity_norm == b_norm]
      for ia, ib in itertools.product(ids_a, ids_b):
        if ia != ib:
          key = self._canonical_pair(ia, ib)
          if key not in blocked:
            forced.add(key)

    candidate_set: Set[Tuple[str, str]] = set()
    obs_list = self.observations
    for i, oa in enumerate(obs_list):
      for ob in obs_list[i + 1 :]:
        key = self._canonical_pair(oa.obs_id, ob.obs_id)
        if key in blocked:
          continue
        dt = abs(oa.time_offset_sec - ob.time_offset_sec)
        same_loc = bool(oa.location.strip()) and oa.location.strip().lower() == ob.location.strip().lower()
        same_role = oa.role.strip().lower() == ob.role.strip().lower()
        cross_modal = oa.modality.lower() != ob.modality.lower()
        if same_loc or dt <= self.temporal_window_sec or (cross_modal and same_role and dt <= self.max_temporal_gap_sec):
          candidate_set.add(key)

    all_candidates = list(forced) + [p for p in candidate_set if p not in forced]
    self.candidate_pairs = all_candidates[: self.max_pairs]

  def _feat_mention_consistency(self, oa: Observation, ob: Observation) -> Tuple[float, bool]:
    if oa.modality == ob.modality:
      if oa.entity_norm == ob.entity_norm:
        dt = abs(oa.time_offset_sec - ob.time_offset_sec)
        loc = oa.location.strip().lower() == ob.location.strip().lower() if oa.location and ob.location else False
        if dt <= self.temporal_window_sec and loc:
          return 0.85, True
        if dt <= self.max_temporal_gap_sec:
          return 0.45, False
        return 0.20, False
      return 0.30, False
    dt = abs(oa.time_offset_sec - ob.time_offset_sec)
    if dt <= self.temporal_window_sec:
      return 0.78, True
    if dt <= self.max_temporal_gap_sec:
      return 0.42, False
    return 0.10, False

  def _feat_temporal(self, oa: Observation, ob: Observation) -> float:
    dt = abs(oa.time_offset_sec - ob.time_offset_sec)
    if dt > self.max_temporal_gap_sec:
      return 0.0
    if dt <= self.temporal_window_sec:
      return max(0.0, 1.0 - dt / self.temporal_window_sec)
    span = self.max_temporal_gap_sec - self.temporal_window_sec
    return max(0.0, 0.3 * (1.0 - (dt - self.temporal_window_sec) / span))

  def _feat_location(self, oa: Observation, ob: Observation) -> float:
    la, lb = oa.location.strip().lower(), ob.location.strip().lower()
    if not la or not lb:
      return 0.35
    if la == lb:
      return 1.0
    # FIX: semantic similarity instead of pure token overlap - see ## 2c.
    # Old approach scored "ATM entrance" vs "ATM exit" as 81% similar
    # (shared words "ATM"/"main"/"door"), despite opposite meaning.
    return semantic_location_similarity(la, lb)

  def _feat_context(self, oa: Observation, ob: Observation) -> float:
    return self.context_agent.score(oa.content, ob.content)

  def _feat_lexical(self, oa: Observation, ob: Observation) -> float:
    return fuzz.token_sort_ratio(oa.entity_norm, ob.entity_norm) / 100.0

  def _feat_interaction(self, oa: Observation, ob: Observation) -> float:
    modality_pairs = {
      frozenset({"video", "audio"}): 0.85,
      frozenset({"video", "text"}): 0.72,
      frozenset({"audio", "text"}): 0.68,
      frozenset({"video"}): 0.50,
      frozenset({"audio"}): 0.50,
      frozenset({"text"}): 0.45,
    }
    base = modality_pairs.get(frozenset({oa.modality.lower(), ob.modality.lower()}), 0.30)
    if oa.role and ob.role and oa.role.lower() != ob.role.lower():
      conflicting = {frozenset({"suspect", "witness"}), frozenset({"suspect", "victim"}), frozenset({"perpetrator", "victim"})}
      if frozenset({oa.role.lower(), ob.role.lower()}) in conflicting:
        base *= 0.55
    return min(1.0, base)

  def _feat_modality(self, oa: Observation, ob: Observation) -> float:
    return 0.40 if oa.modality.lower() == ob.modality.lower() else 0.82

  def _feat_entity_coreference(self, oa: Observation, ob: Observation) -> float:
    # FIX: an exact alias match is near-certain evidence of the same entity
    # (the generator guarantees no alias string is reused across different
    # real entities within a modality). The heuristic fallback already
    # encoded this (returns 1.0 for oa.entity_norm == ob.entity_norm), but
    # the LLM path did not enforce it - it scores pairs holistically on
    # content, and can under-score two same-alias observations describing
    # visibly different moments (e.g. two different actions by the same
    # witness), pulling the merge composite below threshold and splitting
    # one real entity into two clusters. This override makes the guarantee
    # hold regardless of which scoring path (LLM or heuristic) is active.
    if oa.entity_norm == ob.entity_norm:
        return 1.0
    return self.entity_agent.score(oa, ob, self.temporal_window_sec, self.max_temporal_gap_sec)

  def _stage_4_feature_computation(self) -> None:
    self.context_agent.precompute_batch_scores(self.candidate_pairs, self._obs_by_id)
    self.entity_agent.precompute_batch_scores(
      self.candidate_pairs, self._obs_by_id, self.temporal_window_sec, self.max_temporal_gap_sec
    )
    self._pair_features = {}
    for key in self.candidate_pairs:
      oa, ob = self._obs_by_id.get(key[0]), self._obs_by_id.get(key[1])
      if not oa or not ob:
        continue
      pf = PairFeatures(obs_a=oa, obs_b=ob)
      pf.entity_coreference = self._feat_entity_coreference(oa, ob)
      pf.mention_consistency, _ = self._feat_mention_consistency(oa, ob)
      pf.temporal = self._feat_temporal(oa, ob)
      pf.location = self._feat_location(oa, ob)
      pf.context = self._feat_context(oa, ob)
      pf.lexical = self._feat_lexical(oa, ob)
      pf.interaction = self._feat_interaction(oa, ob)
      pf.modality = self._feat_modality(oa, ob)
      self._pair_features[key] = pf

  def _stage_5_scoring(self) -> None:
    blocked_alias_pairs = {
      self._canonical_pair(normalize_alias(a), normalize_alias(b))
      for a, b in self.constraints.must_not_merge
    }
    for key, pf in self._pair_features.items():
      oa, ob = pf.obs_a, pf.obs_b
      hint_key = self._canonical_pair(oa.entity_norm, ob.entity_norm)
      pf.compute_composite(soft_hint=self.constraints.soft_hints.get(hint_key, 0.0))
      pf.reasons = [name for name in FEATURE_NAMES if getattr(pf, name) > 0.5]
      if self._canonical_pair(oa.entity_norm, ob.entity_norm) in blocked_alias_pairs:
        pf.composite = 0.0
        pf.reasons = []
        pf.hard_negative = True
      # FIX: real over-merging bug, verified against CASE_ATM_002 ground
      # truth. The generator guarantees each real entity gets AT MOST
      # ONE alias per modality (video Person_NN, audio Speaker_X, text
      # from a fixed set - never two different alias strings for the
      # same entity in one modality). The converse is a hard, provable
      # fact: two DIFFERENT alias strings in the SAME modality can never
      # be the same real entity. Without this, the cross-modal bonus
      # (designed to bridge one person across sensors) incorrectly also
      # bridges multiple coordinating same-role actors on different
      # channels - verified: Person_97/Person_50 (both video),
      # Speaker_D/Speaker_Q (both audio) were being merged despite
      # being different real suspects, purely because this invariant was
      # never checked.
      elif oa.modality == ob.modality and oa.entity_norm != ob.entity_norm:
        pf.composite = 0.0
        pf.reasons = []
        pf.hard_negative = True

  def _stage_6_classification(self) -> None:
    self.edges = []
    for pf in self._pair_features.values():
      if pf.hard_negative:
        cls = "rejected"
      elif pf.composite >= self.confirmed_threshold:
        cls = "confirmed"
      elif pf.composite >= self.candidate_threshold_high:
        cls = "likely"
      elif pf.composite >= self.candidate_threshold_low:
        cls = "possible"
      else:
        cls = "rejected"
      self.edges.append(
        EdgeRecord(
          alias_1=pf.obs_a.entity_norm,
          alias_2=pf.obs_b.entity_norm,
          obs_id_1=pf.obs_a.obs_id,
          obs_id_2=pf.obs_b.obs_id,
          weight=pf.composite,
          classification=cls,
          features=pf,
          reasons=list(pf.reasons),
          hard_negative=pf.hard_negative,
        )
      )

  def _stage_7_graph_construction(self) -> None:
    G = nx.Graph()
    for obs in self.observations:
      G.add_node(obs.obs_id)
    for edge in self.edges:
      if edge.classification != "confirmed":
        continue
      if G.has_edge(edge.obs_id_1, edge.obs_id_2):
        G[edge.obs_id_1][edge.obs_id_2]["weight"] = max(G[edge.obs_id_1][edge.obs_id_2]["weight"], edge.weight)
        G[edge.obs_id_1][edge.obs_id_2]["support"] += 1
      else:
        G.add_edge(edge.obs_id_1, edge.obs_id_2, weight=edge.weight, support=1)
    self.graph = G

  def _stage_8_clustering(self) -> None:
    components = {f"C{i+1}": sorted(comp) for i, comp in enumerate(nx.connected_components(self.graph))}
    uf = UnionFind([o.obs_id for o in self.observations])
    for comp in components.values():
      for j in range(1, len(comp)):
        uf.union(comp[0], comp[j])

    # FIX: unconditional same-alias merge pass, decoupled entirely from
    # edge classification/composite scoring. Without this, a same-alias
    # pair whose OTHER features (context, location, ...) score low enough
    # to get classified "rejected" (composite < CANDIDATE_THRESHOLD_LOW)
    # never even reaches the same-alias merge logic below, regardless of
    # the _feat_entity_coreference override - splitting one real entity
    # into two clusters purely because content happened to read as
    # dissimilar. Same alias string within one modality is near-certain
    # ground truth (the generator guarantees no alias reuse across real
    # entities), so this merges unconditionally, gated only by a generous
    # temporal sanity bound - never by composite score or edge
    # classification, so it cannot be undermined by LLM scoring variance.
    by_alias: Dict[str, List[Observation]] = defaultdict(list)
    for obs in self.observations:
      by_alias[obs.entity_norm].append(obs)
    for alias, obs_group in by_alias.items():
      if len(obs_group) < 2:
        continue
      base = obs_group[0]
      for other in obs_group[1:]:
        dt = abs(base.time_offset_sec - other.time_offset_sec)
        if dt <= self.max_temporal_gap_sec:
          uf.union(base.obs_id, other.obs_id)

    for edge in self.edges:
      if edge.hard_negative or edge.classification == "rejected":
        continue
      oa, ob = self._obs_by_id[edge.obs_id_1], self._obs_by_id[edge.obs_id_2]
      dt = abs(oa.time_offset_sec - ob.time_offset_sec)
      temporal_ok = dt <= self.max_temporal_gap_sec
      role_ok = oa.role.strip().lower() == ob.role.strip().lower()
      entity_corr = edge.features.entity_coreference if edge.features else 0.0
      composite = edge.weight

      if oa.entity_norm == ob.entity_norm:
        # Same alias can move across locations during an incident.
        if temporal_ok and composite >= MERGE_COMPOSITE_MIN:
          uf.union(edge.obs_id_1, edge.obs_id_2)
      elif (
        role_ok
        and temporal_ok
        and composite >= MERGE_COMPOSITE_MIN
      ):
        # FIX: heuristic entity_coreference (~0.78) is not discriminative
        # enough in multi-actor cases. Use a much higher bar when only
        # heuristics are active — real LLM/embedding scores use standard bar.
        coref_min = (
            CROSS_MODAL_MERGE_MIN
            if self._llm_actually_active
            else CROSS_MODAL_MERGE_MIN_HEURISTIC
        )
        if entity_corr >= coref_min:
          uf.union(edge.obs_id_1, edge.obs_id_2)

    self.clusters = {f"C{i+1}": sorted(m) for i, (_, m) in enumerate(uf.groups().items())}

  def _stage_9_guarded_attachment(self) -> None:
    obs_to_cluster = {oid: cid for cid, members in self.clusters.items() for oid in members}
    self._candidate_data = defaultdict(list)

    def is_singleton(obs_id: str) -> bool:
      cid = obs_to_cluster.get(obs_id)
      return cid is not None and len(self.clusters.get(cid, [])) == 1

    for edge in self.edges:
      if edge.classification not in ("likely", "possible") or edge.hard_negative:
        continue
      for singleton_id, partner_id in (
        (edge.obs_id_1, edge.obs_id_2) if is_singleton(edge.obs_id_1) else (None, None),
        (edge.obs_id_2, edge.obs_id_1) if is_singleton(edge.obs_id_2) else (None, None),
      ):
        if not singleton_id:
          continue
        partner_cid = obs_to_cluster.get(partner_id)
        if not partner_cid:
          continue
        if edge.weight >= ATTACHMENT_THRESHOLD:
          old_cid = obs_to_cluster[singleton_id]
          self.clusters[partner_cid].append(singleton_id)
          self.clusters[old_cid].remove(singleton_id)
          if not self.clusters[old_cid]:
            del self.clusters[old_cid]
          obs_to_cluster[singleton_id] = partner_cid
        else:
          s_obs, p_obs = self._obs_by_id[singleton_id], self._obs_by_id[partner_id]
          reasons = list(edge.reasons)
          self._candidate_data[singleton_id].append({"candidate_alias": p_obs.entity_norm, "score": round(edge.weight, 4), "reasons": reasons})
          self._candidate_data[partner_id].append({"candidate_alias": s_obs.entity_norm, "score": round(edge.weight, 4), "reasons": reasons})
    self.clusters = {k: v for k, v in self.clusters.items() if v}

  def _cluster_confidence(self, cluster_obs_ids: List[str]) -> float:
    if len(cluster_obs_ids) <= 1:
      obs = self._obs_by_id.get(cluster_obs_ids[0]) if cluster_obs_ids else None
      return obs.confidence if obs else 0.5
    member_set = set(cluster_obs_ids)
    total_weight = weighted_sum = 0.0
    for edge in self.edges:
      if edge.classification == "confirmed" and edge.obs_id_1 in member_set and edge.obs_id_2 in member_set:
        oa, ob = self._obs_by_id[edge.obs_id_1], self._obs_by_id[edge.obs_id_2]
        w = (oa.confidence + ob.confidence) / 2.0
        total_weight += w
        weighted_sum += w * edge.weight
    if total_weight == 0.0:
      confs = [self._obs_by_id[oid].confidence for oid in cluster_obs_ids if oid in self._obs_by_id]
      return sum(confs) / len(confs) if confs else 0.5
    return weighted_sum / total_weight

  def _stage_10_conflict_detection(self) -> None:
    # FIX: reset at the top so this is safe to call twice - Stage 10b
    # (resplitting) needs an accurate re-check against the post-split
    # cluster structure, not a stale accumulation from before the split.
    self.conflicts = []
    self.status = "success"
    sizes = [len(v) for v in self.clusters.values()]
    if len(sizes) >= 2:
      mean_sz = sum(sizes) / len(sizes)
      std_sz = (sum((s - mean_sz) ** 2 for s in sizes) / len(sizes)) ** 0.5
    else:
      mean_sz = sizes[0] if sizes else 1
      std_sz = 1.0
    oversized_threshold = mean_sz + OVERSIZED_CLUSTER_FACTOR * std_sz

    for cid, members in self.clusters.items():
      obs_list = [self._obs_by_id[oid] for oid in members if oid in self._obs_by_id]
      loc_time_map: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
      for obs in obs_list:
        if obs.location.strip():
          loc_time_map[(obs.location.strip().lower(), obs.time_offset_sec)].add(obs.entity_norm)
      for (loc, ts), _ in loc_time_map.items():
        if any(o.time_offset_sec == ts and o.location.strip() and o.location.strip().lower() != loc for o in obs_list):
          self.conflicts.append({
            "type": "physical_impossibility",
            "cluster_id": cid,
            "detail": f"Concurrent different locations at t={ts}s in {cid}.",
          })
          self.status = "awaiting_human_validation"
          break
      cc = self._cluster_confidence(members)
      if cc < CLUSTER_CONFIDENCE_FLOOR:
        self.conflicts.append({
          "type": "low_confidence",
          "cluster_id": cid,
          "cluster_confidence": round(cc, 4),
          "detail": f"Cluster {cid} confidence {cc:.3f} below floor.",
        })
        self.status = "awaiting_human_validation"
      if oversized_threshold > 0 and len(members) > oversized_threshold:
        self.conflicts.append({
          "type": "oversized_cluster",
          "cluster_id": cid,
          "size": len(members),
          "threshold": round(oversized_threshold, 1),
          "detail": f"Cluster {cid} oversized ({len(members)}).",
        })
        self.status = "awaiting_human_validation"

    # FIX: real over-merging on CASE_ATM_002 (verified against ground
    # truth) went undetected end-to-end - confidently reported 2 entities
    # when the FIR itself claims 3 suspects + 2 witnesses. The
    # same-modality rule above catches the most direct violations, but
    # cross-modal bridging between genuinely different same-role actors
    # coordinating closely in time remains a hard, open problem with the
    # current feature set. Rather than pretend a threshold tweak reliably
    # solves that, this makes the discrepancy VISIBLE: compare resolved
    # entity counts per role against the FIR own stated counts
    # (legitimate input, not ground truth) and flag a mismatch, so it
    # surfaces as AMBIGUOUS for investigator review instead of a
    # confident, wrong answer.
    if self.fir_role_counts:
      resolved_role_counts: Dict[str, Set[str]] = defaultdict(set)
      for cid, members in self.clusters.items():
        obs_list = [self._obs_by_id[oid] for oid in members if oid in self._obs_by_id]
        for obs in obs_list:
          resolved_role_counts[obs.role.strip().lower()].add(cid)
      for role, expected_count in self.fir_role_counts.items():
        role_key = str(role).strip().lower()
        actual_count = len(resolved_role_counts.get(role_key, set()))
        if actual_count > 0 and actual_count < int(expected_count):
          self.conflicts.append({
            "type": "role_count_mismatch",
            "role": role_key,
            "expected_count": int(expected_count),
            "resolved_count": actual_count,
            "detail": (
              f"FIR states {expected_count} distinct '{role_key}' role(s), "
              f"but only {actual_count} were resolved. This can mean the "
              "clustering over-merged distinct people, OR that some FIR-claimed "
              "actors simply have no observations in this evidence set (a "
              "legitimate gap, not an error) - the pipeline cannot distinguish "
              "the two without ground truth. Review manually."
            ),
          })
          self.status = "awaiting_human_validation"

  def _stage_10b_constrained_resplit(self) -> None:
    """
    FIX: attempts to algorithmically resolve role_count_mismatch instead of
    only flagging it, using single-linkage clustering (build a maximum
    spanning tree over the offending cluster's observations by composite
    score already computed - zero new scoring, zero new API calls - then
    cut the weakest bridging edges).

    Safety constraint: a cut is only applied if EVERY edge being cut scores
    below SPLIT_CUT_MAX_WEIGHT - a genuinely weak bridge, not just the
    least-strong of several confidently strong bonds. Verified against
    CASE_ATM_003 (role_count_mismatch caused by missing data, not
    over-merging - every internal edge is strong, so this correctly
    declines to force a split) and CASE_ATM_002 (the true cross-suspect
    bridge, e.g. speaker_d<->person_50 at 0.751, scores HIGHER than
    legitimate within-suspect bonds like log_30<->speaker_d at 0.681 - so
    this also correctly declines rather than risk cutting the wrong edge).
    """
    self._resplit_log = []
    if not self.fir_role_counts:
      return

    role_to_clusters: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for cid, members in self.clusters.items():
      obs_list = [self._obs_by_id[oid] for oid in members if oid in self._obs_by_id]
      for role in {o.role.strip().lower() for o in obs_list}:
        role_to_clusters[role][cid] = members

    for role, expected_count in self.fir_role_counts.items():
      role_key = str(role).strip().lower()
      clusters_for_role = role_to_clusters.get(role_key, {})
      resolved_count = len(clusters_for_role)
      deficit = int(expected_count) - resolved_count
      if deficit <= 0 or not clusters_for_role:
        continue

      target_cid = max(clusters_for_role, key=lambda c: len(clusters_for_role[c]))
      target_members = clusters_for_role[target_cid]
      target_subclusters = deficit + 1

      new_groups = self._attempt_single_linkage_split(target_members, target_subclusters)
      if new_groups is None:
        continue

      del self.clusters[target_cid]
      for i, group in enumerate(new_groups):
        self.clusters[f"{target_cid}s{i + 1}"] = group
      self._resplit_log.append({
        "original_cluster": target_cid,
        "role": role_key,
        "split_into": len(new_groups),
        "fir_expected_count": int(expected_count),
      })

    if self._resplit_log:
      log.info("Stage 10b: applied %d resplit(s): %s", len(self._resplit_log), self._resplit_log)
      self._stage_10_conflict_detection()

  def _attempt_single_linkage_split(self, obs_ids: List[str], k: int):
    """
    Build a maximum spanning tree over obs_ids (edge weight = best
    composite score already computed for the pair), then cut the (k-1)
    weakest tree edges to yield k components. Returns None if there aren't
    enough observations, the graph isn't connected, or any candidate cut
    edge is not weak enough to justify treating it as a genuine boundary
    between different people.
    """
    if len(obs_ids) < k:
      return None

    obs_set = set(obs_ids)
    G = nx.Graph()
    G.add_nodes_from(obs_ids)
    for pf in self._pair_features.values():
      a, b = pf.obs_a.obs_id, pf.obs_b.obs_id
      if a in obs_set and b in obs_set and a != b:
        w = pf.composite
        if G.has_edge(a, b):
          G[a][b]["weight"] = max(G[a][b]["weight"], w)
        else:
          G.add_edge(a, b, weight=w)

    by_alias: Dict[str, List[str]] = defaultdict(list)
    for oid in obs_ids:
      by_alias[self._obs_by_id[oid].entity_norm].append(oid)
    for alias_group in by_alias.values():
      for i in range(len(alias_group)):
        for j in range(i + 1, len(alias_group)):
          a, b = alias_group[i], alias_group[j]
          if G.has_edge(a, b):
            G[a][b]["weight"] = max(G[a][b]["weight"], 1.0)
          else:
            G.add_edge(a, b, weight=1.0)

    if G.number_of_edges() == 0 or not nx.is_connected(G):
      return None

    mst = nx.maximum_spanning_tree(G, weight="weight")
    mst_edges = sorted(mst.edges(data="weight"), key=lambda e: e[2])
    cut_edges = mst_edges[: k - 1]

    if any(w >= SPLIT_CUT_MAX_WEIGHT for _, _, w in cut_edges):
      return None

    mst.remove_edges_from([(a, b) for a, b, _ in cut_edges])
    components = list(nx.connected_components(mst))
    if len(components) != k:
      return None
    return [sorted(c) for c in components]

  def _stage_11_entity_labeling(self) -> None:
    self._canonical_entities = []
    assigned: Set[str] = set()
    deduped: Dict[str, List[str]] = {}
    for cid, members in sorted(self.clusters.items()):
      unique = [m for m in members if m not in assigned]
      if unique:
        deduped[cid] = unique
        assigned |= set(unique)
    self.clusters = deduped

    def fmt_ts(epoch: float) -> str:
      return "" if epoch <= 0 else datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for entity_idx, (_, members) in enumerate(self.clusters.items(), start=1):
      obs_list = [self._obs_by_id[oid] for oid in members if oid in self._obs_by_id]
      if not obs_list:
        continue
      raw_aliases = [o.entity for o in obs_list]
      primary_alias = Counter(raw_aliases).most_common(1)[0][0]
      aliases_unique = list(dict.fromkeys(raw_aliases))
      epochs = [o._ts_epoch for o in obs_list if o._ts_epoch > 0]
      member_set = set(members)
      confirmed_obs_ids: Set[str] = set()
      for e in self.edges:
        if e.classification == "confirmed" and (e.obs_id_1 in member_set or e.obs_id_2 in member_set):
          if e.obs_id_1 in member_set:
            confirmed_obs_ids.add(e.obs_id_1)
          if e.obs_id_2 in member_set:
            confirmed_obs_ids.add(e.obs_id_2)
      candidate_mentions: List[Dict[str, Any]] = []
      seen: Set[Tuple[str, str]] = set()
      for oid in members:
        for cdata in self._candidate_data.get(oid, []):
          key = (cdata["candidate_alias"], str(round(cdata["score"], 2)))
          if key not in seen:
            candidate_mentions.append({
              "candidate_alias": cdata["candidate_alias"],
              "score": cdata["score"],
              "reasons": list(cdata.get("reasons", [])),
            })
            seen.add(key)
      self._canonical_entities.append({
        "entity_id": f"entity_{entity_idx}",
        "aliases": aliases_unique,
        "primary_alias": primary_alias,
        "total_mentions": len(obs_list),
        "confirmed_mentions": sorted(confirmed_obs_ids),
        "candidate_mentions": candidate_mentions,
        "confidence_score": round(self._cluster_confidence(members), 4),
        "confirmed_edges": sum(1 for e in self.edges if e.classification == "confirmed" and (e.obs_id_1 in member_set or e.obs_id_2 in member_set)),
        "candidate_edges": sum(1 for e in self.edges if e.classification in ("likely", "possible") and not e.hard_negative and (e.obs_id_1 in member_set or e.obs_id_2 in member_set)),
        "modalities": sorted({o.modality for o in obs_list}),
        "locations": sorted({o.location for o in obs_list if o.location}),
        "roles": sorted({o.role for o in obs_list}),
        "sources": sorted({o.obs_id for o in obs_list}),
        "earliest_timestamp": fmt_ts(min(epochs) if epochs else 0),
        "latest_timestamp": fmt_ts(max(epochs) if epochs else 0),
        "time_span_seconds": int(max(epochs) - min(epochs)) if epochs else 0,
      })

  def _stage_12_package(self, total_time: float) -> Dict[str, Any]:
    cluster_output = []
    for cid, members in self.clusters.items():
      member_set = set(members)
      edge_dicts = []
      for e in self.edges:
        if e.obs_id_1 in member_set or e.obs_id_2 in member_set:
          edge_reasons = [] if e.hard_negative or e.classification == "rejected" else list(e.reasons)
          edge_dicts.append({
            "alias_1": e.alias_1,
            "alias_2": e.alias_2,
            "weight": round(e.weight, 4),
            "support": e.support,
            "classifications": {
              "confirmed": 1 if e.classification == "confirmed" else 0,
              "candidate": 1 if e.classification in ("likely", "possible") else 0,
              "rejected": 1 if e.classification == "rejected" else 0,
            },
            "hard_negative": e.hard_negative,
            "reasons": edge_reasons,
          })
      cluster_output.append({
        "cluster_id": cid,
        "size": len(members),
        "aliases": list({self._obs_by_id[oid].entity for oid in members if oid in self._obs_by_id}),
        "obs_ids": sorted(members),
        "edges": edge_dicts,
      })

    remap = {
      "stage_1_intake": "stage_1_intake",
      "stage_2_normalization": "stage_2_normalization",
      "stage_3_blocking": "stage_3_blocking",
      "stage_4_feature_computation": "stage_4_features",
      "stage_5_scoring": "stage_5_scoring",
      "stage_6_classification": "stage_6_classification",
      "stage_7_graph_construction": "stage_7_graph_building",
      "stage_8_clustering": "stage_8_clustering",
      "stage_9_guarded_attachment": "stage_9_attachment",
      "stage_10_conflict_detection": "stage_10_conflict_detection",
      "stage_11_entity_labeling": "stage_11_labeling",
    }
    final_timings = {remap.get(k, k): round(v, 6) for k, v in self.stage_timings.items()}
    final_timings["stage_12_packaging"] = 0.0

    _output_classification, _output_classification_reason = classify_resolution_output(
      self._canonical_entities, self.conflicts
    )
    return {
      "case_id": self.case_id,
      "status": self.status,
      "error_message": self.error_message,
      "entity_count": len(self._canonical_entities),
      "canonical_entities": self._canonical_entities,
      "clusters": cluster_output,
      "conflicts_detected": len(self.conflicts),
      # FIX (Timeline Agent contract bug): expose the actual conflict
      # records, not just their count. Without this, the Timeline Agent
      # (or any downstream consumer) cannot know WHICH observations were
      # flagged, and conflict-aware confidence penalties silently never fire.
      "conflicts": self.conflicts,
      "output_classification": _output_classification,
      "output_classification_reason": _output_classification_reason,
      # Transparency, mirroring the Timeline Agent notebook's llm_calls_made field.
      "llm_calls_made": self._context_budget.calls_made + self._entity_budget.calls_made,
      "llm_calls_budget": self._context_budget.max_calls + self._entity_budget.max_calls,
      # Transparency: which similarity backend actually scored locations -
      # "embedding" (real semantic model) or "lexical_fallback" (token
      # overlap, used if the model could not be loaded).
      "location_similarity_backend": get_semantic_scorer().backend_used(),
      "resplit_log": self._resplit_log,
      "configuration": {
        "check_duplicates": self.check_duplicates,
        "case_base_time": self.case_base_time,
        "temporal_window_sec": self.temporal_window_sec,
        "max_temporal_gap_sec": self.max_temporal_gap_sec,
        "max_pairs": self.max_pairs,
        "confirmed_threshold": self.confirmed_threshold,
        "candidate_threshold_low": self.candidate_threshold_low,
        "candidate_threshold_high": self.candidate_threshold_high,
        "llm_enabled": self.llm_enabled,
        "groq_model": GROQ_MODEL,
        "cross_modal_merge_min": CROSS_MODAL_MERGE_MIN,
        "merge_composite_min": MERGE_COMPOSITE_MIN,
      },
      "total_processing_time_sec": round(total_time, 6),
      "stage_timings": final_timings,
      "created_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def resolve_entities(
  observations_payload: Dict[str, Any],
  config: Optional[Dict[str, Any]] = None,
  human_constraints: Optional[HumanConstraints] = None,
  llm_enabled: bool = True,
  max_llm_calls: int = MAX_LLM_CALLS_PER_RUN,
) -> Dict[str, Any]:
  return EntityResolutionPipeline(config=config, human_constraints=human_constraints, llm_enabled=llm_enabled, max_llm_calls=max_llm_calls).run(observations_payload)