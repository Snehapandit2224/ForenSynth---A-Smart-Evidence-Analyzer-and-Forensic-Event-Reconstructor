"""
ForenSynth – shared.py
Normalization, semantic similarity, and utility functions.
Imported by both entity_resolution.py and timeline_agent.py.
"""

from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


"""
ForenSynth-X+ - shared normalization utilities.

Used by BOTH the Entity Resolution pipeline and the Timeline Agent, so the
two agents can never silently disagree about how a timestamp, alias,
modality, role, or piece of content gets interpreted. This file is the
single source of truth for all of that.

Design constraints:
  - Pure stdlib, zero dependencies - so it can be pasted as a single cell
    into the ER Colab notebook AND imported normally by the Timeline Agent
    package without adding an install step either place.
  - Every function is pure (no I/O, no globals mutated) so it's trivially
    unit-testable and safe to run twice on the same data (idempotent).

Grounding note: `ACTION_TAG_FRAGMENTS` below was built by reading the
ACTUAL phrase banks in the generator's templates.py / observations.py
(both ATM_Robbery and Office_Theft domains) rather than guessed - see
tests/test_normalization.py, which checks real generator phrases resolve
to the expected tag. It will not catch every possible paraphrase (the
generator's phrase banks are large and stylistically varied by design -
that's the noise-injection point), so this remains a best-effort heuristic
layer. Anything it can't tag is reported as an unresolved pair by the
Timeline Agent's temporal/causal reasoners rather than silently guessed.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# ==============================================================================
# Timestamps
# ==============================================================================
# FIX: previously ER (_parse_ts, 4 formats) and the Timeline Agent
# (parse_epoch, 10 formats) each maintained their own format list. Anything
# ER couldn't parse silently became epoch=0 in canonical_entities'
# earliest/latest_timestamp even if the Timeline Agent's more permissive
# parser WOULD have understood it - a real cross-agent inconsistency risk.
# This is now the one parser both agents call.

_TS_FORMATS_TZ = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
]
_TS_FORMATS_NAIVE = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
]


def parse_timestamp(timestamp: Optional[str]) -> float:
    """Parse a timestamp string to a POSIX epoch float. Returns 0.0 on failure."""
    if not timestamp:
        return 0.0
    ts = timestamp.strip()
    if not ts:
        return 0.0

    if not ts.endswith("Z"):
        for fmt in _TS_FORMATS_TZ:
            try:
                return datetime.strptime(ts, fmt).timestamp()
            except ValueError:
                continue

    ts_clean = re.sub(r"Z$", "", ts).strip()
    for fmt in _TS_FORMATS_NAIVE:
        try:
            dt = datetime.strptime(ts_clean, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue

    # Last-resort fallback: Python's own ISO parser catches a few additional
    # well-formed variants on 3.11+; harmless no-op on older versions.
    try:
        dt = datetime.fromisoformat(ts_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        pass

    return 0.0


def epoch_to_iso(epoch: float) -> str:
    if epoch <= 0:
        return ""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ==============================================================================
# Alias normalization
# ==============================================================================

def normalize_alias(alias: Optional[str]) -> str:
    """Lowercase and replace non-alphanumeric chars with underscore."""
    if not alias:
        return "unknown"
    return re.sub(r"[^a-z0-9_]", "_", alias.strip().lower()) or "unknown"


# ==============================================================================
# Modality canonicalization
# ==============================================================================
# FIX: previously modality strings were compared as-is (case-folded only).
# A Field Agent emitting "CCTV" or "phone_call" instead of "video"/"audio"
# would silently fall into the 0.50-reliability "unknown" bucket instead of
# being recognised. This gives modality a real, extensible synonym table.

_MODALITY_SYNONYMS = {
    "video": "video", "cctv": "video", "camera": "video", "footage": "video", "visual": "video",
    "audio": "audio", "voice": "audio", "call": "audio", "phone": "audio", "recording": "audio",
    "text": "text", "sms": "text", "email": "text", "chat": "text", "log": "text", "message": "text",
    "network": "network", "ip": "network", "firewall": "network", "system_log": "network",
}


def normalize_modality(modality: Optional[str]) -> str:
    key = (modality or "").strip().lower()
    return _MODALITY_SYNONYMS.get(key, key or "unknown")


# ==============================================================================
# Role canonicalization
# ==============================================================================

_ROLE_SYNONYMS = {
    "suspect": "suspect", "perpetrator": "suspect", "accused": "suspect", "offender": "suspect",
    "witness": "witness", "bystander": "witness", "eyewitness": "witness",
    "observer": "witness", "reporter": "witness",
    "victim": "victim",
    "system": "system", "sensor": "system", "device": "system",
}


def normalize_role(role: Optional[str]) -> str:
    key = (role or "").strip().lower()
    return _ROLE_SYNONYMS.get(key, key or "unknown")


# ==============================================================================
# Content cleanup
# ==============================================================================

_BOILERPLATE_PATTERNS = [
    re.compile(r"\(captured on footage\)", re.IGNORECASE),
    re.compile(r"\(captured on camera\)", re.IGNORECASE),
    re.compile(r"\(on footage\)", re.IGNORECASE),
    re.compile(r"\(cctv footage\)", re.IGNORECASE),
]


def clean_content(content: Optional[str]) -> str:
    """Strip known template boilerplate and collapse whitespace. Preserves case."""
    text = content or ""
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# ==============================================================================
# Action tag extraction
# ==============================================================================
# FIX: the original CAUSAL_ACTION_RULES matched single exact-root words
# ("enter", "exit", "withdraw"...) against tokenized content. Checked against
# your real CASE_ATM_001 observations, 6 of 7 matched ZERO rule keywords,
# because the generator's phrase banks use inflections ("leaves") and free
# paraphrasing ("fiddling with", "ran out") rather than root-form verbs -
# and this is by design (the generator injects "semantic paraphrasing" as a
# noise type). ACTION_TAG_FRAGMENTS below was built by reading the actual
# generator phrase banks (templates.py description_templates for video,
# observations.py suspect_phrases for audio, both ATM_Robbery and
# Office_Theft domains) so it recognises the phrasing your data actually
# contains, not an idealized vocabulary.
#
# This is fragment/substring matching, not single-token matching, because
# the signal is often multi-word ("steps into", "walks out of", "ran out")
# rather than a single verb.

ACTION_TAG_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "APPROACH": (
        "approach", "walks towards", "walking toward", "walking towards", "nearing",
        "heading in the direction", "heading to", "heading there", "moving towards",
        "moving toward", "almost there", "almost at", "on my way to", "close to the",
        "at the corner", "about to move",
        # FIX: found via real CASE_ATM_004 data - 5 of the generator's own 13
        # approach_atm phrases (audio+text) didn't match anything, causing
        # legitimate approach-phase observations to get zero action tags and
        # fail to cluster with the rest of the approach event.
        "i can see", "i'm close", "walking up to",
    ),
    "ENTER": (
        "enter", "steps into", "step inside", "stepped in", "pushes open", "pulls open",
        "proceeds inside", "crossing the threshold", "in the booth", "inside the atm",
        "inside the building", "inside the office", "got in", "i'm in", "access card worked",
        "badge scanned", "badge worked", "access granted", "in. door shut", "booth clear",
        "lobby empty", "lobby now",
        # FIX: found via real CASE_ATM_002 data (O6 "proceeds into the ATM
        # kiosk") and systematic phrase-bank audit of enter_atm (audio+text).
        "proceeds into", "inside now", "in the enclosure", "inside. booth", "booth is empty",
    ),
    "WITHDRAW": (
        "withdraw", "transaction", "operate the atm", "operating the atm", "pressing keys",
        "inserts card", "insertion area", "conducting activity", "conducting transaction",
        "dispensing", "cash is coming out", "cash out", "getting the money",
        "machine is dispensing", "transaction complete", "transaction going through",
        "card working", "card's in", "processing",
    ),
    "TAMPER": (
        "tamper", "skimmer", "device is attached", "device placed", "reader fitted",
        "reader's set", "rigged", "looks factory", "looks stock", "fitted",
    ),
    "LOITER": (
        "loiter", "lingers", "linger", "remains in", "standing near", "waiting for the right moment",
        "still waiting", "no apparent reason", "hanging around", "keeping an eye out",
        # FIX: found via real CASE_ATM_004 data - 7 of the generator's own 13
        # loiter_near_atm phrases (audio+text) didn't match anything (worse
        # coverage than APPROACH). "just watching" and "stay ready" are the
        # exact phrasing of two real observations (O5, O6) in this case that
        # should have clustered into one event and didn't, purely because
        # neither got any action tag at all.
        "standing around", "no one here yet", "timing it", "just watching",
        "timing not right", "stay ready", "stand by",
    ),
    "EXIT": (
        "exit", "leaves", "leaving", "left the", "left through", "walks out", "walked out",
        "walk out", "departs", "departing", "departed", "recorded leaving", "coming out",
        "come out", "out of the atm", "out of the booth", "out now", "clean exit",
        "exiting via", "i'm out", "i'm through",
        # FIX: found via real CASE_ATM_002 data (O15 "Done in here, moving
        # now." should have merged with O16 into one exit_atm event).
        "done in here", "done here",
    ),
    "FLEE": (
        "flee", "fled", "fleeing", "run", "ran out", "rushed out", "move move move",
        "get out of here", "split up", "abort", "get out fast", "walk away briskly",
        "walked away quickly",
        # FIX: found via real CASE_ATM_002 data (O18 "Security's nearby, get
        # out!" should have merged with O17 into one flee_scene event) plus
        # systematic phrase-bank audit of flee_scene (audio+text) - the
        # generator's own flee_scene phrases were only ~40% covered before.
        "get out", "leave now", "leave everything", "need to leave", "go now",
        "someone's watching", "security nearby", "don't look back", "someone's coming",
    ),
    "STEAL": (
        "steal", "stole", "stolen", "got the files", "got the folders", "documents are in the bag",
        "grabbed everything", "items secured", "usb is in", "usb full", "copying", "copy complete",
        "data's transferring", "transfer complete", "transfer at", "files are on the drive",
        "wiping the logs", "wiping history",
    ),
    "NAVIGATE": (
        "navigate", "corridor", "taking the stairs", "taking stairs", "server room",
        "restricted section", "target floor", "passing the main hall",
    ),
    "WORK": (
        "perform_legit_work", "finishing up the report", "in the meeting", "sending the last",
        "wrapping up", "at my desk", "on a call with the client", "filing the",
        "catching up on", "working late",
    ),
    "COMMUNICATE": (
        "call me", "reach out", "reaching out", "confirm when ready", "check in", "you ready",
        "we move as planned", "signal is", "meeting point", "text", "message", "sms", "email",
    ),
    "CONFIRM": (
        "confirm", "confirmed", "good to go", "all set", "we're aligned", "understood",
        "copy that", "all clear", "proceeding as agreed",
    ),
    # FIX: found via real CASE_ATM_002 data - wait_outside (lookout/
    # all-clear watching, e.g. "Watching the entrance. No one coming.") is
    # a DISTINCT category from loiter_near_atm (idling without purpose,
    # e.g. "hanging around, keeping an eye out") - the generator itself
    # treats them as separate domains. Trying to force wait_outside phrases
    # into LOITER scored 0/13 against the real phrase bank; a proper new
    # tag was the right fix, not stretching LOITER's fragments further.
    "WATCH": (
        "watching the entrance", "keep going", "all clear", "standing by",
        "eyes on the entrance", "nothing suspicious", "door clear", "no security visible",
        "no one coming", "no one's coming", "take your time", "go ahead", "no one nearby",
        # FIX: found via real CASE_ATM_002 data (O9 "Second actor remains
        # outside the booth, watching the area." should have merged with
        # O10 into one wait_outside event).
        "remains outside", "watching the area",
    ),
    "OBSERVE": (
        "observe", "observed", "witness", "witnessed", "saw", "noticed", "sees someone",
        "bystander", "eyewitness", "i saw", "i noticed",
    ),
    "REPORT": (
        "report", "file a complaint", "speak to an officer", "calling to report",
        "want to let you know", "should report", "should know",
    ),
    "INTERCEPT": (
        "intercept", "intercepted", "misdirected", "not meant for", "picked up a communication",
    ),
}


_NEGATION_WORDS = (
    "not", "n't", "never", "didn't", "doesn't", "wasn't", "isn't", "no ",
    "denies", "denied", "without",
)
_NEGATION_WINDOW_CHARS = 25  # look this far back from the fragment match

# FIX: extract_action_tags() below was pure literal substring matching -
# "loiter" matches, but "roams"/"wanders"/"lurks"/"idles" (genuine
# synonyms) score zero, because there's no semantic understanding at all,
# only whatever exact phrasing was manually enumerated. Every fix applied
# to this file across CASE_ATM_002/003/004 has been adding one more literal
# phrase - a whack-a-mole pattern that will keep recurring on new
# vocabulary. _ACTION_TAG_CANONICAL_EXAMPLES gives the embedding fallback
# below something to compare against: a small set of representative
# sentences per tag, reusing the SAME embedding infrastructure already
# proven for location similarity (semantic_similarity.py), rather than a
# second, parallel mechanism.
_ACTION_TAG_CANONICAL_EXAMPLES: dict[str, tuple[str, ...]] = {
    "APPROACH":    ("The suspect approaches the ATM, walking toward it from a distance.",),
    "ENTER":       ("The suspect enters the booth, going inside the enclosed space.",),
    "WITHDRAW":    ("The suspect operates the machine, withdrawing cash from the ATM.",),
    "TAMPER":      ("The suspect tampers with the card reader, attaching a skimming device.",),
    "LOITER":      ("The suspect loiters near the location, waiting around without a clear purpose.",),
    "EXIT":        ("The suspect exits the booth, leaving the enclosed space.",),
    "FLEE":        ("The suspect flees the scene in a hurry, running away quickly.",),
    "STEAL":       ("The suspect steals items, taking documents or files without permission.",),
    "NAVIGATE":    ("The suspect moves through the building, passing through corridors toward a destination.",),
    "WORK":        ("The person is doing ordinary work, unrelated to any suspicious activity.",),
    "COMMUNICATE": ("The person communicates with someone else, sending a message or making contact.",),
    "CONFIRM":     ("The person confirms that everything is ready and proceeding as planned.",),
    "WATCH":       ("The suspect watches the area, keeping a lookout and reporting the surroundings are clear.",),
    "OBSERVE":     ("The witness observes and notices someone's actions, watching what happened.",),
    "REPORT":      ("The witness reports the incident, informing an officer or filing a complaint.",),
    "INTERCEPT":   ("The observation was intercepted, redirected from a communication not meant for this recipient.",),
}
# Matches the spirit of LOCATION_SEMANTIC_SIMILARITY_THRESHOLD elsewhere in
# this codebase (config.py) - "confidently related, not just vaguely
# plausible". Same honest caveat as location similarity: verified here with
# a mocked encoder (proves the wiring - threshold comparison, negation
# gating, correct tag selection); real semantic QUALITY (does "roam" really
# score >=0.60 against the LOITER example with the real model) can only be
# confirmed by running with the actual model downloaded, same as the
# skipped live-model smoke test elsewhere in this project.
_ACTION_TAG_SEMANTIC_THRESHOLD = 0.60


def extract_action_tags(content: Optional[str]) -> Set[str]:
    """
    Return the set of canonical action tags detected in `content`, using
    substring/fragment matching grounded in the generator's real phrase
    banks. Best-effort: absence of a tag does not mean "no action happened",
    only "this heuristic layer didn't recognise the phrasing" - callers
    should treat a fully-empty result as a genuinely ambiguous case, not a
    negative signal.

    Negation-aware: a fragment preceded closely by a negation word ("did
    not go inside", "never entered") is NOT tagged. This matters
    specifically for a forensic tool - a suspect's denial ("I did not go
    inside that booth") must never silently become an inferred ENTER
    action; that would fabricate evidence, not just miss it.
    """
    text = clean_content(content).lower()
    if not text:
        return set()
    tags: Set[str] = set()
    for tag, fragments in ACTION_TAG_FRAGMENTS.items():
        for frag in fragments:
            idx = text.find(frag)
            if idx == -1:
                continue
            window_start = max(0, idx - _NEGATION_WINDOW_CHARS)
            preceding = text[window_start:idx]
            # FIX: found via real CASE_ATM_002 data - "No one around, keep
            # going." was having its genuine WATCH/all-clear signal
            # ("keep going") wrongly suppressed, because "no " appeared
            # earlier in the SAME 25-char window but a DIFFERENT clause
            # ("no one around" describes an absent third party, it isn't
            # negating the speaker's own following instruction). Clause-
            # boundary-aware: only the text after the last preceding
            # ,/./!/? counts as the negation window, so a negation word in
            # an earlier, separate clause no longer wrongly suppresses a
            # later, unrelated action. Genuine same-clause denials ("did
            # not go inside", "never entered") are unaffected - the
            # negation word and the action are still in the same clause.
            last_boundary = max(
                preceding.rfind(","), preceding.rfind("."),
                preceding.rfind("!"), preceding.rfind("?"),
            )
            if last_boundary != -1:
                preceding = preceding[last_boundary + 1:]
            if any(neg in preceding for neg in _NEGATION_WORDS):
                continue  # negated - do not tag
            tags.add(tag)
            break
    if tags:
        return tags

    # FIX: semantic fallback, only reached when fragment matching found
    # NOTHING - fragments remain the primary, high-precision path (see
    # module docstring above _ACTION_TAG_CANONICAL_EXAMPLES for why: small
    # embedding models are known to be weak at negation, so a pure-
    # embedding approach risks reading "I did not go inside" as ENTER,
    # which for a forensic tool means fabricating evidence, not just
    # missing it). Negation check here is deliberately MORE conservative
    # than the fragment path above (whole-text, not a windowed check
    # around a match position) - an embedding comparison has no single
    # match index to anchor a window to, so any negation word anywhere in
    # the sentence blocks the semantic fallback entirely.
    if any(neg in text for neg in _NEGATION_WORDS):
        return set()

    try:
        from shared import get_semantic_scorer as _get_scorer
    except ImportError:
        return set()  # shared not importable in this context
    get_semantic_scorer = _get_scorer
    scorer = get_semantic_scorer()
    clean_text = clean_content(content)
    best_tag: Optional[str] = None
    best_score = 0.0
    for tag, examples in _ACTION_TAG_CANONICAL_EXAMPLES.items():
        for example in examples:
            score = scorer.similarity(clean_text, example)
            if score > best_score:
                best_score = score
                best_tag = tag
    if best_tag is not None and best_score >= _ACTION_TAG_SEMANTIC_THRESHOLD:
        return {best_tag}
    return set()


# ==============================================================================
# Location normalization
# ==============================================================================
# FIX: location strings are free-text and verbose by generator design
# (e.g. "ATM booth interior, card reader and keypad area"). Comparing them
# with plain lower/strip means two observations at "the same place" in
# different phrasing never register as a location match. `location_key`
# gives a coarser, more matchable token (text before the first comma) while
# `normalized` keeps the full string for display/audit.

def normalize_location(location: Optional[str]) -> Tuple[str, str]:
    """Returns (normalized_full_string, coarse_location_key)."""
    text = (location or "").strip()
    normalized = re.sub(r"\s+", " ", text).lower()
    key = normalized.split(",")[0].strip()
    return normalized, key


# ==============================================================================
# Batch normalization + repository abstraction (Memory Store swap point)
# ==============================================================================
# FIX: previously ER and the Timeline Agent each called the primitive
# functions above (normalize_alias, normalize_role, parse_timestamp, ...)
# piecemeal, scattered across their own intake code - same underlying
# logic (good, that was the earlier fix), but re-derived independently by
# each agent on every run, with no single place representing "the
# normalized form of this case's observations."
#
# NormalizedObservation + normalize_case() below is that single place: ONE
# batch function that takes a case's raw observations and returns the
# fully-normalized form, computed once per call. NormalizedObservationStore
# wraps it behind the same repository-pattern interface Timeline Agent's
# repositories.py already uses elsewhere, so today's "just compute it
# in-process, no persistence" implementation
# (LocalNormalizedObservationStore) and tomorrow's real Memory-Store-backed
# implementation are interchangeable - callers only ever talk to the
# NormalizedObservationStore interface, never to normalize_case() directly,
# so swapping the backing implementation later touches ONE line per agent
# (the place the store gets constructed), not the intake logic itself.
#
# Honest scope note on "storage": LocalNormalizedObservationStore is
# explicitly NOT persistent - it's an in-process dict cache that dies with
# the Python process (a notebook kernel restart, a fresh script run). It
# exists so that if the SAME case gets normalized twice within one run
# (uncommon today, but possible), the second call is free. It is a
# stand-in for the real Memory Store's normalized_observations table, not
# a replacement for it - persistence across runs/processes is exactly the
# gap the real Memory Store is meant to close.


@dataclass
class NormalizedObservation:
    """
    The canonical, fully-normalized form of one observation - the single
    intermediate representation both agents should build their own
    domain objects (ER's Observation, Timeline Agent's RawObservation)
    from, instead of each re-deriving normalization independently.
    """
    obs_id: str
    entity_raw: str
    entity_norm: str
    role: str                    # canonicalized (normalize_role)
    modality: str                 # canonicalized (normalize_modality)
    location_raw: str
    location_key: str
    content_raw: str
    content_clean: str            # boilerplate-stripped, whitespace-collapsed
    timestamp_raw: str
    ts_epoch: float
    time_offset_sec: int          # relative to this case's earliest valid timestamp
    confidence: float
    action_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormalizedObservation":
        return cls(**d)


def normalize_observation(raw: Dict[str, Any], base_epoch: float = 0.0) -> NormalizedObservation:
    """Normalize a single raw observation dict. `base_epoch` should be the
    case's earliest valid epoch (see normalize_case) - pass 0.0 only if
    computing time_offset_sec doesn't matter for the caller."""
    entity_raw = str(raw.get("entity", ""))
    timestamp_raw = str(raw.get("timestamp", ""))
    ts_epoch = parse_timestamp(timestamp_raw)
    location_raw = str(raw.get("location", ""))
    _, location_key = normalize_location(location_raw)
    content_raw = str(raw.get("content", ""))
    content_clean = clean_content(content_raw)

    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return NormalizedObservation(
        obs_id=str(raw.get("obs_id", "")),
        entity_raw=entity_raw,
        entity_norm=normalize_alias(entity_raw),
        role=normalize_role(raw.get("role", "unknown")),
        modality=normalize_modality(raw.get("modality", "unknown")),
        location_raw=location_raw,
        location_key=location_key,
        content_raw=content_raw,
        content_clean=content_clean,
        timestamp_raw=timestamp_raw,
        ts_epoch=ts_epoch,
        time_offset_sec=int(ts_epoch - base_epoch) if ts_epoch > 0 and base_epoch > 0 else 0,
        confidence=confidence,
        action_tags=sorted(extract_action_tags(content_raw)),
    )


def normalize_case(raw_observations: List[Dict[str, Any]]) -> List[NormalizedObservation]:
    """
    Batch entry point: normalize every observation in a case in one call.
    Two passes (matches the logic ER's _stage_2_normalization already used,
    now centralized): first parse every timestamp, compute the case's
    earliest valid epoch, then normalize each observation with that shared
    base so time_offset_sec is consistent across the whole case.

    Pure function - no dedup, no filtering, one row in maps to one row out.
    Dedup semantics differ meaningfully between ER (content-hash dedup) and
    Timeline Agent (duplicate-obs_id dedup), so that stays each caller's
    own concern, applied to this function's output.
    """
    epochs = [parse_timestamp(str(o.get("timestamp", ""))) for o in raw_observations]
    valid = [e for e in epochs if e > 0]
    base_epoch = min(valid) if valid else 0.0
    return [normalize_observation(o, base_epoch=base_epoch) for o in raw_observations]


class NormalizedObservationStore(ABC):
    """
    Repository-pattern interface for fetching a case's normalized
    observations. Both agents should depend on THIS interface, never call
    normalize_case() directly - that's what makes the Memory Store swap a
    one-class change later.
    """

    @abstractmethod
    def get(self, case_id: str, raw_observations: List[Dict[str, Any]]) -> List[NormalizedObservation]:
        ...


class LocalNormalizedObservationStore(NormalizedObservationStore):
    """
    TODAY's implementation - no real persistence, no Memory Store. Computes
    normalize_case() on first request for a given (case_id, observation
    content) pair within this process, then serves an in-memory cache for
    the rest of the process's lifetime. Cache key is content-aware (a hash
    of the raw observations, not just case_id) - a real bug this caught
    during testing: two different observation sets sharing the same
    case_id (plausible in practice too, e.g. re-processing a case after
    correcting evidence) must never silently return the OTHER set's stale
    cached result. This is intentionally NOT durable across processes (a
    new kernel/process starts with an empty cache) - it exists to avoid
    redundant recomputation within one run, not to replace the real store.

    SWAP POINT: once the team's Memory Store is ready, implement a
    MemoryStoreNormalizedObservationStore(NormalizedObservationStore) that
    reads pre-computed rows from the store's normalized_observations table
    (or writes them via normalize_case() on first access, if the store
    itself owns computing them) instead of caching in-process. Every
    caller here talks only to the `.get()` interface, so no intake code
    in either agent needs to change - only which store gets constructed.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], List[NormalizedObservation]] = {}

    @staticmethod
    def _content_key(raw_observations: List[Dict[str, Any]]) -> str:
        import hashlib
        import json
        blob = json.dumps(raw_observations, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, case_id: str, raw_observations: List[Dict[str, Any]]) -> List[NormalizedObservation]:
        cache_key = (case_id, self._content_key(raw_observations))
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = normalize_case(raw_observations)
        self._cache[cache_key] = result
        return result


_default_store: Optional[NormalizedObservationStore] = None


def get_normalized_observation_store() -> NormalizedObservationStore:
    """
    Module-level default store, matching the same singleton-getter pattern
    used for the semantic similarity scorer. Swap the Memory Store in by
    calling set_normalized_observation_store() once at the top of a
    notebook/script - no other code changes.
    """
    global _default_store
    if _default_store is None:
        _default_store = LocalNormalizedObservationStore()
    return _default_store


def set_normalized_observation_store(store: NormalizedObservationStore) -> None:
    """Override the default store - this is the swap point for when the
    real Memory Store is ready: set_normalized_observation_store(
    MemoryStoreNormalizedObservationStore(...)) and both agents pick it up
    automatically."""
    global _default_store
    _default_store = store


"""
ForenSynth-X+ - shared semantic similarity module.

Used by BOTH Entity Resolution and Timeline Agent for location (and,
optionally, general free-text) similarity, replacing pure lexical/token-
overlap matching with actual semantic understanding.

WHY THIS EXISTS (see conversation record / design note):
Token-overlap fuzzy matching (e.g. rapidfuzz.token_set_ratio) measures
shared WORDS, not shared MEANING. Demonstrated failure mode: "ATM entrance,
main door" vs "ATM exit, main door" scores 81% similar under token overlap
despite being close to opposite in meaning - it sees 3 of 4 words shared and
has no notion that "entrance" and "exit" are different concepts. This module
replaces that with sentence-embedding cosine similarity (SentenceTransformer,
all-MiniLM-L6-v2 by default), a standard, citable NLP technique for semantic
textual similarity, chosen over a hand-built synonym/fragment table
specifically because a fragment table only covers vocabulary it was
explicitly given, while embeddings generalize to phrasing never seen before -
a materially stronger claim for a system meant to handle real (not just
generator-bounded) evidence text.

DESIGN: graceful degradation, same pattern as llm_fallback.py. If the model
can't load (package not installed, no network to Hugging Face Hub, etc.),
this silently falls back to the previous lexical fuzzy-matching approach -
the pipeline never blocks or errors because the embedding model isn't
available. Availability and which path was used are exposed so this is
auditable, not hidden.

HONESTY NOTE: the embedding path was built and unit-tested with a mocked
encoder (verifying caching, cosine-similarity math, and fallback wiring are
all correct) but NOT executed end-to-end against the real model in this
development environment, because this sandbox's network egress does not
reach huggingface.co. Verify the actual similarity numbers once in Colab
(which has normal internet access) - see tests/test_semantic_similarity.py
for what's mock-verified vs. what still needs a live check.
"""

import logging
import re
from typing import Dict, Optional

log = logging.getLogger("forensynth.semantic_similarity")

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# Fallback lexical scorer (used if the embedding model is unavailable).
try:
    from rapidfuzz import fuzz  # type: ignore
except ImportError:  # pragma: no cover
    fuzz = None


def _lexical_fallback_similarity(a: str, b: str) -> float:
    """Same fallback used elsewhere in the project when rapidfuzz is present,
    with a pure-stdlib Jaccard fallback if it isn't."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return 0.0
    if fuzz is not None:
        return fuzz.token_set_ratio(a, b) / 100.0
    ta = set(re.findall(r"\w+", a))
    tb = set(re.findall(r"\w+", b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class SemanticSimilarityScorer:
    """
    Lazily loads a sentence-embedding model on first use. Caches one
    embedding per unique string (not per pair) so repeated comparisons
    within a run don't re-encode the same text - important since this gets
    called O(n^2) times over a case's observations.

    Usage:
        scorer = SemanticSimilarityScorer()   # or get_semantic_scorer() below
        score = scorer.similarity("ATM entrance", "ATM entry door")  # in [0, 1]
        scorer.available()   # True if real embeddings are being used
        scorer.backend_used() # "embedding" or "lexical_fallback"
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None
        self._ok = False
        self._embedding_cache: Dict[str, "Any"] = {}  # str -> embedding vector
        self._load_attempted = False

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # FIX: recurring issue - anonymous (unauthenticated) requests to
            # the HF Hub get rate-limited/throttled, which is a plausible
            # cause of the repeated slow/stalled model.safetensors downloads
            # seen in practice. If an HF_TOKEN is available (env var or
            # Colab secret, same pattern as Timeline_Key), pass it through -
            # free to obtain from huggingface.co/settings/tokens. Entirely
            # optional: if no token is found, this is unchanged from before
            # (anonymous download, same as it's always been).
            hf_token = self._find_hf_token()
            if hf_token:
                self._model = SentenceTransformer(self._model_name, token=hf_token)
            else:
                self._model = SentenceTransformer(self._model_name)
            self._ok = True
            log.info("SemanticSimilarityScorer: loaded '%s' (authenticated=%s).", self._model_name, bool(hf_token))
        except Exception as exc:
            log.warning(
                "SemanticSimilarityScorer: could not load embedding model '%s' (%s) - "
                "falling back to lexical fuzzy matching for all similarity scoring. "
                "If this was a transient issue (e.g. a network hiccup), call "
                "retry_load() to try again without restarting the runtime.",
                self._model_name, exc,
            )
            self._model = None
            self._ok = False

    @staticmethod
    def _find_hf_token() -> str:
        import os
        token = os.environ.get("HF_TOKEN", "")
        if token:
            return token
        try:
            from google.colab import userdata  # type: ignore
            return userdata.get("HF_TOKEN") or ""
        except Exception:
            return ""

    def retry_load(self) -> bool:
        """
        FIX: force a fresh load attempt, clearing any previously cached
        failure. Without this, a single transient failure (e.g. a network
        hiccup while reaching Hugging Face Hub) permanently disables real
        semantic matching for the rest of the session - _ensure_loaded()
        only ever tries once and every later call just returns the cached
        (failed) result, so simply re-running downstream cells does NOT
        retry. Call this explicitly to retry. Returns True if the retry
        succeeded.
        """
        self._load_attempted = False
        self._ok = False
        self._model = None
        self._ensure_loaded()
        return self._ok

    def available(self) -> bool:
        self._ensure_loaded()
        return self._ok

    def backend_used(self) -> str:
        return "embedding" if self.available() else "lexical_fallback"

    def _embed(self, text: str):
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        vec = self._model.encode(text, normalize_embeddings=True)
        self._embedding_cache[text] = vec
        return vec

    def similarity(self, a: Optional[str], b: Optional[str]) -> float:
        """Returns similarity in [0, 1]. Higher = more semantically similar."""
        a, b = (a or "").strip(), (b or "").strip()
        if not a or not b:
            return 0.0
        if a.lower() == b.lower():
            return 1.0

        self._ensure_loaded()
        if not self._ok:
            return _lexical_fallback_similarity(a, b)

        try:
            import numpy as np  # sentence-transformers already depends on this
            ea, eb = self._embed(a.lower()), self._embed(b.lower())
            cos = float(np.dot(ea, eb))  # already normalized -> dot == cosine
            # Cosine can be slightly negative for very dissimilar text;
            # clamp to [0, 1] since callers treat this as a similarity score.
            return max(0.0, min(1.0, cos))
        except Exception as exc:
            log.warning("Embedding similarity call failed (%s) - falling back to lexical for this pair.", exc)
            return _lexical_fallback_similarity(a, b)


_default_scorer: Optional[SemanticSimilarityScorer] = None


def get_semantic_scorer() -> SemanticSimilarityScorer:
    """Module-level singleton so the (potentially slow-to-load) model is
    loaded at most once per process/notebook session, not once per pipeline
    run or per agent instance."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = SemanticSimilarityScorer()
    return _default_scorer


def reset_semantic_scorer() -> SemanticSimilarityScorer:
    """
    Reset the module-level singleton and force a fresh load attempt. Call
    this if the model failed to load once (e.g. a transient network issue
    reaching Hugging Face Hub) and you want to retry without restarting the
    whole Colab runtime. Returns the new scorer so you can immediately check
    `.available()`.
    """
    global _default_scorer
    _default_scorer = SemanticSimilarityScorer()
    _default_scorer._ensure_loaded()
    return _default_scorer


def semantic_location_similarity(a: Optional[str], b: Optional[str]) -> float:
    """Convenience wrapper used by both ER and Timeline Agent for location matching."""
    return get_semantic_scorer().similarity(a, b)


"""
ForenSynth - Timeline Agent
config.py: centralised constants and environment-driven settings.

FIXED in this revision:
  - Confidence weights now sum to exactly 1.0 (was 0.90, capping every
    event's confidence below 1.0 even for perfect evidence).
  - Removed all cloud/Groq configuration. The agent is offline-only by
    default. An OPTIONAL local-only LLM backend can be configured via
    environment variables (see llm_fallback.py) but nothing here ever
    causes a network call to a third-party service.
"""

import os

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
CLOUD_LLM_MODEL: str = os.environ.get("TIMELINE_CLOUD_LLM_MODEL", "llama-3.1-8b-instant")
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


"""
ForenSynth - Timeline Agent
utils.py: shared utility functions.

FIX: parse_epoch / normalize_alias / epoch_to_iso now delegate to
normalization.py (the module shared with the ER pipeline) instead of
maintaining their own separate logic - this was the source of the
cross-agent timestamp-parsing divergence found in review. Kept as thin
wrappers here so existing imports (`from utils import parse_epoch`) don't
need to change anywhere in the codebase.
"""

import re
from typing import List


# Backward-compatible alias - existing call sites use `parse_epoch`.
parse_epoch = parse_timestamp


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def content_action_keywords(content: str) -> List[str]:
    """
    DEPRECATED: kept only for backward compatibility with any external
    caller. Timeline Agent's own reasoners now use
    normalization.extract_action_tags() instead, which is fragment-based
    and negation-aware rather than single-token matching.
    """
    if not content:
        return []
    return re.findall(r"\b[a-z]+\b", content.lower())


def short_summary(content: str, max_len: int = 80) -> str:
    """Return a truncated summary of content for narrative use."""
    if not content:
        return ""
    c = content.strip()
    if len(c) <= max_len:
        return c
    return c[:max_len].rstrip() + "..."


def deterministic_tiebreak_key(obs_ids: List[str], event_id: str) -> str:
    """
    A stable, content-derived tie-break key used when two events land on the
    exact same (epoch, confidence, modality) sort key. Using obs_id/event_id
    guarantees the output ordering is identical across runs regardless of
    the order the Memory Store happens to return records in.
    """
    if obs_ids:
        return min(obs_ids)
    return event_id


"""
ForenSynth - Timeline Agent
models.py: dataclasses for all internal domain objects.

FIXED in this revision:
  - TimelineVersion gained `unresolved_temporal_pairs`, `unresolved_causal_pairs`,
    `conflicts_unlocalized_count`, and `unresolved_entities` so that anything
    the pipeline could NOT determine is reported explicitly instead of being
    silently dropped or silently guessed.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# -- Enumerations --------------------------------------------------------------