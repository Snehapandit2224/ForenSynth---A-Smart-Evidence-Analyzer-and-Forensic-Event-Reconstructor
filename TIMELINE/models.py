"""
ForenSynth – Timeline Agent
models.py: Pydantic-free dataclasses for all internal domain objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class EdgeType(str, Enum):
    TEMPORAL = "TEMPORAL"
    CAUSAL   = "CAUSAL"
    INFERRED = "INFERRED"


class TemporalRelation(str, Enum):
    BEFORE       = "BEFORE"
    AFTER        = "AFTER"
    SIMULTANEOUS = "SIMULTANEOUS"
    UNKNOWN      = "UNKNOWN"


# ── Raw observation (mirrors ER pipeline output) ──────────────────────────────

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
    # Populated by ER normalisation (optional, may be absent in raw input)
    entity_norm:     str   = ""
    time_offset_sec: int   = 0
    _ts_epoch:       float = field(default=0.0, repr=False)


# ── Canonical entity from ER ──────────────────────────────────────────────────

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


# ── Timeline event (enriched observation) ────────────────────────────────────

@dataclass
class TimelineEvent:
    event_id:      str
    obs_ids:       List[str]          # provenance → raw observations
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

    # Explainability
    reasoning:     List[str] = field(default_factory=list)

    # Versioning hook – populated by TimelineVersion wrapper
    version:       str = "V1"


# ── Graph edge ────────────────────────────────────────────────────────────────

@dataclass
class TimelineEdge:
    source:     str       # event_id
    target:     str       # event_id
    edge_type:  EdgeType
    confidence: float
    relation:   TemporalRelation = TemporalRelation.BEFORE
    label:      str = ""


# ── Uncertainty record ────────────────────────────────────────────────────────

@dataclass
class UncertaintyRecord:
    event_id:          str
    uncertainty_score: float          # 1.0 - confidence
    sources:           List[str]      # obs_ids
    reasons:           List[str]


# ── Conflict record ───────────────────────────────────────────────────────────

@dataclass
class ConflictRecord:
    conflict_type: str
    cluster_id:    str
    detail:        str
    affected_obs:  List[str] = field(default_factory=list)


# ── Narrative line ─────────────────────────────────────────────────────────────

@dataclass
class NarrativeLine:
    timestamp:  str
    actor:      str
    action:     str
    location:   str
    evidence:   List[str]   # obs_ids
    confidence: float
    event_id:   str


# ── Explainability record ─────────────────────────────────────────────────────

@dataclass
class ExplainabilityRecord:
    event_id:    str
    derived_from: List[str]   # obs_ids
    entity_used: str          # canonical entity_id
    reasoning:   List[str]
    confidence:  float


# ── Timeline version wrapper ──────────────────────────────────────────────────

@dataclass
class TimelineVersion:
    version:          str              # "V1", "V2", …
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
                "entity_id":     ev.entity_id,
                "primary_alias": ev.primary_alias,
                "aliases":       ev.aliases,
                "modality":      ev.modality,
                "content":       ev.content,
                "confidence":    round(ev.confidence, 4),
                "role":          ev.role,
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
        }