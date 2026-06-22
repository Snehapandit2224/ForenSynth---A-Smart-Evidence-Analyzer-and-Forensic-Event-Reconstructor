"""
ForenSynth – Timeline Agent
temporal_reasoner.py: deterministic-first temporal ordering and relationship inference.

Design: Rule-based logic resolves the majority of cases.
Grok LLM is called ONCE in batch only for truly ambiguous pairs.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from config import (
    CAUSAL_ACTION_RULES,
    MODALITY_RELIABILITY,
    SIMULTANEOUS_WINDOW_SEC,
)
from grok_client import GrokClient
from models import EdgeType, TemporalRelation, TimelineEdge, TimelineEvent
from utils import content_action_keywords

log = logging.getLogger("forensynth.timeline.temporal_reasoner")


def _modality_reliability(modality: str) -> float:
    return MODALITY_RELIABILITY.get(modality.lower(), 0.50)


def _sorted_events_key(ev: TimelineEvent) -> Tuple[float, float, float]:
    """
    Sort key: (epoch ASC, confidence DESC, modality_reliability DESC).
    Ties broken by highest-confidence, most-reliable-modality first.
    """
    return (
        ev.ts_epoch if ev.ts_epoch > 0 else float("inf"),
        -ev.confidence,
        -_modality_reliability(ev.modality),
    )


class TemporalReasoner:
    """
    Stage 5 – Temporal Reasoning.

    1. Sort events by timestamp / confidence / modality.
    2. Infer TEMPORAL edges deterministically where possible.
    3. Batch-call Grok for unresolved ambiguous pairs (at most 1 call).
    """

    def __init__(self, grok_client: GrokClient) -> None:
        self._grok = grok_client

    def sort_events(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Return a new list sorted chronologically."""
        return sorted(events, key=_sorted_events_key)

    def build_temporal_edges(
        self, events: List[TimelineEvent]
    ) -> List[TimelineEdge]:
        """
        Build TEMPORAL edges between consecutive events in the sorted list.
        Also detect SIMULTANEOUS pairs.
        """
        edges: List[TimelineEdge] = []
        sorted_evs = self.sort_events(events)

        # Index by event_id for quick lookup
        ev_by_id: Dict[str, TimelineEvent] = {e.event_id: e for e in sorted_evs}

        # Build sequential temporal edges
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

        # Both have valid epochs
        if a.ts_epoch > 0 and b.ts_epoch > 0:
            gap = b.ts_epoch - a.ts_epoch
            if abs(gap) <= SIMULTANEOUS_WINDOW_SEC:
                return TemporalRelation.SIMULTANEOUS, 0.90
            if gap > 0:
                return TemporalRelation.BEFORE, 0.95
            return TemporalRelation.AFTER, 0.88

        # Only a has epoch
        if a.ts_epoch > 0 and b.ts_epoch <= 0:
            return TemporalRelation.BEFORE, 0.60

        # Only b has epoch
        if a.ts_epoch <= 0 and b.ts_epoch > 0:
            return TemporalRelation.BEFORE, 0.55

        # Neither has epoch – check causal action rules for ordering hints
        relation = self._rule_based_action_order(a.content, b.content)
        if relation != TemporalRelation.UNKNOWN:
            return relation, 0.65

        return TemporalRelation.UNKNOWN, 0.35

    def _rule_based_action_order(
        self, content_a: str, content_b: str
    ) -> TemporalRelation:
        """
        Apply CAUSAL_ACTION_RULES to determine ordering from action keywords.
        Returns BEFORE if a prerequisite of b is found in a, AFTER if vice-versa.
        """
        tokens_a = set(content_action_keywords(content_a))
        tokens_b = set(content_action_keywords(content_b))

        for prereq, dependent in CAUSAL_ACTION_RULES:
            if prereq in tokens_a and dependent in tokens_b:
                return TemporalRelation.BEFORE
            if prereq in tokens_b and dependent in tokens_a:
                return TemporalRelation.AFTER

        return TemporalRelation.UNKNOWN

    def resolve_ambiguous_with_llm(
        self,
        ambiguous_pairs: List[Tuple[TimelineEvent, TimelineEvent]],
    ) -> Dict[str, Tuple[TemporalRelation, float]]:
        """
        Batch-call Grok ONCE to resolve temporally ambiguous event pairs.
        Returns dict of (event_a.event_id, event_b.event_id) → (relation, confidence).
        """
        if not ambiguous_pairs or not self._grok.available:
            return {}

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

        log.info("Calling Grok for temporal resolution of %d ambiguous pairs.", len(payloads))
        results = self._grok.infer_temporal_relations_batch(payloads)

        # Map results back by index
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
                resolved[key] = (rel, float(r.get("confidence", 0.5)))
            else:
                resolved[key] = (TemporalRelation.UNKNOWN, 0.35)

        return resolved
