"""
ForenSynth – Timeline Agent
causal_reasoner.py: deterministic + optional LLM-based causal inference.

Strategy:
 1. Reject pairs that cross the actor/observer boundary (a witness's
    observation cannot be "caused by" a suspect's earlier action — the
    witness merely reported it; that is correlation, not causation).
 2. Apply CAUSAL_ACTION_RULES (deterministic).
 3. Score by temporal proximity + shared entity + shared location.
 4. Only send unresolved pairs to Grok as a single batch (at most 1 LLM call).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from config import (
    CAUSAL_ACTION_RULES,
    CAUSAL_WINDOW_SEC,
)
from grok_client import GrokClient
from models import EdgeType, TemporalRelation, TimelineEdge, TimelineEvent
from utils import clamp, content_action_keywords

log = logging.getLogger("forensynth.timeline.causal_reasoner")

# Roles that merely *report* events rather than *perform* them.
# An event whose role is in this set can never be the causal TARGET of
# another entity's action — it can only be a corroborating observation.
OBSERVER_ROLES = {"witness", "bystander", "reporter"}


class CausalReasoner:
    """
    Stage 6 – Causal Reasoning.

    Produces CAUSAL edges in the timeline graph.
    """

    def __init__(self, grok_client: GrokClient) -> None:
        self._grok = grok_client

    def infer_causal_links(
        self, events: List[TimelineEvent]
    ) -> List[TimelineEdge]:
        """
        Main entry point.  Returns a list of CAUSAL edges.
        """
        if len(events) < 2:
            return []

        causal_edges: List[TimelineEdge] = []
        unresolved_pairs: List[Tuple[TimelineEvent, TimelineEvent]] = []

        # Consider all ordered pairs within the causal window
        ev_sorted = sorted(events, key=lambda e: (e.ts_epoch if e.ts_epoch > 0 else float("inf")))

        for i, a in enumerate(ev_sorted):
            for b in ev_sorted[i + 1:]:
                if a.ts_epoch > 0 and b.ts_epoch > 0:
                    gap = b.ts_epoch - a.ts_epoch
                    if gap > CAUSAL_WINDOW_SEC:
                        break  # further events even further away
                    if gap < 0:
                        continue  # b before a — skip in this pass

                # Hard guard: never treat a different actor's action as the
                # cause of a witness/observer's report. Witnesses corroborate;
                # they are not causally downstream of the suspect's actions.
                if not self._eligible_for_causal_link(a, b):
                    continue

                # Attempt deterministic causal classification
                result = self._deterministic_causal(a, b)
                if result is not None:
                    edge, _ = result
                    causal_edges.append(edge)
                else:
                    unresolved_pairs.append((a, b))

        # Batch-resolve unresolved pairs with Grok (at most 1 API call)
        if unresolved_pairs and self._grok.available:
            llm_edges = self._llm_causal_resolve(unresolved_pairs)
            causal_edges.extend(llm_edges)

        return causal_edges

    def _eligible_for_causal_link(self, a: TimelineEvent, b: TimelineEvent) -> bool:
        """
        Returns False when a causal link a → b would be a category error:

        - Different entities where b is an observer/witness role: b's event
          is an independent report ABOUT a, not something a's action caused.
          (Person_14 walking up to an ATM does not "cause" a bystander to
          witness an exit — the bystander's testimony is corroboration.)

        Same-entity pairs and same-role pairs are always eligible; only
        cross-entity, cross-role (actor → observer) links are blocked here.
        """
        if a.entity_id == b.entity_id:
            return True  # same actor's own sequence of actions — always fine

        b_role = (b.role or "").strip().lower()
        a_role = (a.role or "").strip().lower()

        if b_role in OBSERVER_ROLES and a_role not in OBSERVER_ROLES:
            # a is an actor (suspect/system/etc.), b is a witness reporting on
            # the scene — block actor-action → witness-report causal claims.
            return False

        return True

    def _deterministic_causal(
        self, a: TimelineEvent, b: TimelineEvent
    ) -> Tuple[TimelineEdge, float] | None:
        """
        Return a CAUSAL edge if a → b can be determined by rules.
        Returns None if unresolved.
        """
        tokens_a = set(content_action_keywords(a.content))
        tokens_b = set(content_action_keywords(b.content))

        # Rule 1: Action dependency
        for prereq, dependent in CAUSAL_ACTION_RULES:
            if prereq in tokens_a and dependent in tokens_b:
                confidence = self._causal_confidence(a, b, base=0.82)
                edge = TimelineEdge(
                    source=a.event_id,
                    target=b.event_id,
                    edge_type=EdgeType.CAUSAL,
                    confidence=confidence,
                    relation=TemporalRelation.BEFORE,
                    label=f"action dependency: {prereq}→{dependent}",
                )
                return edge, confidence

        # Rule 2: Same entity, same location, sequential
        if (
            a.entity_id == b.entity_id
            and a.location and b.location
            and a.location.lower().strip() == b.location.lower().strip()
            and a.ts_epoch > 0 and b.ts_epoch > 0
            and 0 < (b.ts_epoch - a.ts_epoch) <= CAUSAL_WINDOW_SEC
        ):
            confidence = self._causal_confidence(a, b, base=0.68)
            edge = TimelineEdge(
                source=a.event_id,
                target=b.event_id,
                edge_type=EdgeType.CAUSAL,
                confidence=confidence,
                relation=TemporalRelation.BEFORE,
                label="same entity, same location, sequential",
            )
            return edge, confidence

        return None

    def _causal_confidence(
        self, a: TimelineEvent, b: TimelineEvent, base: float
    ) -> float:
        """
        Combine base causal rule confidence with contextual signals.
        """
        score = base

        # Boost for same entity
        if a.entity_id == b.entity_id:
            score += 0.06

        # Boost for same location
        if (
            a.location and b.location
            and a.location.lower().strip() == b.location.lower().strip()
        ):
            score += 0.04

        # Decay for large temporal gaps
        if a.ts_epoch > 0 and b.ts_epoch > 0:
            gap = b.ts_epoch - a.ts_epoch
            if gap > CAUSAL_WINDOW_SEC / 2:
                score -= 0.08

        # Factor in observation confidence
        score *= (a.confidence + b.confidence) / 2.0

        return clamp(score)

    def _llm_causal_resolve(
        self, pairs: List[Tuple[TimelineEvent, TimelineEvent]]
    ) -> List[TimelineEdge]:
        """Single batched Grok call for unresolved causal pairs."""
        payloads = [
            {
                "id":          idx,
                "a_content":   a.content,
                "a_timestamp": a.timestamp,
                "a_entity":    a.primary_alias,
                "a_role":      a.role,
                "a_location":  a.location,
                "b_content":   b.content,
                "b_timestamp": b.timestamp,
                "b_entity":    b.primary_alias,
                "b_role":      b.role,
                "b_location":  b.location,
            }
            for idx, (a, b) in enumerate(pairs)
        ]

        log.info("Calling Grok for causal resolution of %d ambiguous pairs.", len(payloads))
        results = self._grok.infer_causal_relations_batch(payloads)

        result_by_id = {r["id"]: r for r in results}
        edges: List[TimelineEdge] = []
        for idx, (a, b) in enumerate(pairs):
            info = result_by_id.get(idx)
            if info and info.get("causal", False):
                explanation = info.get("explanation", "LLM inferred causal link")
                confidence = clamp(float(info.get("confidence", 0.50)))
                edges.append(TimelineEdge(
                    source=a.event_id,
                    target=b.event_id,
                    edge_type=EdgeType.CAUSAL,
                    confidence=confidence,
                    relation=TemporalRelation.BEFORE,
                    label=f"LLM: {explanation}",
                ))
        return edges