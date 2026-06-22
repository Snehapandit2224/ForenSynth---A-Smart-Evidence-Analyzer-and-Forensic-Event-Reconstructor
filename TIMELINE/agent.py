"""
ForenSynth – Timeline Agent
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
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from causal_reasoner import CausalReasoner
from config import (
    DEFAULT_OUTPUT_DIR,
    GRAPH_EXPORT_FILENAME,
    MODALITY_RELIABILITY,
    TIMELINE_SCHEMA_VERSION,
    WEIGHT_CONFLICT_PENALTY,
    WEIGHT_ENTITY_RESOLUTION,
    WEIGHT_OBS_CONFIDENCE,
    WEIGHT_TEMPORAL_CERTAINTY,
)
from explainability import ExplainabilityLayer
from graph_builder import GraphBuilder
from grok_client import GrokClient
from models import (
    CanonicalEntity,
    EdgeType,
    RawObservation,
    TemporalRelation,
    TimelineEdge,
    TimelineEvent,
    TimelineVersion,
)
from repositories import (
    JsonEntityRepository,
    JsonObservationRepository,
    JsonTimelineRepository,
)
from temporal_reasoner import TemporalReasoner
from utils import clamp, epoch_to_iso, normalize_alias, parse_epoch, utc_now_iso
from validators import ValidationError, validate_input

log = logging.getLogger("forensynth.timeline.agent")

_EV_ID_RE = re.compile(r"[^a-z0-9_]")


def _make_event_id(obs_ids: List[str], entity_id: str) -> str:
    key = f"{entity_id}_{'_'.join(sorted(obs_ids))}"
    return "EVT_" + _EV_ID_RE.sub("_", key.lower())[:48]


class TimelineAgent:
    """
    ForenSynth Timeline Agent V1.

    Usage::

        agent = TimelineAgent()
        result = agent.run(payload_dict)
        print(json.dumps(result, indent=2))
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        save_outputs: bool = True,
    ) -> None:
        self._grok = GrokClient()
        self._temporal = TemporalReasoner(self._grok)
        self._causal = CausalReasoner(self._grok)
        self._graph_builder = GraphBuilder()
        self._explainability = ExplainabilityLayer()
        self._timeline_repo = JsonTimelineRepository(output_dir)
        self._output_dir = Path(output_dir)
        self._save_outputs = save_outputs

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full 13-stage timeline construction pipeline.

        Args:
            payload: Combined input dict with 'case_id', 'obs_only', 'entity_resolved'.

        Returns:
            Serialised TimelineVersion dict.

        Raises:
            ValidationError: on malformed input.
        """
        t_start = time.perf_counter()
        stage_timings: Dict[str, float] = {}

        # ── Stage 1: Input Validation ─────────────────────────────────────────
        t0 = time.perf_counter()
        case_id, warnings = validate_input(payload)
        for w in warnings:
            log.warning("Validation warning: %s", w)
        stage_timings["stage_1_validation"] = time.perf_counter() - t0
        log.info("Stage 1 complete – case_id=%s, %d warning(s)", case_id, len(warnings))

        # ── Repositories ──────────────────────────────────────────────────────
        obs_repo = JsonObservationRepository(payload["obs_only"]["observations"])
        er = payload["entity_resolved"]
        ent_repo = JsonEntityRepository(er.get("canonical_entities", []))

        raw_observations = obs_repo.get_all(case_id)
        canonical_entities = ent_repo.get_all(case_id)
        conflicts_raw = er.get("conflicts_detected", [])

        log.info(
            "Loaded %d observations, %d canonical entities",
            len(raw_observations),
            len(canonical_entities),
        )

        # ── Stage 2: Event Enrichment ─────────────────────────────────────────
        t0 = time.perf_counter()
        events, obs_conflict_set = self._stage_2_enrich(
            case_id, raw_observations, canonical_entities, ent_repo, conflicts_raw
        )
        stage_timings["stage_2_enrichment"] = time.perf_counter() - t0
        log.info("Stage 2 complete – %d events created", len(events))

        # ── Stage 3: Timestamp-First Ordering ─────────────────────────────────
        t0 = time.perf_counter()
        events = self._temporal.sort_events(events)
        stage_timings["stage_3_ordering"] = time.perf_counter() - t0
        log.info("Stage 3 complete – events sorted")

        # ── Stage 4 + 5: Event Graph + Temporal Reasoning ─────────────────────
        t0 = time.perf_counter()
        temporal_edges = self._temporal.build_temporal_edges(events)

        # Identify ambiguous temporal pairs for potential LLM resolution
        ambiguous = [
            (
                next(e for e in events if e.event_id == ed.source),
                next(e for e in events if e.event_id == ed.target),
            )
            for ed in temporal_edges
            if ed.relation == TemporalRelation.UNKNOWN
        ]
        if ambiguous:
            resolved = self._temporal.resolve_ambiguous_with_llm(ambiguous)
            # Patch edges with LLM-resolved relations
            for ed in temporal_edges:
                key = f"{ed.source}|{ed.target}"
                if key in resolved:
                    ed.relation, ed.confidence = resolved[key]
                    ed.label = f"{ed.relation.value} ({ed.confidence:.2f}) [LLM]"

        stage_timings["stage_4_5_temporal"] = time.perf_counter() - t0
        log.info("Stage 4/5 complete – %d temporal edges", len(temporal_edges))

        # ── Stage 6: Causal Reasoning ─────────────────────────────────────────
        t0 = time.perf_counter()
        causal_edges = self._causal.infer_causal_links(events)
        stage_timings["stage_6_causal"] = time.perf_counter() - t0
        log.info("Stage 6 complete – %d causal edges", len(causal_edges))

        # ── Stage 7: Uncertainty Modelling ───────────────────────────────────
        t0 = time.perf_counter()
        # (confidence already computed per-event in stage 2; uncertainties extracted in stage 10)
        stage_timings["stage_7_uncertainty"] = time.perf_counter() - t0

        # ── Stage 8: Conflict Awareness ───────────────────────────────────────
        t0 = time.perf_counter()
        conflicts_summary = self._stage_8_conflicts(conflicts_raw, events, obs_conflict_set)
        stage_timings["stage_8_conflicts"] = time.perf_counter() - t0
        log.info("Stage 8 complete – %d conflict entries", len(conflicts_summary))

        # ── Stage 9: Provenance (guaranteed by enrichment + event structure) ──
        # (obs_ids on every event = provenance; verified here)
        for ev in events:
            if not ev.obs_ids:
                log.error(
                    "PROVENANCE VIOLATION: event %s has no source observations!", ev.event_id
                )

        # ── Stage 10: Explainability ──────────────────────────────────────────
        t0 = time.perf_counter()
        explainability = self._explainability.build_explainability(events)
        uncertainties = self._explainability.build_uncertainties(events)
        stage_timings["stage_10_explainability"] = time.perf_counter() - t0

        # ── Stage 11: Narrative ───────────────────────────────────────────────
        t0 = time.perf_counter()
        narrative = self._explainability.build_narrative(events, causal_edges)
        stage_timings["stage_11_narrative"] = time.perf_counter() - t0

        # ── Stage 12: Versioning ──────────────────────────────────────────────
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
        )
        stage_timings["stage_12_versioning"] = time.perf_counter() - t0

        # ── Stage 13: Graph Export ────────────────────────────────────────────
        t0 = time.perf_counter()
        if self._save_outputs:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            graph_path = self._output_dir / GRAPH_EXPORT_FILENAME
            self._graph_builder.export_to_file(G, str(graph_path))
            self._timeline_repo.save(timeline)
        stage_timings["stage_13_export"] = time.perf_counter() - t0

        total_time = time.perf_counter() - t_start
        log.info("Pipeline complete in %.3fs", total_time)

        result = timeline.to_dict()
        result["stage_timings"] = {k: round(v, 4) for k, v in stage_timings.items()}
        result["total_time_sec"] = round(total_time, 4)
        result["validation_warnings"] = warnings
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 – Event Enrichment
    # ─────────────────────────────────────────────────────────────────────────

    def _stage_2_enrich(
        self,
        case_id: str,
        raw_observations: List[RawObservation],
        canonical_entities: List[CanonicalEntity],
        ent_repo: JsonEntityRepository,
        conflicts_raw: Any,
    ) -> Tuple[List[TimelineEvent], Set[str]]:
        """
        Convert raw observations → TimelineEvent objects.

        Each observation that maps to a canonical entity produces one event.
        Observations without a canonical entity match get a synthetic fallback entity.

        Returns (events, conflict_obs_id_set).
        """

        # Build obs-id → canonical entity map from ER output
        obs_to_entity: Dict[str, CanonicalEntity] = {}
        for ent in canonical_entities:
            for obs_id in ent.sources:
                obs_to_entity[obs_id] = ent

        # Identify obs_ids involved in conflicts
        conflict_obs_ids: Set[str] = self._collect_conflict_obs_ids(conflicts_raw)

        # Determine conflict-affected entities
        conflict_entity_ids: Set[str] = set()
        for obs_id in conflict_obs_ids:
            ent = obs_to_entity.get(obs_id)
            if ent:
                conflict_entity_ids.add(ent.entity_id)

        events: List[TimelineEvent] = []
        seen_event_ids: Set[str] = set()

        for obs in raw_observations:
            ent = obs_to_entity.get(obs.obs_id)

            # Parse timestamp
            ts_epoch = parse_epoch(obs.timestamp)
            if ts_epoch <= 0 and obs._ts_epoch > 0:
                ts_epoch = obs._ts_epoch
            timestamp = epoch_to_iso(ts_epoch) if ts_epoch > 0 else obs.timestamp

            # Entity fields
            if ent:
                entity_id = ent.entity_id
                primary_alias = ent.primary_alias
                aliases = list(ent.aliases)
                er_confidence = ent.confidence_score
            else:
                # Fallback: treat observation as its own unresolved entity
                entity_id = f"UNRESOLVED_{normalize_alias(obs.entity)}"
                primary_alias = obs.entity
                aliases = [obs.entity]
                er_confidence = 0.40

            # Compute event confidence (Stage 7 model embedded here)
            in_conflict = (
                obs.obs_id in conflict_obs_ids
                or (ent and ent.entity_id in conflict_entity_ids)
            )
            temporal_certainty = 1.0 if ts_epoch > 0 else 0.50
            conflict_penalty = WEIGHT_CONFLICT_PENALTY if in_conflict else 0.0

            confidence = clamp(
                obs.confidence * WEIGHT_OBS_CONFIDENCE
                + er_confidence * WEIGHT_ENTITY_RESOLUTION
                + temporal_certainty * WEIGHT_TEMPORAL_CERTAINTY
                - conflict_penalty
            )

            # Reasoning chain
            reasoning: List[str] = ["timestamp ordering"]
            if ent:
                reasoning.append("canonical entity match")
                if len(ent.aliases) > 1:
                    reasoning.append("alias cluster resolved")
            else:
                reasoning.append("unresolved entity – fallback")
            if in_conflict:
                reasoning.append("conflict flagged by entity resolution")
            if ts_epoch > 0:
                reasoning.append("valid timestamp")
            else:
                reasoning.append("timestamp missing or unparseable")

            # Build event
            event_id = _make_event_id([obs.obs_id], entity_id)
            # Guarantee uniqueness
            base = event_id
            suffix = 0
            while event_id in seen_event_ids:
                suffix += 1
                event_id = f"{base}_{suffix}"
            seen_event_ids.add(event_id)

            event = TimelineEvent(
                event_id=event_id,
                obs_ids=[obs.obs_id],          # provenance
                timestamp=timestamp,
                ts_epoch=ts_epoch,
                location=obs.location,
                entity_id=entity_id,
                primary_alias=primary_alias,
                aliases=aliases,
                modality=obs.modality,
                content=obs.content,
                confidence=confidence,
                role=obs.role or "unknown",
                conflict_flag=in_conflict,
                conflict_note=(
                    "Entity resolution conflict detected; confidence reduced."
                    if in_conflict else ""
                ),
                reasoning=reasoning,
                version="V1",
            )
            events.append(event)

        return events, conflict_obs_ids

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 8 – Conflict Awareness
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_conflict_obs_ids(self, conflicts_raw: Any) -> Set[str]:
        """Extract all observation IDs mentioned in ER conflict records."""
        ids: Set[str] = set()
        if isinstance(conflicts_raw, list):
            for c in conflicts_raw:
                if isinstance(c, dict):
                    for key in ("obs_ids", "members", "affected_obs"):
                        val = c.get(key, [])
                        if isinstance(val, list):
                            ids.update(str(v) for v in val)
        # int form → no obs-level information
        return ids

    def _stage_8_conflicts(
        self,
        conflicts_raw: Any,
        events: List[TimelineEvent],
        obs_conflict_set: Set[str],
    ) -> List[Dict[str, Any]]:
        """
        Translate ER conflict records into Timeline-Agent conflict summaries.
        """
        summaries: List[Dict[str, Any]] = []

        if isinstance(conflicts_raw, int):
            if conflicts_raw > 0:
                summaries.append({
                    "conflict_type": "entity_resolution_conflicts",
                    "count":         conflicts_raw,
                    "detail":        f"Entity Resolution reported {conflicts_raw} conflict(s).",
                    "affected_events": [
                        ev.event_id for ev in events if ev.conflict_flag
                    ],
                })
            return summaries

        if isinstance(conflicts_raw, list):
            for c in conflicts_raw:
                if not isinstance(c, dict):
                    continue
                affected_events = [
                    ev.event_id for ev in events
                    if ev.conflict_flag and any(o in c.get("obs_ids", []) for o in ev.obs_ids)
                ]
                summaries.append({
                    "conflict_type":     c.get("type", "unknown"),
                    "cluster_id":        c.get("cluster_id", ""),
                    "detail":            c.get("detail", ""),
                    "affected_events":   affected_events,
                })

        return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def run_timeline_agent(
    payload: Dict[str, Any],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Module-level convenience wrapper."""
    agent = TimelineAgent(output_dir=output_dir, save_outputs=save_outputs)
    return agent.run(payload)