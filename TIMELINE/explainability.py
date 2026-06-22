"""
ForenSynth – Timeline Agent
explainability.py: generates ExplainabilityRecords and human-readable narrative.
"""
from __future__ import annotations

import logging
from typing import List

from models import (
    ExplainabilityRecord,
    NarrativeLine,
    TimelineEdge,
    TimelineEvent,
    UncertaintyRecord,
)
from utils import short_summary

log = logging.getLogger("forensynth.timeline.explainability")


class ExplainabilityLayer:
    """
    Stage 10 / Stage 11 – Explainability and Narrative generation.
    """

    def build_explainability(
        self, events: List[TimelineEvent]
    ) -> List[ExplainabilityRecord]:
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
        self,
        events: List[TimelineEvent],
        causal_edges: List[TimelineEdge],
    ) -> List[NarrativeLine]:
        """
        Generate one NarrativeLine per event, in chronological order.
        """
        narrative: List[NarrativeLine] = []

        # Build a set of events that are causal targets (for annotation)
        causal_targets = {e.target: e.label for e in causal_edges if e.label}

        for ev in events:
            action = short_summary(ev.content, max_len=100)
            if not action:
                action = f"{ev.primary_alias} observed at {ev.location or 'unknown location'}"

            causal_note = causal_targets.get(ev.event_id, "")
            if causal_note:
                action = f"{action}  [↳ {causal_note}]"

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

    def build_uncertainties(
        self, events: List[TimelineEvent]
    ) -> List[UncertaintyRecord]:
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
        """
        Render the narrative as a human-readable forensic text report.
        """
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
