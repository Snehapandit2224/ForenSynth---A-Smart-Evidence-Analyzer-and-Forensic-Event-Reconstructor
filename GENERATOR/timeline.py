"""
ForenSynth-X+ Timeline
Ground truth event timeline generation from templates and canonical entities.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from entities import CanonicalEntity
from templates import Template, EventSlot


@dataclass
class GroundTruthEvent:
    """
    A single event in the hidden ground truth timeline.

    Attributes:
        event_id: Sequential event label ("E1", "E2", …).
        slot_id: Template slot this event corresponds to.
        entity_id: Canonical entity performing the action.
        role: Role of that entity.
        action: Canonical action verb.
        timestamp: ISO 8601 datetime string.
        time_offset: Seconds from the start of the time window.
    """
    event_id: str
    slot_id: str
    entity_id: str
    role: str
    action: str
    timestamp: str
    time_offset: int

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "entity_id": self.entity_id,
            "role": self.role,
            "action": self.action,
            "timestamp": self.timestamp,
            "time_offset": self.time_offset,
        }


def _resolve_entity(
    slot: EventSlot,
    entities: list[CanonicalEntity],
    rng: random.Random,
) -> CanonicalEntity | None:
    """
    Find the canonical entity that matches the slot's role and role_index.
    Returns None if no suitable entity exists.
    """
    candidates = [e for e in entities if e.role == slot.role]
    if not candidates:
        return None

    if slot.role_index is not None:
        if slot.role_index < len(candidates):
            return candidates[slot.role_index]
        # role_index out of range → pick last available
        return candidates[-1]

    return rng.choice(candidates)


def generate_timeline(
    template: Template,
    entities: list[CanonicalEntity],
    fir: dict,
    base_datetime_str: str,
    rng: random.Random,
) -> list[GroundTruthEvent]:
    """
    Instantiate the template's event slots into a coherent, ordered timeline.

    Args:
        template: The selected Template object.
        entities: List of CanonicalEntity objects for this case.
        fir: The FIR dict (used for time_window bounds).
        base_datetime_str: ISO anchor datetime string.
        rng: Seeded random instance.

    Returns:
        Ordered list of GroundTruthEvent objects.
    """
    base_dt = datetime.fromisoformat(base_datetime_str)
    time_window_end: int = fir["time_window"][1]

    events: list[GroundTruthEvent] = []
    current_offset: int = rng.randint(10, 60)  # slight lead-in
    event_counter: int = 1

    for slot in template.slots:
        # Skip optional slots with ~30% probability
        if not slot.required and rng.random() < 0.30:
            continue

        entity = _resolve_entity(slot, entities, rng)
        if entity is None:
            continue  # No matching entity — skip slot

        delta = rng.randint(slot.min_offset_delta, slot.max_offset_delta)
        current_offset += delta

        # Clamp to time window
        if current_offset > time_window_end:
            current_offset = time_window_end - rng.randint(5, 30)
            if current_offset < 0:
                current_offset = time_window_end

        event_dt = base_dt + timedelta(seconds=current_offset)

        events.append(GroundTruthEvent(
            event_id=f"E{event_counter}",
            slot_id=slot.slot_id,
            entity_id=entity.entity_id,
            role=entity.role,
            action=slot.action,
            timestamp=event_dt.isoformat(),
            time_offset=current_offset,
        ))
        event_counter += 1

    return events
