"""
ForenSynth-X+ Timeline
Ground truth event timeline generation from templates and canonical entities.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from entities import CanonicalEntity
from templates import Template, EventSlot


class TemplateRealizationError(Exception):
    """
    Raised when a required template slot cannot be instantiated due to missing entities.
    
    This indicates a structural problem: the template requires an actor that doesn't
    exist (or the entity count is too low).
    """
    pass


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

    Returns None only when no candidate of the required role exists at all
    (used for optional-slot graceful skipping).

    For required slots with a specific role_index, raises TemplateRealizationError
    when the index is out of range — never silently falls back to another entity.
    Falling back silently would cause a multi-actor template to reuse the same actor
    for two distinct roles, corrupting the ground truth.

    Args:
        slot: The EventSlot to resolve.
        entities: All CanonicalEntity objects for this case.
        rng: Seeded random instance (used only when role_index is None).

    Returns:
        The matching CanonicalEntity, or None if no candidate of the required role exists.

    Raises:
        TemplateRealizationError: If role_index is specified and out of range.
    """
    candidates = [e for e in entities if e.role == slot.role]
    if not candidates:
        return None

    if slot.role_index is not None:
        if slot.role_index < len(candidates):
            return candidates[slot.role_index]
        # role_index OOB: silent fallback would corrupt multi-actor ground truth.
        # Raise so the caller can surface the misconfiguration.
        raise TemplateRealizationError(
            f"Template slot '{slot.slot_id}' requires {slot.role}[{slot.role_index}], "
            f"but only {len(candidates)} {slot.role}(s) exist. "
            f"Entity count enforcement should have prevented this — check "
            f"_enforce_template_entity_counts logic."
        )

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
    
    Required slots must always be realized. If a required slot cannot find a matching
    entity, raises TemplateRealizationError instead of silently skipping it.

    Args:
        template: The selected Template object.
        entities: List of CanonicalEntity objects for this case.
        fir: The FIR dict (used for time_window bounds).
        base_datetime_str: ISO anchor datetime string.
        rng: Seeded random instance.

    Returns:
        Ordered list of GroundTruthEvent objects.
        
    Raises:
        TemplateRealizationError: If a required slot cannot be realized due to
            missing entities or insufficient role counts.
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
            # If this is a required slot, we have a structural problem
            if slot.required:
                available_roles = [e.role for e in entities]
                raise TemplateRealizationError(
                    f"Template '{template.name}' requires {slot.role} at index {slot.role_index} "
                    f"(slot '{slot.slot_id}'), but no such entity exists.\n"
                    f"  Available entities: {available_roles}\n"
                    f"  Required slot: {slot.action}\n"
                    f"This indicates the entity count was not enforced before timeline generation."
                )
            # If optional, we can skip it
            continue

        delta = rng.randint(slot.min_offset_delta, slot.max_offset_delta)
        prev_offset = current_offset
        current_offset += delta

        # Clamp to time window — distribute overflowing events across the final
        # 20% of the window rather than piling them all at the same offset.
        # IMPORTANT: lower bound must be >= current_offset to preserve monotonic
        # temporal ordering in the ground truth (clamped value must never go back).
        if current_offset > time_window_end:
            tail_start = int(time_window_end * 0.80)

            lower_bound = max(prev_offset + 1, tail_start)
            lower_bound = min(lower_bound, time_window_end)
            upper_bound = time_window_end

            if lower_bound <= upper_bound:
                current_offset = rng.randint(lower_bound, upper_bound)
            else:
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
