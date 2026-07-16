"""
ForenSynth-X+ Entities
Canonical entity creation and modality-specific alias generation.
"""

import random
from dataclasses import dataclass, field

from config import ROLE_MODALITY_WEIGHTS


@dataclass
class CanonicalEntity:
    """
    Represents a true actor in the case (hidden ground truth).

    Attributes:
        entity_id: Human-readable role key, e.g. "suspect_1".
        role: One of "suspect", "witness", "system".
        aliases: Dict mapping modality → alias label visible in observations.
    """
    entity_id: str
    role: str
    aliases: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "role": self.role,
            "aliases": self.aliases,
        }


# ---------------------------------------------------------------------------
# Alias label pools
# ---------------------------------------------------------------------------

_VIDEO_PERSON_COUNTER: dict[str, int] = {}
_AUDIO_SPEAKER_COUNTER: dict[str, int] = {}
_TEXT_EMAIL_COUNTER: dict[str, int] = {}


def _unique_video_label(rng: random.Random, used: set[str]) -> str:
    while True:
        num = rng.randint(1, 99)
        label = f"Person_{num:02d}"
        if label not in used:
            return label


def _unique_audio_label(rng: random.Random, used: set[str]) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    while True:
        letter = rng.choice(letters)
        label = f"Speaker_{letter}"
        if label not in used:
            return label


def _unique_text_label(rng: random.Random, used: set[str]) -> str:
    prefixes = ["email", "sms", "report", "log"]
    while True:
        num = rng.randint(10, 99)
        prefix = rng.choice(prefixes)
        label = f"{prefix}_{num}"
        if label not in used:
            return label


def build_entities(
    suspect_count: int,
    witness_count: int,
    rng: random.Random,
) -> list[CanonicalEntity]:
    """
    Build canonical entities for all roles, assigning modality aliases
    probabilistically per ROLE_MODALITY_WEIGHTS.

    Args:
        suspect_count: Number of suspects.
        witness_count: Number of witnesses.
        rng: Seeded random instance.

    Returns:
        List of CanonicalEntity objects.
    """
    entities: list[CanonicalEntity] = []
    used_video: set[str] = set()
    used_audio: set[str] = set()
    used_text: set[str] = set()

    role_specs = (
        [("suspect", i + 1) for i in range(suspect_count)]
        + [("witness", i + 1) for i in range(witness_count)]
    )

    for role, idx in role_specs:
        entity_id = f"{role}_{idx}"
        weights = ROLE_MODALITY_WEIGHTS[role]
        aliases: dict[str, str] = {}

        # Each modality is independently sampled
        if rng.random() < weights["video"]:
            label = _unique_video_label(rng, used_video)
            used_video.add(label)
            aliases["video"] = label

        if rng.random() < weights["audio"]:
            label = _unique_audio_label(rng, used_audio)
            used_audio.add(label)
            aliases["audio"] = label

        if rng.random() < weights["text"]:
            label = _unique_text_label(rng, used_text)
            used_text.add(label)
            aliases["text"] = label

        # Guarantee at least one modality per entity
        if not aliases:
            dominant = max(weights, key=lambda k: weights[k])
            if dominant == "video":
                label = _unique_video_label(rng, used_video)
                used_video.add(label)
            elif dominant == "audio":
                label = _unique_audio_label(rng, used_audio)
                used_audio.add(label)
            else:
                label = _unique_text_label(rng, used_text)
                used_text.add(label)
            aliases[dominant] = label

        entities.append(CanonicalEntity(entity_id=entity_id, role=role, aliases=aliases))

    return entities


def build_entity_mapping(entities: list[CanonicalEntity]) -> dict[str, str]:
    """
    Build a flat alias → canonical entity_id mapping.

    Example:
        {"Person_17": "suspect_1", "Speaker_B": "suspect_1", ...}
    """
    mapping: dict[str, str] = {}
    for entity in entities:
        for alias in entity.aliases.values():
            mapping[alias] = entity.entity_id
    return mapping
