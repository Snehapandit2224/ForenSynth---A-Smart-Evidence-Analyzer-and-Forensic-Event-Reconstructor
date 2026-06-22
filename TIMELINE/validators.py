"""
ForenSynth – Timeline Agent
validators.py: input validation with descriptive error messages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class ValidationError(Exception):
    """Raised when the Timeline Agent input fails schema checks."""


def _check_field(obj: Dict[str, Any], field: str, expected_type: type, path: str) -> None:
    if field not in obj:
        raise ValidationError(f"[{path}] Missing required field: '{field}'")
    if not isinstance(obj[field], expected_type):
        raise ValidationError(
            f"[{path}.{field}] Expected {expected_type.__name__}, "
            f"got {type(obj[field]).__name__}"
        )


def _validate_observation(obs: Any, idx: int) -> List[str]:
    """Validate a single raw observation dict. Returns list of warnings (non-fatal)."""
    path = f"obs_only.observations[{idx}]"
    warnings: List[str] = []
    if not isinstance(obs, dict):
        raise ValidationError(f"[{path}] Observation must be a dict, got {type(obs).__name__}")
    if "obs_id" not in obs:
        raise ValidationError(f"[{path}] Missing 'obs_id'")
    if "entity" not in obs:
        raise ValidationError(f"[{path}] Missing 'entity'")
    if "timestamp" not in obs:
        warnings.append(f"[{path}] Missing 'timestamp' — ordering will be uncertain")
    confidence = obs.get("confidence")
    if confidence is not None:
        try:
            c = float(confidence)
            if not 0.0 <= c <= 1.0:
                warnings.append(f"[{path}] confidence={c} out of [0,1] range — will be clamped")
        except (TypeError, ValueError):
            warnings.append(f"[{path}] Non-numeric confidence '{confidence}' — will default to 0.5")
    return warnings


def _validate_canonical_entity(ent: Any, idx: int) -> None:
    path = f"entity_resolved.canonical_entities[{idx}]"
    if not isinstance(ent, dict):
        raise ValidationError(f"[{path}] Entity must be a dict, got {type(ent).__name__}")
    for field in ("entity_id", "primary_alias", "aliases"):
        if field not in ent:
            raise ValidationError(f"[{path}] Missing required field '{field}'")
    if not isinstance(ent["aliases"], list):
        raise ValidationError(f"[{path}] 'aliases' must be a list")


def validate_input(payload: Any) -> Tuple[str, List[str]]:
    """
    Validate the full Timeline Agent input payload.

    Returns:
        (case_id, warnings)

    Raises:
        ValidationError on fatal issues.
    """
    if not isinstance(payload, dict):
        raise ValidationError("Input payload must be a JSON object (dict)")

    # ── case_id ──────────────────────────────────────────────────────────────
    case_id = payload.get("case_id")
    if not case_id or not isinstance(case_id, str) or not case_id.strip():
        raise ValidationError("'case_id' must be a non-empty string")
    case_id = case_id.strip()

    # ── obs_only section ──────────────────────────────────────────────────────
    obs_only = payload.get("obs_only")
    if obs_only is None:
        raise ValidationError("'obs_only' section is missing")
    if not isinstance(obs_only, dict):
        raise ValidationError("'obs_only' must be a dict")
    observations = obs_only.get("observations")
    if observations is None:
        raise ValidationError("'obs_only.observations' is missing")
    if not isinstance(observations, list):
        raise ValidationError("'obs_only.observations' must be a list")
    if len(observations) == 0:
        raise ValidationError("'obs_only.observations' is empty — nothing to process")

    warnings: List[str] = []
    for i, obs in enumerate(observations):
        warnings.extend(_validate_observation(obs, i))

    # ── entity_resolved section ───────────────────────────────────────────────
    er = payload.get("entity_resolved")
    if er is None:
        raise ValidationError("'entity_resolved' section is missing")
    if not isinstance(er, dict):
        raise ValidationError("'entity_resolved' must be a dict")
    canonical_entities = er.get("canonical_entities")
    if canonical_entities is None:
        raise ValidationError("'entity_resolved.canonical_entities' is missing")
    if not isinstance(canonical_entities, list):
        raise ValidationError("'entity_resolved.canonical_entities' must be a list")
    if len(canonical_entities) == 0:
        warnings.append(
            "No canonical entities found — all observations will be treated as unresolved"
        )
    for i, ent in enumerate(canonical_entities):
        _validate_canonical_entity(ent, i)

    # conflicts_detected is optional / may be int or list
    conflicts = er.get("conflicts_detected", [])
    if not isinstance(conflicts, (list, int)):
        warnings.append(
            f"'entity_resolved.conflicts_detected' has unexpected type "
            f"{type(conflicts).__name__} — treating as empty"
        )

    return case_id, warnings
