"""
ForenSynth-X+ Utils
Shared utility functions for ID generation, JSON export, and validation.
"""

import json
import random
import string
from pathlib import Path


def make_case_id(domain: str, index: int) -> str:
    """Generate a deterministic case ID string."""
    domain_short = domain.replace("_", "")[:3].upper()
    return f"CASE_{domain_short}_{index:03d}"


def seed_rng(seed: int | None) -> random.Random:
    """Return a seeded Random instance (or unseeded if seed is None)."""
    rng = random.Random()
    if seed is not None:
        rng.seed(seed)
    return rng


def export_case(case: dict, output_path: str | Path) -> None:
    """Serialise a case dict to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)


def validate_case_schema(case: dict) -> list[str]:
    """
    Lightweight schema validation.
    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []
    required_top_keys = {"case_id", "domain", "template", "fir", "observations", "ground_truth"}
    for key in required_top_keys:
        if key not in case:
            errors.append(f"Missing top-level key: '{key}'")

    if "observations" in case:
        for i, obs in enumerate(case["observations"]):
            for field in ("obs_id", "entity", "role", "modality", "source", "location", "content", "timestamp", "confidence"):
                if field not in obs:
                    errors.append(f"Observation {i} missing field: '{field}'")
            if "canonical_entity" in obs:
                errors.append(f"Observation {i} leaks 'canonical_entity' — ground truth exposed!")

    if "ground_truth" in case:
        gt = case["ground_truth"]
        for key in ("entities", "events", "entity_mapping"):
            if key not in gt:
                errors.append(f"ground_truth missing key: '{key}'")

    return errors


def pretty_print_case(case: dict) -> None:
    """Print case to stdout with indentation."""
    print(json.dumps(case, indent=2, ensure_ascii=False))


def extract_observations_only(case: dict) -> dict:
    """
    Return a stripped observation-only document from a case file.

    Drops 'event_ref' and 'noise_tags' from every observation so the
    downstream model sees only raw evidence without ground-truth linkage
    or noise metadata.

    Returns a dict with: case_id, domain, template, fir, observations
    (event_ref and noise_tags removed from each observation).
    """
    keep_obs_fields = {
        "obs_id", "entity", "role", "modality",
        "source", "location", "content", "timestamp",
        "time_offset", "confidence",
    }
    stripped_obs = [
        {k: v for k, v in obs.items() if k in keep_obs_fields}
        for obs in case.get("observations", [])
    ]
    return {
        "case_id": case["case_id"],
        "domain": case["domain"],
        "template": case["template"],
        "fir": case["fir"],
        "observations": stripped_obs,
    }


def export_observations_only(case: dict, output_path) -> None:
    """
    Serialise the stripped observation-only document to a JSON file.

    Drops 'event_ref' and 'noise_tags' so the model input contains
    only raw evidence fields. Writes a companion file alongside the
    full case file, typically suffixed '_obs_only.json'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obs_doc = extract_observations_only(case)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obs_doc, f, indent=2, ensure_ascii=False)