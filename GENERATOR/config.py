"""
ForenSynth-X+ Configuration
Global settings, noise parameters, and role-modality probability tables.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NoiseConfig:
    """Configurable noise injection parameters."""
    temporal_noise_min_sec: int = 3
    temporal_noise_max_sec: int = 8
    missing_modality_rate: float = 0.20       # Drop ~20% of observations
    contradiction_rate: float = 0.075         # 5–10% contradictions
    temporal_noise_rate: float = 0.60         # Apply temporal noise to 60% of obs
    semantic_variation_rate: float = 0.40     # Paraphrase 40% of content


@dataclass
class GeneratorConfig:
    """Top-level generator configuration."""
    seed: Optional[int] = None
    min_events: int = 6
    max_events: int = 10
    base_datetime: str = "2024-01-15T10:00:00"  # ISO anchor for timestamps
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    # --- LLM enrichment via Cohere (optional) ---
    # enrich=True triggers a single API call that rewrites:
    #   fir.description, fir.location, observations[].content, observations[].source
    # Ground truth, timestamps, noise_tags, and aliases are never touched.
    enrich: bool = False
    cohere_api_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation location lookup
# Domain + modality + source_label → spatial location string
# This gives every observation a structured spatial anchor independent of
# content and source, enabling the Timeline Agent to group by location
# and the Critique Agent to flag same-location contradictions.
# ---------------------------------------------------------------------------

OBSERVATION_LOCATIONS: dict[str, dict[str, dict[str, str]]] = {
    "ATM_Robbery": {
        "video": {
            "camera_1":       "ATM booth entrance, exterior facing",
            "camera_2":       "ATM kiosk exterior, street-side view",
            "camera_3":       "ATM street frontage, wide-angle coverage",
            "cctv_entrance":  "ATM lobby doorway, entry/exit point",
            "cctv_atm":       "ATM booth interior, card reader and keypad area",
        },
        "audio": {
            "mic_booth":          "ATM booth interior",
            "phone_record":       "near ATM location, mobile network",
            "witness_statement":  "ATM vicinity, bystander position",
            "intercepted_call":   "ATM vicinity, intercepted channel",
        },
        "text": {
            "email_log":          "remote — email server log",
            "sms_record":         "remote — mobile network SMS",
            "complaint_register": "local police station, complaint desk",
            "incident_report":    "bank security office, incident registry",
        },
    },
    "Office_Theft": {
        "video": {
            "camera_1":       "main office corridor, ground floor",
            "camera_2":       "server room entrance, restricted zone",
            "camera_3":       "accounts department area, third floor",
            "cctv_entrance":  "office building main lobby, entry point",
            "cctv_atm":       "cash handling area, internal office",
        },
        "audio": {
            "mic_booth":          "security monitoring room, office premises",
            "phone_record":       "office premises, seized device",
            "witness_statement":  "office floor, colleague position",
            "intercepted_call":   "office internal network, PBX system",
        },
        "text": {
            "email_log":          "corporate email server, IT infrastructure",
            "sms_record":         "off-premises — accused personal device",
            "complaint_register": "HR department, complaint registry",
            "incident_report":    "facilities security office, incident log",
        },
    },
    "Communication": {
        "video": {
            "camera_1":       "suspect traced location, public area",
            "camera_2":       "near suspect known address, street camera",
            "camera_3":       "device active area, public CCTV coverage",
            "cctv_entrance":  "premises linked to communication, entry point",
            "cctv_atm":       "ATM area near suspect last known location",
        },
        "audio": {
            "mic_booth":          "monitored premises, interception unit",
            "phone_record":       "remote — unregistered SIM, mobile network",
            "witness_statement":  "witness location, third-party premises",
            "intercepted_call":   "network interception point, surveillance node",
        },
        "text": {
            "email_log":          "remote — email server, cyber forensics extract",
            "sms_record":         "remote — seized or monitored device",
            "complaint_register": "police station, complaint registry",
            "incident_report":    "cyber cell office, incident registry",
        },
    },
}

_LOCATION_FALLBACK = "location unspecified"


def get_observation_location(domain: str, modality: str, source: str) -> str:
    """
    Return a spatial location string for the given domain, modality, and source label.
    Falls back gracefully if any key is missing.
    """
    return (
        OBSERVATION_LOCATIONS
        .get(domain, {})
        .get(modality, {})
        .get(source, _LOCATION_FALLBACK)
    )


# Role → Modality probability weights
# Keys: "video", "audio", "text"
ROLE_MODALITY_WEIGHTS: dict[str, dict[str, float]] = {
    "suspect": {"video": 0.70, "audio": 0.50, "text": 0.40},
    "witness": {"video": 0.25, "audio": 0.75, "text": 0.50},
    "system":  {"video": 0.00, "audio": 0.00, "text": 1.00},
}

# Camera / source labels per modality
SOURCE_LABELS: dict[str, list[str]] = {
    "video": ["camera_1", "camera_2", "camera_3", "cctv_entrance", "cctv_atm"],
    "audio": ["mic_booth", "phone_record", "witness_statement", "intercepted_call"],
    "text":  ["email_log", "sms_record", "complaint_register", "incident_report"],
}
