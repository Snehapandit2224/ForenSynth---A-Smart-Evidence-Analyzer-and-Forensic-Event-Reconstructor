"""
Static configuration for the video agent.

Everything here is copied from (or derived from) the case generator in
GENERATOR_FIXED/, so extracted observations use exactly the same vocabulary
as the synthetic cases:

  * VIDEO_LOCATIONS     <- config.OBSERVATION_LOCATIONS[domain]["video"]
  * VIDEO_QUALITY_BANK  <- noise._CONTRADICTION_VIDEO_QUALITY
  * atm_video_vocabulary.json <- every ATM template's description_templates,
    grouped by (action, role). The Timeline agent's ACTION_TAG_FRAGMENTS were
    built from these same phrases, so reusing them keeps action tagging intact.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULT_MODEL = "gemini-3.8-flash"     # free tier in Google AI Studio; change if Google renames it
DEFAULT_FPS = 4                        # frames/s the model sees; enough to read a ticking clock
DEFAULT_ANCHOR = "2024-01-15T10:00:00" # generator base_datetime: time_offset counts from here
SANITY_WINDOW_HOURS = (-1, 12)         # overlay times outside anchor-1h .. anchor+12h are flagged

# Case ID prefix -> domain (CASE_ATM_001 -> ATM_Robbery), mirroring the generator's make_case_id
CASE_PREFIX_TO_DOMAIN = {"ATM": "ATM_Robbery", "OFFICE": "Office_Theft", "OFF": "Office_Theft",
                         "COMM": "Communication", "COM": "Communication"}


def domain_from_case_id(case_id):
    parts = case_id.upper().split("_")
    for part in parts[1:] if parts[0] == "CASE" else parts:
        if part in CASE_PREFIX_TO_DOMAIN:
            return CASE_PREFIX_TO_DOMAIN[part]
    return DEFAULT_DOMAIN
DEFAULT_DOMAIN = "ATM_Robbery"

VIDEO_SOURCES = ["camera_1", "camera_2", "camera_3", "cctv_entrance", "cctv_atm"]

VIDEO_LOCATIONS = {
    "ATM_Robbery": {
        "camera_1":      "ATM booth entrance, exterior facing",
        "camera_2":      "ATM kiosk exterior, street-side view",
        "camera_3":      "ATM street frontage, wide-angle coverage",
        "cctv_entrance": "ATM lobby doorway, entry/exit point",
        "cctv_atm":      "ATM booth interior, card reader and keypad area",
    },
    "Office_Theft": {
        "camera_1":      "main office corridor, ground floor",
        "camera_2":      "server room entrance, restricted zone",
        "camera_3":      "accounts department area, third floor",
        "cctv_entrance": "office building main lobby, entry point",
        "cctv_atm":      "cash handling area, internal office",
    },
}
LOCATION_FALLBACK = "location unspecified"

# What each camera looks like, used only when the on-screen label can't be read.
CAMERA_VIEW_TO_SOURCE = {
    "exterior_above_entrance": "camera_1",
    "exterior_street_side":    "camera_2",
    "wide_street_frontage":    "camera_3",
    "lobby_doorway":           "cctv_entrance",
    "interior_card_reader":    "cctv_atm",
}

# On-screen camera labels -> source. Matched after uppercasing and removing
# spaces, dashes and underscores. Edit to match whatever your clips burn in.
OVERLAY_LABEL_TO_SOURCE = {
    "CAM01": "camera_1", "CAM1": "camera_1", "CAMERA1": "camera_1",
    "CAM02": "camera_2", "CAM2": "camera_2", "CAMERA2": "camera_2",
    "CAM03": "camera_3", "CAM3": "camera_3", "CAMERA3": "camera_3",
    "ENT": "cctv_entrance", "ENTRANCE": "cctv_entrance", "LOBBY": "cctv_entrance",
    "CCTVENTRANCE": "cctv_entrance",
    "ATMINT": "cctv_atm", "ATM": "cctv_atm", "CCTVATM": "cctv_atm", "INT": "cctv_atm",
}

# Used verbatim as content when the footage itself is the finding.
VIDEO_QUALITY_BANK = {
    "angle_unclear":          "Camera angle unclear; individual's identity uncertain.",
    "low_light":              "Footage inconclusive due to lighting conditions.",
    "feed_interrupted":       "Visual feed interrupted; action not fully captured.",
    "low_quality":            "Recording quality insufficient to confirm the individual's identity.",
    "obstructed":             "Camera obstructed at the relevant timestamp; sequence unclear.",
    "frame_drop":             "Frame rate drop during this segment renders the footage unreliable.",
    "face_not_visible":       "Subject's face not visible in available footage.",
    "timestamp_inconsistent": "Video timestamp metadata inconsistent with reported incident time.",
}

WITNESS_ACTIONS = {"observe_exit", "report_incident"}

# Joint actions. The generator records two people acting together as ONE observation with a
# plural sentence. A sentence is plural when its SUBJECT is ("Two individuals ...", "Both suspects
# ...", "The two suspects ..."); a witness sentence such as "Passerby ... watches two people leave"
# has one actor and is not. Some joint actions are filed under another action key: two people
# leaving together is flee_scene, while exit_atm only has single-person sentences.
PLURAL_SUBJECT = re.compile(r"^(the )?(two|both)\b", re.I)
GROUP_ACTION = {"exit_atm": "flee_scene"}
# A witness of one of these must be declared as staying behind (see tools.validate): a "watcher"
# who then walks off with the person they watched is a group member, the usual mislabel.
LEAVING_ACTIONS = {"exit_atm", "flee_scene"}


def is_plural(sentence):
    return bool(sentence and PLURAL_SUBJECT.match(sentence.strip()))


def plural_sentences(vocab):
    """action -> indexes of its plural sentences (only actions that have any)."""
    out = {a: [i for i, s in enumerate(v["templates"]) if is_plural(s)] for a, v in vocab.items()}
    return {a: idx for a, idx in out.items() if idx}

# Confidence = clamp(W_CERT*certainty + W_VIS*visibility + W_QUAL*quality - penalties)
# then optionally a*x + b from calibration.json (written by evaluate.py --calibrate).
CONF_WEIGHTS = {"certainty": 0.45, "visibility": 0.25, "quality": 0.30}
CONF_PENALTY = {"footage_issue": 0.25, "timestamp_unread": 0.10, "unidentifiable": 0.10}
CONF_RANGE = (0.10, 0.99)


def load_vocabulary(domain: str = DEFAULT_DOMAIN):
    if domain != "ATM_Robbery":
        raise SystemExit(f"Only ATM_Robbery vocabulary is bundled (got {domain}).")
    with open(HERE / "atm_video_vocabulary.json", encoding="utf-8") as f:
        rows = json.load(f)
    # action -> {"role": ..., "templates": [...]}
    return {r["action"]: {"role": r["role"], "templates": r["templates"]} for r in rows}
