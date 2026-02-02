import json
import random
from copy import deepcopy
from datetime import datetime, timedelta
import dateutil.parser

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_FILE = "mvp_video.jsonl"   # JSONL
AUDIO_FILE = "mvp_audio.json"    # JSON
TEXT_FILE  = "mvp_text.json"     # JSON

OUTPUT_FILE = "generated_scenarios.json"

MIN_EVENTS_PER_SCENARIO = 5
MAX_EVENTS_PER_SCENARIO = 20

TEXT_DATE_MAX_OFFSET_DAYS = 2    # ± days for text date normalization
PERTURBATION_PROBABILITY = 0.4   # chance of perturbing a scenario

ENABLE_RANDOMNESS = True
RANDOM_SEED = None               # set int for reproducibility

# ============================================================
# RANDOM CONTROL
# ============================================================

if ENABLE_RANDOMNESS:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
else:
    random.seed(42)

# ============================================================
# LOADERS
# ============================================================

def load_video_events():
    events = []
    with open(VIDEO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                e["_source_modality"] = "video"
                events.append(e)
    return events


def load_json_events(path, modality):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for e in data:
            e["_source_modality"] = modality
        return data


def load_all_events():
    events = []
    events.extend(load_video_events())
    events.extend(load_json_events(AUDIO_FILE, "audio"))
    events.extend(load_json_events(TEXT_FILE, "text"))
    return events

# ============================================================
# SCENARIO BUILDING
# ============================================================

def build_scenarios(events):
    random.shuffle(events)
    scenarios = []
    idx = 0

    while idx < len(events):
        block_size = random.randint(
            MIN_EVENTS_PER_SCENARIO,
            MAX_EVENTS_PER_SCENARIO
        )
        block = events[idx:idx + block_size]
        idx += block_size

        if len(block) < MIN_EVENTS_PER_SCENARIO:
            break

        scenarios.append(block)

    return scenarios

# ============================================================
# PERTURBATION (OPTIONAL, LIGHT)
# ============================================================

def perturb_scenario(events):
    events = deepcopy(events)

    perturb_type = random.choice([
        "shuffle_order",
        "duplicate_event",
        "drop_entity",
        "none"
    ])

    if perturb_type == "shuffle_order":
        random.shuffle(events)

    elif perturb_type == "duplicate_event":
        events.append(deepcopy(random.choice(events)))

    elif perturb_type == "drop_entity":
        e = random.choice(events)
        if "entities" in e:
            e["entities"] = []

    return events, perturb_type

# ============================================================
# TEXT DATE NORMALIZATION (KEY REQUIREMENT)
# ============================================================


def normalize_text_dates_nearby(events, max_day_offset):
    """
    Normalize ONLY the DATE of text events to be near each other.
    Time-of-day (HH:MM:SS) and timezone are strictly preserved.
    """

    text_events = [e for e in events if e.get("modality") == "text"]

    if len(text_events) <= 1:
        return events

    # Choose anchor date from a random text event (not always first)
    anchor_event = random.choice(text_events)
    anchor_dt = datetime.fromisoformat(anchor_event["timestamp"])
    anchor_date = anchor_dt.date()

    for e in text_events:
        original_dt = datetime.fromisoformat(e["timestamp"])

        # Random nearby day offset
        day_offset = random.randint(-max_day_offset, max_day_offset)
        new_date = anchor_date + timedelta(days=day_offset)

        # IMPORTANT: preserve original time + tzinfo
        new_dt = datetime(
            year=new_date.year,
            month=new_date.month,
            day=new_date.day,
            hour=original_dt.hour,
            minute=original_dt.minute,
            second=original_dt.second,
            microsecond=original_dt.microsecond,
            tzinfo=original_dt.tzinfo
        )

        e["timestamp"] = new_dt.isoformat()

    return events



# ============================================================
# GENERATOR PIPELINE
# ============================================================

def generate():
    all_events = load_all_events()
    scenario_blocks = build_scenarios(all_events)

    output = []
    scenario_id = 1

    for block in scenario_blocks:
        if random.random() < PERTURBATION_PROBABILITY:
            events, perturb_type = perturb_scenario(block)
        else:
            events = deepcopy(block)
            perturb_type = None

        # Normalize TEXT dates only (your requirement)
        events = normalize_text_dates_nearby(
            events,
            TEXT_DATE_MAX_OFFSET_DAYS
        )

        output.append({
            "scenario_id": f"SCN_{scenario_id:03d}",
            "events": events,
            "metadata": {
                "synthetic": True,
                "event_count": len(events),
                "modalities": sorted(
                    {e["_source_modality"] for e in events}
                ),
                "perturbation": perturb_type,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        })

        scenario_id += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("✅ Scenario generation completed")
    print(f"📦 Total scenarios: {len(output)}")
    print(f"📁 Output file: {OUTPUT_FILE}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate()
