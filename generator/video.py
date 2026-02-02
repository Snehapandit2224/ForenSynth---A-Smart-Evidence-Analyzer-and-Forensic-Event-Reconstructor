'''import yaml
import json
from collections import OrderedDict

# =============================
# CONFIG
# =============================
INPUT_FILE = "raw_virat.yml"
OUTPUT_FILE = "normalized_virat.json"

SCENARIO_ID = "VIRAT_S_000001"
SOURCE_NAME = "VIRAT"
FPS = 30.0  # change if your VIRAT video uses different FPS


# =============================
# NORMALIZE SINGLE EVENT
# =============================
def normalize_event(act_block, event_index):
    """
    Convert one VIRAT 'act' block into a normalized event
    """

    # ---- action & confidence ----
    act2 = act_block.get("act2")
    if not act2:
        return None

    action, confidence = list(act2.items())[0]

    # ---- time (frame-based in VIRAT) ----
    timespan = act_block.get("timespan")
    if not timespan:
        return None

    ts = timespan[0].get("tsr0")
    if not ts or len(ts) != 2:
        return None

    start_frame, end_frame = ts

    # ---- actors ----
    raw_actors = act_block.get("actors", [])
    actors = []

    for a in raw_actors:
        actor_id = a.get("id1")
        if actor_id is not None:
            actors.append(f"A{actor_id}")

    if not actors:
        return None

    return OrderedDict({
        "event_id": f"E{event_index}",
        "action": action,
        "actors": actors,
        "start": {
            "frame": int(start_frame),
            "time_sec": round(start_frame / FPS, 3)
        },
        "end": {
            "frame": int(end_frame),
            "time_sec": round(end_frame / FPS, 3)
        },
        "confidence": float(confidence)
    })


# =============================
# NORMALIZE FULL YAML FILE
# =============================
def normalize_virat_yaml(raw_data):
    events = []
    actor_set = set()
    event_idx = 1

    for entry in raw_data:
        # Skip meta entries safely
        if "act" not in entry:
            continue

        event = normalize_event(entry["act"], event_idx)
        if event:
            events.append(event)
            event_idx += 1

            for a in event["actors"]:
                actor_set.add(a)

    # ---- actors list ----
    actors = []
    for actor_id in sorted(actor_set):
        actors.append({
            "actor_id": actor_id,
            "type": "person"  # safe default; vehicle inference can be added later
        })

    normalized_scene = OrderedDict({
        "scenario_id": SCENARIO_ID,
        "source": SOURCE_NAME,
        "fps": FPS,
        "actors": actors,
        "events": events
    })

    return normalized_scene


# =============================
# MAIN
# =============================
def main():
    with open(INPUT_FILE, "r") as f:
        raw_data = yaml.safe_load(f)

    normalized = normalize_virat_yaml(raw_data)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(normalized, f, indent=2)

    print("✅ Normalization complete")
    print(f"• Events extracted : {len(normalized['events'])}")
    print(f"• Actors extracted : {len(normalized['actors'])}")
    print(f"• FPS used         : {FPS}")
    print(f"• Output saved to  : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
'''

import yaml
import json
import re

# =========================
# CONFIG
# =========================
INPUT_YML = "raw_virat.yml"     # raw VIRAT file
OUTPUT_FILE = "mvp_video_events.jsonl"
FPS = 30.0


# =========================
# HELPERS
# =========================
def normalize_action(action: str) -> str:
    action = action.strip()
    action = action.replace(" ", "_")
    action = re.sub(r"[^a-zA-Z0-9_]", "", action)
    return action.lower()


def extract_mid_time_sec(timespan):
    """
    Extract midpoint time (seconds) from VIRAT timespan
    """
    # timespan: [{ tsr0: [start, end] }]
    frame_range = list(timespan[0].values())[0]
    start, end = frame_range
    return ((start + end) / 2) / FPS


def extract_actors(actor_list):
    """
    Extract all actor IDs
    """
    return [f"Person_{a['id1']}" for a in actor_list]


# =========================
# CORE PIPELINE
# =========================
def build_mvp_from_raw():
    with open(INPUT_YML, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    events = []
    event_counter = 1

    for item in raw_data:
        if "act" not in item:
            continue

        act = item["act"]

        # ---- action name ----
        act_name = list(act["act2"].keys())[0]

        # ---- actors ----
        entities = extract_actors(act.get("actors", []))

        # ---- time ----
        time_sec = extract_mid_time_sec(act["timespan"])

        # ---- confidence ----
        confidence = list(act["act2"].values())[0]

        mvp_event = {
            "event_id": f"EVT_{event_counter:06d}",
            "entities": entities,
            "action": normalize_action(act_name),
            "time": {
                "type": "relative",
                "value_sec": time_sec
            },
            "location": None,
            "modality": "video",
            "confidence": confidence,
            "source": {
                "dataset": "VIRAT",
                "ref": f"act_{act['id2']}"
            }
        }

        events.append(mvp_event)
        event_counter += 1

    # ---- write JSONL ----
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for e in events:
            out.write(json.dumps(e) + "\n")

    print("✅ Raw VIRAT → MVP conversion complete")
    print(f"📦 Total events: {len(events)}")
    print(f"📁 Output file: {OUTPUT_FILE}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    build_mvp_from_raw()
