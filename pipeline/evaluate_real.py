#!/usr/bin/env python3
"""
ForenSynth - score a pipeline run on REAL evidence against a hand-written
answer key (cases_real/<case>_answer_key.json).

The generator's ground truth was written for its own simplified world: one
person and one event per observation, the reporter as the actor. Real field
agents read the media more finely - a clip with a suspect leaving and a
witness watching becomes two observations; an email's "two individuals left"
gets the people leaving as actors and the author as reporter - and the
generator's key scores that as wrong. An answer key lists, for every
evidence item, everyone in it, the part they play, and the event they show.

Each merged observation is traced back to its evidence item by provenance
(video: camera label + time window; audio: file; text: source document),
then to the person in that item it is about (video: appearance / role hint;
text and audio: reporter vs the person described), and so to an event. The
result is written in the generator's ground-truth format and scored with the
same evaluators (evaluate.py, evaluate_er.py).

    python pipeline/evaluate_real.py --key cases_real/CASE_001_answer_key.json \\
        --observations <run>/_intake/<case>/<case>_obs_only.json \\
        --timeline <run>/timelines/<case>_timeline_V1.json --er <run>/er/<case>_er_v1_output.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_root = Path(__file__).parent.parent
for _p in [str(_root / "pipeline"), str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

WINDOW_SLACK_SEC = 5


def _ts(value: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _camera(label: str) -> str:
    """'CAM 01' / 'camera_1' / 'cam01' -> '1'."""
    digits = "".join(ch for ch in str(label or "") if ch.isdigit())
    return str(int(digits)) if digits else ""


def find_item(obs: Dict[str, Any], evidence: Dict[str, Any]) -> Optional[str]:
    """The evidence item an observation was extracted from."""
    for item_id, item in evidence.items():
        if item["modality"] != obs.get("modality"):
            continue
        if item["modality"] == "audio" and item.get("audio_file") and obs.get("audio_file") == item["audio_file"]:
            return item_id
        if item["modality"] == "text" and item.get("source_doc") and \
                str(obs.get("source_doc") or "").split("#")[0] == item["source_doc"]:
            return item_id
        if item["modality"] == "video":
            cam = _camera(obs.get("camera_label") or obs.get("source"))
            t, (start, end) = _ts(obs.get("timestamp")), (_ts(item["window"][0]), _ts(item["window"][1]))
            slack = timedelta(seconds=WINDOW_SLACK_SEC)
            if cam == _camera(item.get("camera")) and t and start - slack <= t <= end + slack:
                return item_id
    return None


def find_person(obs: Dict[str, Any], item: Dict[str, Any], people: Dict[str, Any]) -> Tuple[str, str]:
    """(person, event) within an item that the observation is about."""
    entries = item["people"]
    if len(entries) == 1:
        return entries[0]["person"], entries[0]["event"]
    by_part = {e["part"]: e for e in entries}
    if item["modality"] in ("text", "audio"):
        described = obs.get("reported_by") and obs.get("reported_by") != obs.get("entity")
        for part in (("described",) if described else ("author", "speaker")):
            if part in by_part:
                return by_part[part]["person"], by_part[part]["event"]
    # video with several people: by appearance, then by role hint
    from entity_resolution import EntityResolutionPipeline as ER
    pa = ER._parse_appearance(obs.get("appearance") or "")
    scored = []
    for e in entries:
        pb = ER._parse_appearance(people[e["person"]].get("match_appearance", people[e["person"]]["description"]))
        score, conflict = ER._compare_appearance(pa, pb)
        scored.append((0 if conflict else score, e))
    best = max(scored, key=lambda s: s[0])
    if best[0] > 0:
        return best[1]["person"], best[1]["event"]
    for e in entries:
        if obs.get("role_hint") == people[e["person"]]["role"]:
            return e["person"], e["event"]
    return entries[0]["person"], entries[0]["event"]


def build_ground_truth(key: Dict[str, Any], observations: List[Dict[str, Any]]) -> Tuple[Dict, List[str]]:
    obs_out, obs_entity, unplaced = [], {}, []
    for o in observations:
        item_id = find_item(o, key["evidence"])
        if item_id is None:
            unplaced.append(o["obs_id"])
            continue
        person, event = find_person(o, key["evidence"][item_id], key["people"])
        obs_out.append({**o, "event_ref": event, "evidence_item": item_id})
        obs_entity[o["obs_id"]] = person
    events = [{"event_id": e["event_id"], "entity_id": e["person"], "role": key["people"][e["person"]]["role"],
               "action": e["action"], "timestamp": e["time"]} for e in key["events"]]
    gt = {"case_id": key["case_id"], "observations": obs_out,
          "ground_truth": {"events": events, "entity_mapping": {}, "obs_entity": obs_entity,
                           "entities": [{"entity_id": p, "role": v["role"]} for p, v in key["people"].items()]}}
    return gt, unplaced


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a real-evidence run against an answer key")
    ap.add_argument("--key", required=True)
    ap.add_argument("--observations", required=True, help="the merged obs_only file the run used")
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--er", required=True)
    ap.add_argument("--output", default=str(_root / "output" / "evaluation_real"))
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    from evaluate import evaluate_case
    from evaluate_er import evaluate_er_file

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    obs = json.loads(Path(args.observations).read_text(encoding="utf-8"))["observations"]
    gt, unplaced = build_ground_truth(key, obs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gt_path = out / f"{key['case_id']}_answer_key_groundtruth.json"
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    print("observation -> evidence item -> person / event")
    for o in gt["observations"]:
        print(f"  {o['obs_id']:<4} {o['modality']:<5} {o['entity'][:28]:<28} -> {o['evidence_item']:<9} "
              f"-> {gt['ground_truth']['obs_entity'][o['obs_id']]} / {o['event_ref']}")
    if unplaced:
        print(f"  NOT PLACED (no matching evidence item): {unplaced}")

    ev = evaluate_case(key["case_id"], str(gt_path), args.timeline, str(out))
    er = evaluate_er_file(args.er, str(gt_path)) or {}
    print("\nscores against the answer key")
    for k in ("event_recall", "event_recall_observable", "precision", "precision_linked", "f1_score", "f1_pure", "event_purity", "n_conflated", "moment_precision", "moment_recall", "moment_f1", "moment_time_err_sec",
              "entity_accuracy", "action_tag_accuracy", "temporal_accuracy_sec"):
        print(f"  {k:<26} {ev.get(k)}")
    for k in ("er_precision", "er_recall", "er_recall_supported", "wrong_merges"):
        print(f"  {k:<26} {er.get(k)}")
    print("\nfindings an investigator should get (from the key):")
    for f in key.get("findings_an_investigator_should_get", []):
        print(f"  - [{f['kind']}] {f['detail']}")


if __name__ == "__main__":
    main()
