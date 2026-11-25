#!/usr/bin/env python3
"""
ForenSynth - score a pipeline run on raw evidence against the generator's
ground truth for that case.

The field agents renumber and re-cut the evidence (one clip can give two
observations, a message two events), so the ground truth's observation ids
don't line up with the run's. Each observation of the run is traced to the
ground-truth observation it came from:

    text   its source document: O8.json -> source_doc "O8#1" -> O8
    audio  its recording: the clip's manifest start_time -> the ground-truth
           audio observation at that time
    video  the video agent's own id (O5, O6b -> O6) when --video-logs is
           given, else the ground-truth video observation nearest in time
    FIR    no ground-truth observation (the generator's ground truth has no
           FIR narrative) - left out, and events made only of it are counted

then scored with the standard evaluators (evaluate.py, evaluate_er.py).

    python pipeline/evaluate_traced.py ^
        --ground-truth "Evidence Generation 2/Cases/Enriched_Case_001/CASE_ATM_001.json" ^
        --run output/enriched001 --case ENRICHED_CASE_001 ^
        --audio-dir "Evidence Generation 2/Cases/Enriched_Case_001/Case_001/Audio" ^
        --video-logs agents/field_agents/video_agent/runs/CASE_ATM_001
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = Path(__file__).parent.parent
for _p in [str(_root / "pipeline"), str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _ts(value: Any) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if t.tzinfo:
        from datetime import timezone
        t = t.astimezone(timezone.utc).replace(tzinfo=None)
    return t


def _latest(folder: Path, pattern: str) -> Path:
    files = sorted((p for p in folder.glob(pattern) if "_graph" not in p.name),
                   key=lambda p: int(re.search(r"_(?:V|er_v)(\d+)", p.name).group(1)))
    if not files:
        raise SystemExit(f"Nothing matching {pattern} in {folder}")
    return files[-1]


def trace(merged: Dict[str, Any], gt_case: Dict[str, Any], audio_dir: Optional[Path],
          video_logs: Optional[Path], audio_starts: Optional[Dict[str, str]] = None
          ) -> tuple[Dict[str, Any], List[tuple]]:
    """audio_starts: recording file name -> start time, when the start times
    were typed in (the dashboard) rather than read from manifests."""
    gobs = gt_case["observations"]
    gt = gt_case["ground_truth"]
    starts: Dict[str, datetime] = {Path(k).name: _ts(v) for k, v in (audio_starts or {}).items()}
    if audio_dir:
        for mf in Path(audio_dir).rglob("*anifest*.json"):
            for entry in json.loads(mf.read_text(encoding="utf-8-sig")):
                starts[Path(str(entry.get("path"))).name] = _ts(entry.get("start_time"))
    video_ids: Dict[tuple, str] = {}
    if video_logs:
        hits = sorted(Path(video_logs).rglob("*_video_obs.json"))
        if hits:
            for o in json.loads(hits[0].read_text(encoding="utf-8"))["observations"]:
                video_ids[(str(o.get("timestamp")), o.get("content"))] = o["obs_id"]

    def nearest(modality: str, t: Optional[datetime]) -> Optional[str]:
        pool = [g for g in gobs if g["modality"] == modality and _ts(g.get("timestamp"))]
        if not pool or t is None:
            return None
        return min(pool, key=lambda g: abs((_ts(g["timestamp"]) - t).total_seconds()))["obs_id"]

    ids = {g["obs_id"] for g in gobs}

    def source_of(o: Dict[str, Any]) -> Optional[str]:
        m = o.get("modality")
        if o.get("fir_derived"):
            return None
        if m == "text":
            doc = str(o.get("source_doc") or "").split("#")[0]
            return doc if doc in ids else None
        if m == "audio":
            return nearest("audio", starts.get(Path(str(o.get("audio_file") or "")).name) or _ts(o.get("timestamp")))
        if m == "video":
            own = video_ids.get((str(o.get("timestamp")), o.get("content")))
            base = re.match(r"^(O\d+)", own or "")
            if base and base.group(1) in ids:
                return base.group(1)
            return nearest("video", _ts(o.get("timestamp")))
        return None

    src = {g["obs_id"]: g for g in gobs}
    gt_obs, obs_entity, untraced = [], {}, []
    for o in merged["observations"]:
        s = src.get(source_of(o))
        if not s:
            untraced.append((o["obs_id"], o.get("modality"), str(o.get("entity"))[:30], bool(o.get("fir_derived"))))
            continue
        gt_obs.append({**o, "event_ref": s["event_ref"], "noise_tags": s.get("noise_tags", []), "traced_to": s["obs_id"]})
        obs_entity[o["obs_id"]] = (gt.get("obs_entity") or {}).get(s["obs_id"]) or gt["entity_mapping"].get(s["entity"])
    traced = {**{k: v for k, v in merged.items() if k != "observations"}, "observations": gt_obs,
              "ground_truth": {**gt, "obs_entity": obs_entity}}
    return traced, untraced


def score_run(run: Path, case: str, gt_case: Dict[str, Any], audio_dir: Optional[Path] = None,
              video_logs: Optional[Path] = None, audio_starts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Score one run (its --output folder and --case id) against the ground
    truth. Returns the scores, the tracing and the Critique's findings."""
    from evaluate import evaluate_case
    from evaluate_er import evaluate_er_file

    run = Path(run)
    merged_path = run / "_intake" / case / f"{case}_obs_only.json"
    if not merged_path.exists():
        raise FileNotFoundError(f"No merged case at {merged_path} - was the run's --output / --case the same?")
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    traced, untraced = trace(merged, gt_case, audio_dir, video_logs, audio_starts)
    out = run / "evaluation_traced"
    out.mkdir(parents=True, exist_ok=True)
    gt_path = out / f"{case}_traced_groundtruth.json"
    gt_path.write_text(json.dumps(traced, indent=2), encoding="utf-8")

    timeline = _latest(run / "timelines", f"{case}_timeline_V*.json")
    er = _latest(run / "er", f"{case}_er_v*_output.json")
    ev = evaluate_case(case, str(gt_path), str(timeline), str(out))
    erm = evaluate_er_file(str(er), str(gt_path)) or {}
    tl = json.loads(timeline.read_text(encoding="utf-8"))
    fir_ids = {o["obs_id"] for o in merged["observations"] if o.get("fir_derived")}
    fir_only = sum(1 for e in tl.get("events", []) if e.get("obs_ids") and all(x in fir_ids for x in e["obs_ids"]))
    crit = sorted((run / "critiques").glob(f"{case}_critique_C*.json"))
    findings = [{"check": g["check"], "detail": g["detail"]}
                for g in (json.loads(crit[-1].read_text(encoding="utf-8")).get("gaps", []) if crit else [])
                if g["check"] != "low_confidence_event"]
    return {
        "case_id": case, "timeline_file": timeline.name, "er_file": er.name,
        "scores": [
            {"metric": "True events (ground truth)", "value": ev.get("n_ground_truth_events")},
            {"metric": "Events output / moments", "value": f"{ev.get('n_pipeline_events')} / {ev.get('n_moments')}"},
            {"metric": "Event recall", "value": ev.get("event_recall")},
            {"metric": "Event recall (observable)", "value": ev.get("event_recall_observable")},
            {"metric": "Event precision", "value": ev.get("precision")},
            {"metric": "F1 per event", "value": ev.get("f1_score")},
            {"metric": "Moment precision", "value": ev.get("moment_precision")},
            {"metric": "Moment recall", "value": ev.get("moment_recall")},
            {"metric": "F1 per moment", "value": ev.get("moment_f1")},
            {"metric": "Entity accuracy", "value": ev.get("entity_accuracy")},
            {"metric": "Action-tag accuracy", "value": ev.get("action_tag_accuracy")},
            {"metric": "Timing error mean / median (s)",
             "value": f"{ev.get('temporal_accuracy_sec')} / {ev.get('temporal_error_median_sec')}"},
            {"metric": "Duplicate events", "value": ev.get("n_duplicates")},
            {"metric": "ER precision", "value": erm.get("er_precision")},
            {"metric": "ER recall", "value": erm.get("er_recall")},
            {"metric": "ER recall (supported links)", "value": erm.get("er_recall_supported")},
            {"metric": "Events made only of FIR claims", "value": fir_only},
        ],
        "wrong_merges": erm.get("wrong_merges") or [],
        "missed": ev.get("unmatched_gt_event_ids") or [],
        "half_credit": ev.get("covered_gt_event_ids") or [],
        "unobservable": ev.get("unobservable_gt_event_ids") or [],
        "traced": [{"obs_id": o["obs_id"], "modality": o["modality"], "entity": str(o["entity"]),
                    "traced_to": o["traced_to"], "event": o["event_ref"],
                    "person": traced["ground_truth"]["obs_entity"].get(o["obs_id"])}
                   for o in traced["observations"]],
        "untraced": [{"obs_id": i, "modality": m, "entity": e, "fir": f} for i, m, e, f in untraced],
        "findings": findings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a raw-evidence run against the generator's ground truth")
    ap.add_argument("--ground-truth", required=True, help="the case's ground-truth file (e.g. CASE_ATM_001.json)")
    ap.add_argument("--run", required=True, help="the run's --output folder")
    ap.add_argument("--case", required=True, help="the run's --case id")
    ap.add_argument("--audio-dir", help="the case's Audio folder (its manifests give each recording's start)")
    ap.add_argument("--video-logs", help="the video agent's run folder (recovers each clip's own id)")
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    gt_case = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    try:
        res = score_run(Path(args.run), args.case, gt_case, Path(args.audio_dir) if args.audio_dir else None,
                        Path(args.video_logs) if args.video_logs else None)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    print(f"\nrun observation -> ground-truth observation -> event / person   ({res['timeline_file']}, {res['er_file']})")
    for t in res["traced"]:
        print(f"  {t['obs_id']:<4} {t['modality']:<5} {t['entity'][:30]:<30} -> {t['traced_to']:<4} -> "
              f"{t['event']} / {t['person']}")
    if res["untraced"]:
        print(f"  not traced ({len(res['untraced'])}; FIR ones have no ground-truth observation):",
              ", ".join(f"{u['obs_id']}{' (FIR)' if u['fir'] else ''}" for u in res["untraced"]))
    print("\nscores")
    for row in res["scores"]:
        print(f"  {row['metric']:<34} {row['value']}")
    print(f"  {'ER wrong merges':<34} {res['wrong_merges']}")
    print(f"\n  missed: {res['missed']}  half credit: {res['half_credit']}  unobservable: {res['unobservable']}")
    if res["findings"]:
        print("\ncritique findings:")
        for f in res["findings"]:
            print(f"  [{f['check']}] {f['detail'][:220]}")
    print(f"\nTraced ground truth and evaluation saved under {Path(args.run) / 'evaluation_traced'}")


if __name__ == "__main__":
    main()
