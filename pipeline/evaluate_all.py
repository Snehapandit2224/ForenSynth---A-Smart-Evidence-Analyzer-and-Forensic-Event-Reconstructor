#!/usr/bin/env python3
"""
ForenSynth - Ground Truth Evaluation Batch Runner
Discovers every ground-truth case file in GENERATOR_FIXED/cases_atm/
(dynamically -- no case IDs are hardcoded), evaluates the pipeline's actual
final timeline for that case_id against it, and prints + saves a summary
table.

Usage:
    python pipeline/evaluate_all.py
    python pipeline/evaluate_all.py --generator-dir GENERATOR_FIXED/cases_atm --output-dir output/evaluation
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for _p in [str(_root / "pipeline"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate import evaluate_case

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("forensynth.evaluate_all")


def _discover_ground_truth_files(generator_dir: str):
    """Every *.json in generator_dir that is NOT a *_obs_only.json file."""
    all_json = sorted(glob.glob(str(Path(generator_dir) / "*.json")))
    return [f for f in all_json if not f.endswith("_obs_only.json")]


def _find_final_timeline(case_id: str, timelines_dir: str):
    """The pipeline may finalize at V1, V2, or V3 -- most cases never reach
    V3. Pick by modification time, NOT version-number priority: output/
    can carry stale V2/V3 files left over from an earlier, unrelated run
    (different observation data), and a stale higher version must never
    outrank a fresh lower version from the run actually being evaluated."""
    candidates = [
        Path(timelines_dir) / f"{case_id}_timeline_{v}.json" for v in ("V1", "V2", "V3")
    ]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def main():
    ap = argparse.ArgumentParser(description="Evaluate all discovered ATM cases against ground truth.")
    ap.add_argument("--generator-dir", default="GENERATOR_FIXED/cases_atm")
    ap.add_argument("--timelines-dir", default="output/timelines")
    ap.add_argument("--output-dir",    default="output/evaluation")
    args = ap.parse_args()

    gt_files = _discover_ground_truth_files(args.generator_dir)
    print(f"Found {len(gt_files)} ground-truth case files in {args.generator_dir}")

    rows = []
    for gt_path in gt_files:
        case_id = Path(gt_path).stem
        timeline_path = _find_final_timeline(case_id, args.timelines_dir)
        if not timeline_path:
            print(f"  {case_id}: no timeline output found in {args.timelines_dir} -- skipping")
            rows.append({"case_id": case_id, "result": "NO_TIMELINE"})
            continue

        r = evaluate_case(case_id, gt_path, timeline_path, output_dir=args.output_dir)
        rows.append(r)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'CASE ID':<16} | {'RECALL':<7} | {'ACTION_ACC':<10} | {'TEMPORAL_ACC':<13} | {'ENTITY_ACC':<10} | {'F1':<6} | RESULT")
    print("-" * 90)
    for r in rows:
        if r.get("result") in ("NO_TIMELINE", "ERROR"):
            print(f"{r['case_id']:<16} | {'--':<7} | {'--':<10} | {'--':<13} | {'--':<10} | {'--':<6} | {r.get('result')}")
            continue
        recall     = r.get("event_recall")
        action_acc = r.get("action_tag_accuracy")
        temporal   = r.get("temporal_accuracy_sec")
        entity_acc = r.get("entity_accuracy")
        f1         = r.get("f1_score")
        recall_s   = "--" if recall is None else f"{recall*100:.0f}%"
        action_s   = "--" if action_acc is None else f"{action_acc*100:.0f}%"
        temporal_s = "--" if temporal is None else f"{temporal:.1f}s avg"
        entity_s   = "--" if entity_acc is None else f"{entity_acc*100:.0f}%"
        f1_s       = "--" if f1 is None else f"{f1:.2f}"
        print(f"{r['case_id']:<16} | {recall_s:<7} | {action_s:<10} | {temporal_s:<13} | {entity_s:<10} | {f1_s:<6} | {r.get('result')}")

    summary = {
        "cases":     rows,
        "n_cases":   len(rows),
        "n_pass":    sum(1 for r in rows if r.get("result") == "PASS"),
        "n_partial": sum(1 for r in rows if r.get("result") == "PARTIAL"),
        "n_fail":    sum(1 for r in rows if r.get("result") == "FAIL"),
        "n_error":   sum(1 for r in rows if r.get("result") in ("ERROR", "NO_TIMELINE")),
    }
    out_path = Path(args.output_dir) / "evaluation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved -> {out_path}")


if __name__ == "__main__":
    main()
