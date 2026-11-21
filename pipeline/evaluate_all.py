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
import os
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for _p in [str(_root / "pipeline"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate import evaluate_case

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("forensynth.evaluate_all")

PYTHON   = sys.executable
RUN_CASE = str(_root / "pipeline" / "run_case.py")
DB_PATH  = str(_root / "forensynth.db")

TABLES_WITH_CASE_ID = [
    "cases", "observations", "er_runs", "er_canonical", "er_clusters",
    "er_constraints", "timeline_runs", "timeline_events", "timeline_edges",
    "critique_runs", "critique_issues", "showrunner_runs", "pipeline_runs",
]


def _wipe_case(case_id: str) -> None:
    """Delete any existing DB rows for this case_id so a re-run is genuinely
    from scratch, not resuming state from a previous evaluation pass."""
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    for t in TABLES_WITH_CASE_ID:
        try:
            conn.execute(f"DELETE FROM {t} WHERE case_id=?", (case_id,))
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def _run_pipeline_for_case(obs_only_path: str, no_llm: bool, pipeline_output_dir: str) -> bool:
    """
    Re-run run_case.py for one case so the evaluation reflects a fresh
    pipeline run, not a stale timeline left over from an earlier pass.

    When no_llm=True, forces the FULLY deterministic heuristic-only path:
    run_case.py's own --no-llm flag only disables the LLM in Entity
    Resolution. The Timeline agent independently builds a CloudLLMFallback
    (agents/timeline_agent.py:_build_default_llm) whenever a Timeline_Key
    or GROQ_API_KEY is present in the environment, regardless of that
    flag -- so both env vars (plus TIMELINE_LOCAL_LLM_BACKEND, in case a
    local backend is configured) must be cleared here too for the Timeline
    stage to actually fall back to NoOpLLMFallback (verified against
    agents/timeline_agent.py's _build_default_llm() fallback chain).

    Returns True on success, False if the subprocess failed (logged either way).
    """
    case_id = Path(obs_only_path).name.replace("_obs_only.json", "")
    _wipe_case(case_id)

    child_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = [
        PYTHON, RUN_CASE, "--input", obs_only_path, "--output", pipeline_output_dir,
        "--skip-video", "--skip-report", "--skip-eval",
    ]
    if no_llm:
        cmd.append("--no-llm")
        child_env["GROQ_API_KEY"] = ""
        child_env["Timeline_Key"] = ""
        child_env["TIMELINE_LOCAL_LLM_BACKEND"] = ""

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=child_env)
    if proc.returncode != 0:
        log.error(
            "Pipeline run failed for %s (exit %d):\n%s",
            case_id, proc.returncode, (proc.stdout or "")[-1500:] + (proc.stderr or "")[-1500:],
        )
        return False
    return True


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
    ap.add_argument("--pipeline-output-dir", default="output",
                     help="--output passed through to run_case.py when a re-run is triggered")
    ap.add_argument("--no-llm", action="store_true",
                     help="Force a fully deterministic heuristic-only pipeline run (ER *and* Timeline) "
                          "before evaluating each case, instead of evaluating whatever is already on disk")
    ap.add_argument("--stable-check", action="store_true",
                     help="Run each case --runs times and report F1 mean +/- stdev, to verify results "
                          "are stable before presenting them (implies re-running the pipeline each time)")
    ap.add_argument("--runs", type=int, default=3, help="Repeat count when --stable-check is set (default 3)")
    args = ap.parse_args()

    gt_files = _discover_ground_truth_files(args.generator_dir)
    print(f"Found {len(gt_files)} ground-truth case files in {args.generator_dir}")

    need_rerun = args.no_llm or args.stable_check
    n_runs = args.runs if args.stable_check else 1

    rows = []
    stability_rows = []
    for gt_path in gt_files:
        case_id = Path(gt_path).stem
        obs_only_path = str(Path(args.generator_dir) / f"{case_id}_obs_only.json")

        f1_samples: list = []
        r = None
        for run_i in range(n_runs):
            if need_rerun:
                if not Path(obs_only_path).exists():
                    print(f"  {case_id}: no obs_only file at {obs_only_path} -- cannot re-run, skipping")
                    r = {"case_id": case_id, "result": "NO_TIMELINE"}
                    break
                print(f"  {case_id}: re-running pipeline (run {run_i + 1}/{n_runs}, no_llm={args.no_llm})...")
                _run_pipeline_for_case(obs_only_path, args.no_llm, args.pipeline_output_dir)

            timeline_path = _find_final_timeline(case_id, args.timelines_dir)
            if not timeline_path:
                print(f"  {case_id}: no timeline output found in {args.timelines_dir} -- skipping")
                r = {"case_id": case_id, "result": "NO_TIMELINE"}
                break

            r = evaluate_case(case_id, gt_path, timeline_path, output_dir=args.output_dir)
            if r.get("f1_score") is not None:
                f1_samples.append(r["f1_score"])

        rows.append(r)

        if args.stable_check:
            mean_f1 = round(statistics.mean(f1_samples), 4) if f1_samples else None
            stdev_f1 = round(statistics.stdev(f1_samples), 4) if len(f1_samples) > 1 else 0.0
            stability_rows.append({
                "case_id": case_id, "f1_samples": f1_samples,
                "f1_mean": mean_f1, "f1_stdev": stdev_f1,
            })

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

    all_f1 = [r.get("f1_score") for r in rows if r.get("f1_score") is not None]
    mean_f1_overall = round(statistics.mean(all_f1), 4) if all_f1 else None
    print(f"\nMean F1 across {len(all_f1)} evaluated case(s): {mean_f1_overall}")

    if args.stable_check:
        print(f"\n{'CASE ID':<16} | F1 samples ({args.runs} runs)          | Mean   | Stdev")
        print("-" * 70)
        for sr in stability_rows:
            samples_s = ", ".join(f"{v:.2f}" for v in sr["f1_samples"]) or "--"
            print(f"{sr['case_id']:<16} | {samples_s:<30} | {sr['f1_mean']}  | {sr['f1_stdev']}")

    summary = {
        "cases":     rows,
        "n_cases":   len(rows),
        "n_pass":    sum(1 for r in rows if r.get("result") == "PASS"),
        "n_partial": sum(1 for r in rows if r.get("result") == "PARTIAL"),
        "n_fail":    sum(1 for r in rows if r.get("result") == "FAIL"),
        "n_error":   sum(1 for r in rows if r.get("result") in ("ERROR", "NO_TIMELINE")),
        "mean_f1":   mean_f1_overall,
        "stability": stability_rows if args.stable_check else None,
    }
    out_path = Path(args.output_dir) / "evaluation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved -> {out_path}")


if __name__ == "__main__":
    main()
