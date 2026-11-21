#!/usr/bin/env python3
"""
ForenSynth - benchmark: every test set, every metric, one table.

Runs each case of each test set through the full pipeline (or ER alone,
--er-only, in about a minute) and prints one row per set:

  ER        precision / recall / recall against what the evidence supports
            (evaluate_er.py), and how many cases fused two people
  Timeline  F1, event recall (and over observable events), precision,
            entity / action accuracy, timing error, duplicate events
            (evaluate.py)
  Decisions how the Showrunner ended each case
  FIR       of the head-count errors the generator put in the FIR
            (ground_truth.fir_errors), how many the pipeline flagged

The test sets are the same 13 cases (generator seed 42) as clean
evidence (cases/), realistic field-agent evidence (cases_realistic/) and
harder (cases_realistic_hard/) - see GENERATOR_FIXED/realism.py - so a
difference between rows is the effect of realistic evidence alone.

Judge every change by it:
    python pipeline/benchmark.py --save before        # store this run as "before"
    ... change something ...
    python pipeline/benchmark.py --compare before     # same table, with deltas

    python pipeline/benchmark.py --er-only            # ER columns only, fast
    python pipeline/benchmark.py --sets cases_realistic --cases CASE_ATM_002 CASE_ATM_005
    python pipeline/benchmark.py --llm                # with the LLM (not reproducible, uses API calls)

Output: output/benchmark/<run>/ (per-case pipeline outputs) and
output/benchmark/<run>/summary.json; --save NAME also keeps
output/benchmark/baselines/NAME.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = Path(__file__).parent.parent
for _p in [str(_root / "pipeline"), str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SETS = ["cases", "cases_realistic", "cases_realistic_hard"]
BENCH_DIR = _root / "output" / "benchmark"

# column -> (label, source, format, higher is better)
COLUMNS = [
    ("er_precision",            "ER P",        ".3f", True),
    ("er_recall",               "ER R",        ".3f", True),
    ("er_recall_supported",     "ER R(sup)",   ".3f", True),
    ("wrong_merge_cases",       "fused",       "d",   False),
    ("f1_score",                "TL F1",       ".3f", True),
    ("event_recall",            "ev R",        ".3f", True),
    ("event_recall_observable", "ev R(obs)",   ".3f", True),
    ("precision",               "ev P",        ".3f", True),
    ("precision_linked",        "ev P(link)",  ".3f", True),
    ("f1_linked",               "F1(link)",    ".3f", True),
    ("entity_accuracy",         "ent acc",     ".3f", True),
    ("action_tag_accuracy",     "act acc",     ".3f", True),
    ("temporal_accuracy_sec",   "t err s",     ".1f", False),
    ("n_duplicates",            "dup",         ".1f", False),
]
ER_COLUMNS = {"er_precision", "er_recall", "er_recall_supported", "wrong_merge_cases"}


def _cases(set_dir: Path, only: Optional[List[str]]) -> List[str]:
    ids = sorted(p.stem for p in set_dir.glob("*.json")
                 if not p.stem.endswith("_obs_only") and (set_dir / f"{p.stem}_obs_only.json").exists())
    return [c for c in ids if not only or c in only]


def _fir_errors_flagged(gt: Dict[str, Any], er: Dict[str, Any]) -> tuple[int, int, int]:
    """(planted head-count errors, how many ER's FIR check flagged, false
    alarms = roles flagged where the FIR count was right)."""
    planted = {e["type"].split("_")[0] for e in gt.get("ground_truth", {}).get("fir_errors", [])
               if e["type"].endswith("_count")}
    # "possible_split" is ER saying it may have left one person split, not a
    # claim that the FIR is wrong - not a flag.
    flagged_roles = {c["role"] for c in er.get("fir_role_check", [])
                     if c["status"] in ("mismatch", "unverified", "more_than_fir")}
    return len(planted), len(planted & flagged_roles), len(flagged_roles - planted)


def run_case(set_name: str, case: str, run_dir: Path, llm: bool) -> Dict[str, Any]:
    """Full pipeline for one case (own DB, so cases can run in parallel)."""
    from evaluate_er import evaluate_er_file
    set_dir, out = _root / set_name, run_dir / set_name
    db = out / "db" / f"{case}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(_root / "pipeline" / "run_case.py"), "--input",
           str(set_dir / f"{case}_obs_only.json"), "--output", str(out), "--db", str(db),
           "--skip-video", "--skip-report"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if not llm:
        cmd.append("--no-llm")
        env.update(GROQ_API_KEY="", Timeline_Key="")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_root), env=env, capture_output=True, text=True, encoding="utf-8")
    row: Dict[str, Any] = {"case": case, "rc": proc.returncode, "seconds": round(time.time() - t0, 1)}
    if proc.returncode != 0:
        row["error"] = proc.stderr[-600:]
        return row
    ev_path = out / "evaluation" / f"{case}_evaluation.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        row.update({k: ev.get(k) for k, *_ in COLUMNS if k in ev})
    er_files = sorted((out / "er").glob(f"{case}_er_v*_output.json"),
                      key=lambda p: int(p.stem.split("_er_v")[1].split("_")[0]))
    if er_files:
        er_eval = evaluate_er_file(str(er_files[-1]), str(set_dir / f"{case}.json")) or {}
        row.update({k: er_eval.get(k) for k in ("er_precision", "er_recall", "er_recall_supported")})
        row["wrong_merges"] = er_eval.get("wrong_merges", [])
        gt = json.loads((set_dir / f"{case}.json").read_text(encoding="utf-8"))
        row["fir_planted"], row["fir_flagged"], row["fir_false_alarms"] = _fir_errors_flagged(
            gt, json.loads(er_files[-1].read_text(encoding="utf-8")))
    row["actions"] = [json.loads(p.read_text(encoding="utf-8"))["action"]
                      for p in sorted((out / "showrunner").glob(f"{case}_showrunner_C*.json"))]
    return row


def run_er_only(set_name: str, case: str, llm: bool) -> Dict[str, Any]:
    from entity_resolution import resolve_entities
    from evaluate_er import er_pairwise_metrics
    set_dir = _root / set_name
    gt = json.loads((set_dir / f"{case}.json").read_text(encoding="utf-8"))
    er = resolve_entities(json.loads((set_dir / f"{case}_obs_only.json").read_text(encoding="utf-8")),
                          llm_enabled=llm)
    m = er_pairwise_metrics(er, gt["observations"], gt["ground_truth"]["entity_mapping"],
                            gt["ground_truth"].get("obs_entity"))
    planted, flagged, false_alarms = _fir_errors_flagged(gt, er)
    return {"case": case, "rc": 0, **{k: m[k] for k in ("er_precision", "er_recall", "er_recall_supported")},
            "wrong_merges": m["wrong_merges"], "fir_planted": planted, "fir_flagged": flagged,
            "fir_false_alarms": false_alarms}


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("rc") == 0]
    s: Dict[str, Any] = {"n_cases": len(rows), "n_failed": len(rows) - len(ok)}
    for key, *_ in COLUMNS:
        if key == "wrong_merge_cases":
            s[key] = sum(1 for r in ok if r.get("wrong_merges"))
            continue
        vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
        s[key] = round(statistics.mean(vals), 4) if vals else None
    outcomes: Dict[str, int] = {}
    for r in ok:
        if r.get("actions"):
            outcomes[r["actions"][-1]] = outcomes.get(r["actions"][-1], 0) + 1
    s["final_actions"] = outcomes
    s["fir_planted"] = sum(r.get("fir_planted", 0) for r in ok)
    s["fir_flagged"] = sum(r.get("fir_flagged", 0) for r in ok)
    s["fir_false_alarms"] = sum(r.get("fir_false_alarms", 0) for r in ok)
    return s


def print_table(summary: Dict[str, Dict[str, Any]], er_only: bool,
                baseline: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    cols = [c for c in COLUMNS if not er_only or c[0] in ER_COLUMNS]
    width = max(len(s) for s in summary) + 2
    print("\n" + "set".ljust(width) + "".join(f"{label:>{14 if baseline else 10}}" for _, label, *_ in cols))
    for set_name, s in summary.items():
        line = set_name.ljust(width)
        for key, _, fmt, higher in cols:
            v = s.get(key)
            cell = "-" if v is None else format(v, fmt)
            b = (baseline or {}).get(set_name, {}).get(key)
            if baseline and v is not None and b is not None and v != b:
                # delta, with ^ for better and v for worse
                cell = f"{format(v, fmt)}{'^' if (v > b) == higher else 'v'}{format(abs(v - b), fmt)}"
            line += f"{cell:>{14 if baseline else 10}}"
        print(line)
    print()
    for set_name, s in summary.items():
        extra = (f"  FIR head-count errors flagged: {s['fir_flagged']}/{s['fir_planted']}" if s.get("fir_planted") else "")             + (f"  FIR false alarms: {s['fir_false_alarms']}" if s.get("fir_false_alarms") else "")
        acts = ", ".join(f"{k} {v}" for k, v in sorted(s.get("final_actions", {}).items()))
        fails = f"  FAILED RUNS: {s['n_failed']}" if s.get("n_failed") else ""
        print(f"{set_name.ljust(width)}{s['n_cases']} cases" + (f"  outcomes: {acts}" if acts else "") + extra + fails)
    print("\nER R(sup) = ER recall over the same-person links the evidence supports; ev R(obs) = event "
          "recall over events with surviving observations; t err = mean timing error (lower is better); "
          "dup = extra events per case for one moment; fused = cases where ER merged two people; "
          "P(link)/F1(link) = not counting repeats the timeline links as one moment, or denial-only events.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the pipeline on every test set")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS, help="Test-set folders (default: all three)")
    ap.add_argument("--cases", nargs="+", help="Only these case ids")
    ap.add_argument("--er-only", action="store_true", help="Score ER alone (fast)")
    ap.add_argument("--llm", action="store_true", help="Use the LLM (default: offline, reproducible)")
    ap.add_argument("--jobs", type=int, default=3, help="Cases run in parallel (default 3)")
    ap.add_argument("--run-name", default=None, help="Folder under output/benchmark (default: timestamp)")
    ap.add_argument("--save", metavar="NAME", help="Keep this run's summary as baseline NAME")
    ap.add_argument("--compare", metavar="NAME", help="Show deltas against baseline NAME")
    args = ap.parse_args()

    logging.disable(logging.WARNING)
    run_dir = BENCH_DIR / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, List[Dict[str, Any]]] = {}
    for set_name in args.sets:
        cases = _cases(_root / set_name, args.cases)
        if not cases:
            print(f"[skip] {set_name}: no ground-truth cases")
            continue
        print(f"{set_name}: {len(cases)} case(s) ...", flush=True)
        if args.er_only:
            results[set_name] = [run_er_only(set_name, c, args.llm) for c in cases]
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
                results[set_name] = list(pool.map(lambda c: run_case(set_name, c, run_dir, args.llm), cases))
        for r in results[set_name]:
            if r.get("rc"):
                print(f"  FAILED {r['case']}: {r.get('error', '')[-200:]}")

    summary = {s: summarise(rows) for s, rows in results.items()}
    report = {"settings": {"er_only": args.er_only, "llm": args.llm, "sets": args.sets, "cases": args.cases},
              "summary": summary, "per_case": results}
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    baseline = None
    if args.compare:
        bpath = BENCH_DIR / "baselines" / f"{args.compare}.json"
        baseline = json.loads(bpath.read_text(encoding="utf-8"))["summary"] if bpath.exists() else None
        if baseline is None:
            print(f"[warn] no baseline '{args.compare}' at {bpath}")
    print_table(summary, args.er_only, baseline)
    if args.save:
        (BENCH_DIR / "baselines").mkdir(parents=True, exist_ok=True)
        (BENCH_DIR / "baselines" / f"{args.save}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved baseline '{args.save}'")
    print(f"Details: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
