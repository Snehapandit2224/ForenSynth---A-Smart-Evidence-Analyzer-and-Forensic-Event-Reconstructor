#!/usr/bin/env python3
"""
ForenSynth – run_er.py
Entity Resolution entry point.

Usage:
    # First run (from file)
    python pipeline\run_er.py --input cases\cases_atm\CASE_ATM_001_obs_only.json --output output\er

    # Re-run after Showrunner (picks up constraints from DB automatically)
    python pipeline\run_er.py --case CASE_ATM_001 --rerun --output output\er

    # Force heuristic only
    python pipeline\run_er.py --input cases\cases_atm\CASE_ATM_001_obs_only.json --no-llm --output output\er

Environment (.env):
    GROQ_API_KEY    enables LLM coreference scoring
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

_root = Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

from entity_resolution import resolve_entities


def parse_args():
    p = argparse.ArgumentParser(description="ForenSynth Entity Resolution")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to obs_only JSON file (first run)")
    group.add_argument("--case",  help="Case ID — load from DB")
    p.add_argument("--output",  default="./output/er", help="Output directory")
    p.add_argument("--no-llm",  action="store_true",   help="Heuristic-only mode")
    p.add_argument("--rerun",   action="store_true",    help="Re-run with Showrunner constraints from DB")
    p.add_argument("--no-db",   action="store_true",    help="Skip saving to DB")
    return p.parse_args()


def main():
    args    = parse_args()
    llm     = not args.no_llm and bool(os.environ.get("GROQ_API_KEY", ""))

    human_constraints = None
    run_version       = 1
    obs_data          = None

    # ── Load from file (first run) ────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_path}"); sys.exit(1)
        with open(input_path, encoding="utf-8") as f:
            obs_data = json.load(f)

        # Check if this case already has ER runs in DB — set version correctly
        if not args.no_db:
            try:
                from memory_store import ForenSynthMemory
                mem    = ForenSynthMemory()
                status = mem.get_pipeline_status(obs_data["case_id"])
                er_runs = status.get("er_runs", [])
                if er_runs:
                    run_version = max(r["run_version"] for r in er_runs) + 1
                    print(f"Case already has ER v{run_version-1} — saving as v{run_version}")
            except Exception:
                pass

    # ── Load from DB (re-run after Showrunner) ────────────────────────────────
    else:
        try:
            from memory_store import ForenSynthMemory
            mem      = ForenSynthMemory()
            obs_data = mem.load_observations(args.case)
            if not obs_data:
                print(f"[ERROR] Case '{args.case}' not found in DB."); sys.exit(1)

            if args.rerun:
                # Get latest ER version and increment
                status   = mem.get_pipeline_status(args.case)
                er_runs  = status.get("er_runs", [])
                run_version = (max(r["run_version"] for r in er_runs) + 1) if er_runs else 1

                # Load constraints saved by Showrunner
                c = mem.load_er_constraints(args.case)
                if c:
                    human_constraints = {
                        "must_not_merge": c.get("must_not_merge", []),
                        "must_merge":     c.get("must_merge", []),
                        "soft_hints":     c.get("soft_hints", {}),
                    }
                    print(f"Loaded constraints: must_not_merge={c.get('must_not_merge',[])} must_merge={c.get('must_merge',[])}")
                else:
                    print("[WARNING] --rerun specified but no constraints found in DB")
        except ImportError:
            print("[ERROR] memory_store.py not found."); sys.exit(1)

    case_id = obs_data.get("case_id", "UNKNOWN")

    print(f"\n{'='*60}")
    print(f"  ForenSynth Entity Resolution")
    print(f"  Case    : {case_id}")
    print(f"  Version : v{run_version}")
    print(f"  LLM     : {'ENABLED' if llm else 'DISABLED'}")
    print(f"  Rerun   : {args.rerun}")
    print(f"{'='*60}\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    result = resolve_entities(
        obs_data,
        llm_enabled=llm,
        human_constraints=human_constraints,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Status          : {result['status']}")
    print(f"Classification  : {result['output_classification']}")
    print(f"Entities found  : {result['entity_count']}")
    print(f"Conflicts       : {result['conflicts_detected']}")
    print(f"LLM calls       : {result['llm_calls_made']}/{result['llm_calls_budget']}")
    print(f"Time            : {result['total_processing_time_sec']:.3f}s")
    print()
    for e in result["canonical_entities"]:
        print(f"  {e['entity_id']}: {e['primary_alias']}  aliases={e['aliases']}")

    # ── Save output JSON (versioned filename) ─────────────────────────────────
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # FIX: include version in filename so re-runs don't overwrite V1
    out_path = out_dir / f"{case_id}_er_v{run_version}_output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")

    # ── Save to DB ────────────────────────────────────────────────────────────
    if not args.no_db:
        try:
            from memory_store import ForenSynthMemory
            mem = ForenSynthMemory()
            if args.input:
                mem.save_case(obs_data)   # save obs on first run
            mem.save_er_result(result, run_version=run_version)
            print(f"Saved → forensynth.db  (ER v{run_version})")
        except ImportError:
            print("[DB] memory_store.py not found — skipping")
        except Exception as exc:
            print(f"[DB WARNING] {exc}")


if __name__ == "__main__":
    main()