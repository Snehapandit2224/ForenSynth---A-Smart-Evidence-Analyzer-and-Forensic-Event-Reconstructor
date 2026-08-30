#!/usr/bin/env python3
"""
ForenSynth – run_critique.py
Critique Agent entry point.

Auto-detects the latest timeline version and produces the next critique version.

Usage:
    python pipeline\run_critique.py --case CASE_ATM_001
    python pipeline\run_critique.py --timeline timeline.json --er er.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  # non-standard stdout (e.g. some test runners) - fall back silently

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

from critique_agent import run_critique_agent, run_critique_agent_from_files


def parse_args():
    p = argparse.ArgumentParser(description="ForenSynth Critique Agent")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--case",     help="Case ID — auto-detects latest timeline")
    group.add_argument("--timeline", help="Path to timeline JSON file")
    p.add_argument("--er",           help="Path to ER output JSON (with --timeline)")
    p.add_argument("--output",       default="./output/critiques")
    p.add_argument("--no-db",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.case:
        try:
            from memory_store import ForenSynthMemory
            mem = ForenSynthMemory()

            # ── Auto-detect versions ──────────────────────────────────────────
            versions  = mem.get_latest_versions(args.case)
            tl_ver    = versions["tl_version"]
            crit_ver  = versions["next_crit_version"]

            if not tl_ver:
                print(f"[ERROR] No timeline found for {args.case}. Run Timeline Agent first.")
                sys.exit(1)

            print(f"\n{'='*60}")
            print(f"  ForenSynth Critique Agent")
            print(f"  Case          : {args.case}")
            print(f"  Critiquing    : Timeline {tl_ver}")
            print(f"  Producing     : Critique {crit_ver}")
            print(f"{'='*60}\n")

            payload = mem.load_for_critique(args.case, tl_version=tl_ver)
            if not payload:
                print(f"[ERROR] Could not load critique payload for {args.case}")
                sys.exit(1)

            result = run_critique_agent(payload, critique_version=crit_ver)

        except ImportError:
            print("[ERROR] memory_store.py not found."); sys.exit(1)

    else:
        if not args.er:
            print("[ERROR] --er required with --timeline"); sys.exit(1)
        crit_ver = "C1"
        result   = run_critique_agent_from_files(
            timeline_path=args.timeline,
            er_path=args.er,
            critique_version=crit_ver,
            output_dir=args.output,
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Overall score     : {result['overall_score']:.2f}")
    print(f"Gaps detected     : {len(result['gaps'])}")
    print(f"Unresolvable      : {len(result['unresolvable_gaps'])}")
    print(f"Recommended action: {result['recommended_action']}")
    print(f"Requires revision : {result['requires_revision']}")
    if result.get("revision_target"):
        print(f"Revision target   : {result['revision_target']}")
    print()
    for g in result["gaps"]:
        res = "✓" if g.get("resolvable") else "✗"
        print(f"  {res} [{g['gap_id']}] {g['gap_type']}  severity={g['severity']:.2f}")
        print(f"     {g['root_cause'][:80]}")
    print()
    print(f"Narrative: {result['narrative_summary'][:200]}")

    # ── Save versioned JSON ───────────────────────────────────────────────────
    case_id  = result.get("case_id","UNKNOWN")
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_critique_{crit_ver}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")

    # ── Save to DB ────────────────────────────────────────────────────────────
    if not args.no_db and args.case:
        try:
            from memory_store import ForenSynthMemory
            mem = ForenSynthMemory()
            mem.save_critique(result)
            print(f"Saved → forensynth.db  (Critique {crit_ver})")
        except Exception as exc:
            print(f"[DB WARNING] {exc}")

    # ── Tell user what to run next ────────────────────────────────────────────
    if args.case:
        print(f"\nNext step:")
        if result["requires_revision"]:
            print(f"  python pipeline\\run_showrunner.py --case {case_id}")
        else:
            print(f"  Timeline {result.get('timeline_version','V1')} is clean. Pipeline complete.")


if __name__ == "__main__":
    main()