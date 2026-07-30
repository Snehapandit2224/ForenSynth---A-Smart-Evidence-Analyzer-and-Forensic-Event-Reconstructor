#!/usr/bin/env python3
"""
ForenSynth – run_showrunner.py
Showrunner Agent entry point.

Auto-detects the latest critique and timeline versions.
Tells you exactly what command to run next.

Usage:
    python pipeline\run_showrunner.py --case CASE_ATM_001
    python pipeline\run_showrunner.py --critique c.json --timeline t.json
"""
from __future__ import annotations

import argparse
import json
import logging
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

from showrunner_agent import run_showrunner, run_showrunner_from_files


def parse_args():
    p = argparse.ArgumentParser(description="ForenSynth Showrunner Agent")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--case",      help="Case ID — auto-detects latest versions")
    group.add_argument("--critique",  help="Path to critique JSON file")
    p.add_argument("--timeline",      help="Path to timeline JSON (with --critique)")
    p.add_argument("--output",        default="./output/showrunner")
    p.add_argument("--no-db",         action="store_true")
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
            crit_ver  = versions["crit_version"]

            if not crit_ver:
                print(f"[ERROR] No critique found for {args.case}. Run Critique Agent first.")
                sys.exit(1)

            print(f"\n{'='*60}")
            print(f"  ForenSynth Showrunner Agent")
            print(f"  Case       : {args.case}")
            print(f"  Timeline   : {tl_ver}")
            print(f"  Critique   : {crit_ver}")
            print(f"{'='*60}\n")

            payload = mem.load_for_showrunner(
                args.case,
                tl_version=tl_ver,
                crit_version=crit_ver,
            )
            if not payload:
                print(f"[ERROR] Could not load showrunner payload for {args.case}")
                sys.exit(1)

            result = run_showrunner(payload)

        except ImportError:
            print("[ERROR] memory_store.py not found."); sys.exit(1)

    else:
        if not args.timeline:
            print("[ERROR] --timeline required with --critique"); sys.exit(1)
        result = run_showrunner_from_files(
            critique_path=args.critique,
            timeline_path=args.timeline,
            output_dir=args.output,
        )

    # ── Print decision ────────────────────────────────────────────────────────
    action   = result["action"]
    case_id  = result["case_id"]
    next_tl  = result["output_tl_version"]

    print(f"Action            : {action}")
    print(f"Next TL version   : {next_tl}")
    print(f"Reasoning         : {result['reasoning'][:150]}")
    print(f"Issues addressed  : {result['issues_addressed']}")
    print(f"Issues deferred   : {result['issues_deferred']}")

    if result.get("er_constraints"):
        c = result["er_constraints"]
        print(f"\nER Constraints (v{c.get('er_version',2)}):")
        print(f"  must_not_merge : {c.get('must_not_merge',[])}")
        print(f"  must_merge     : {c.get('must_merge',[])}")

    # ── Save output JSON ──────────────────────────────────────────────────────
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_id}_showrunner_{result.get('input_crit_version','C1')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")

    # ── Save to DB ────────────────────────────────────────────────────────────
    if not args.no_db and args.case:
        try:
            from memory_store import ForenSynthMemory
            mem = ForenSynthMemory()
            if action == "re_run_er" and result.get("er_constraints"):
                c   = result["er_constraints"]
                ver = mem.save_er_constraints(
                    case_id,
                    constraints={
                        "must_not_merge":     c.get("must_not_merge",[]),
                        "must_merge":         c.get("must_merge",[]),
                        "soft_hints":         c.get("soft_hints",{}),
                        "previous_gap_types": [
                            g.get("gap_type","")
                            for g in payload["critique"].get("gaps",[])
                        ],
                    },
                    reason=result.get("reasoning","")[:500],
                )
                print(f"Saved → forensynth.db  (ER constraints v{ver})")
            mem.save_showrunner_run({
                **result,
                "action_taken":       action,
                "input_tl_version":   tl_ver  if args.case else "V1",
                "input_crit_version": crit_ver if args.case else "C1",
                "output_tl_version":  next_tl,
            })
            print(f"Saved → forensynth.db  (Showrunner run)")
        except Exception as exc:
            print(f"[DB WARNING] {exc}")

    # ── Tell user exactly what to run next ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  NEXT COMMAND TO RUN:")
    print(f"{'='*60}")
    if action == "re_run_er":
        print(f"  python pipeline\\run_er.py --case {case_id} --rerun")
        print(f"  python pipeline\\run_timeline.py --case {case_id}")
        print(f"  python pipeline\\run_critique.py --case {case_id}")
        print(f"  python pipeline\\run_showrunner.py --case {case_id}")
    elif action == "re_run_timeline":
        print(f"  python pipeline\\run_timeline.py --case {case_id}")
        print(f"  python pipeline\\run_critique.py --case {case_id}")
        print(f"  python pipeline\\run_showrunner.py --case {case_id}")
    elif action == "human_review":
        print(f"  [MANUAL] Review forensynth.db for case {case_id}")
        print(f"  Reason: {result['reasoning'][:200]}")
    elif action == "no_action":
        print(f"  [DONE] Timeline {next_tl} is final for {case_id}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()