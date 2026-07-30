#!/usr/bin/env python3
"""
ForenSynth – run_timeline.py
Timeline Agent entry point.

Auto-detects the latest ER version and produces the next timeline version.
No need to specify versions manually — the loop handles it.

Usage:
    python pipeline\run_timeline.py --case CASE_ATM_001
    python pipeline\run_timeline.py --obs obs.json --er er.json  # from files
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

from timeline_agent import run_timeline_agent


def parse_args():
    p = argparse.ArgumentParser(description="ForenSynth Timeline Agent")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--case", help="Case ID — auto-detects latest ER version")
    group.add_argument("--obs",  help="Path to obs_only JSON file")
    p.add_argument("--er",       help="Path to ER output JSON (with --obs)")
    p.add_argument("--output",   default="./output/timelines")
    p.add_argument("--no-db",    action="store_true")
    p.add_argument("--json",     action="store_true", dest="print_json")
    return p.parse_args()


def main():
    args = parse_args()

    if args.case:
        try:
            from memory_store import ForenSynthMemory
            mem = ForenSynthMemory()

            # ── Auto-detect versions ──────────────────────────────────────────
            versions = mem.get_latest_versions(args.case)
            er_ver   = versions["er_version"]
            tl_ver   = versions["next_tl_version"]  # always produce next version

            if er_ver == 0:
                print(f"[ERROR] No ER run found for {args.case}. Run ER first.")
                sys.exit(1)

            print(f"\n{'='*60}")
            print(f"  ForenSynth Timeline Agent")
            print(f"  Case       : {args.case}")
            print(f"  Using ER   : v{er_ver}")
            print(f"  Producing  : Timeline {tl_ver}")
            print(f"{'='*60}\n")

            payload = mem.load_for_timeline(args.case, er_version=er_ver)
            if not payload:
                print(f"[ERROR] Could not load payload for {args.case}")
                sys.exit(1)

        except ImportError:
            print("[ERROR] memory_store.py not found."); sys.exit(1)

    elif args.obs and args.er:
        for p in (Path(args.obs), Path(args.er)):
            if not p.exists():
                print(f"[ERROR] Not found: {p}"); sys.exit(1)
        with open(args.obs) as f: obs = json.load(f)
        with open(args.er)  as f: er  = json.load(f)
        payload = {
            "case_id": obs.get("case_id") or er.get("case_id","UNKNOWN"),
            "obs_only": {"observations": obs.get("observations",[])},
            "entity_resolved": {
                "canonical_entities": er.get("canonical_entities",[]),
                "clusters":           er.get("clusters",[]),
                "conflicts_detected": er.get("conflicts_detected",0),
                "conflicts":          er.get("conflicts",[]),
            }
        }
        # File mode: use next version from DB if available, else V1
        tl_ver = "V1"
        if not args.no_db:
            try:
                from memory_store import ForenSynthMemory
                mem      = ForenSynthMemory()
                versions = mem.get_latest_versions(payload["case_id"])
                tl_ver   = versions["next_tl_version"]
            except Exception:
                pass

        print(f"\n{'='*60}")
        print(f"  ForenSynth Timeline Agent — {payload['case_id']}  {tl_ver}")
        print(f"{'='*60}\n")
    else:
        print("[ERROR] Provide --case CASE_ID or --obs FILE --er FILE"); sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    result  = run_timeline_agent(payload, output_dir=str(out_dir))
    result["timeline_version"] = tl_ver

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Timeline version : {result['timeline_version']}")
    print(f"Events           : {len(result['events'])}")
    print(f"Causal links     : {len(result['causal_links'])}")
    print(f"Conflicts        : {len(result['conflicts_summary'])}")
    print(f"Time             : {result.get('total_time_sec',0):.3f}s")
    print()
    for n in result["narrative"]:
        cf = " ⚠" if any(ev.get("conflict_flag") for ev in result["events"]
                          if ev["event_id"] == n["event_id"]) else ""
        print(f"  [{n['timestamp'][11:19]}] {n['actor']:>14}  "
              f"conf={n['confidence']:.2f}{cf}  {n['action'][:55]}")

    # ── Save versioned JSON file ──────────────────────────────────────────────
    case_id      = result["case_id"]
    out_path     = out_dir / f"{case_id}_timeline_{tl_ver}.json"
    graph_path   = out_dir / f"{case_id}_timeline_{tl_ver}_graph.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if result.get("timeline_graph"):
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(result["timeline_graph"], f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {out_path}")
    print(f"Saved → {graph_path}")

    # ── Save to DB ────────────────────────────────────────────────────────────
    if not args.no_db:
        try:
            from memory_store import ForenSynthMemory
            mem = ForenSynthMemory()
            mem.save_timeline(result)
            print(f"Saved → forensynth.db  (Timeline {tl_ver})")
        except ImportError:
            print("[DB] memory_store.py not found — skipping")
        except Exception as exc:
            print(f"[DB WARNING] {exc}")

    if args.print_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()