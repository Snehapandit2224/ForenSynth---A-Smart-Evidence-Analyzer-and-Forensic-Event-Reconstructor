#!/usr/bin/env python3
"""
ForenSynth – Timeline Agent
example_run.py: accepts two separate input files.

Usage (two files):
    python example_run.py --obs path/to/obs_only.json --er path/to/er_output.json

Usage (single combined file, legacy):
    python example_run.py path/to/combined_input.json

Flags:
    --json    also print full JSON output to stdout
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

sys.path.insert(0, str(Path(__file__).parent))

from agent import run_timeline_agent
from explainability import ExplainabilityLayer
from models import NarrativeLine
from validators import ValidationError


def build_payload(obs_path: Path, er_path: Path) -> dict:
    """
    Merge separate obs_only JSON and ER output JSON into the
    combined payload the Timeline Agent expects.
    """
    with open(obs_path, encoding="utf-8") as f:
        obs_data = json.load(f)
    with open(er_path, encoding="utf-8") as f:
        er_data = json.load(f)

    # obs_only file may have observations at top level or nested
    if "observations" in obs_data:
        observations = obs_data["observations"]
    else:
        raise ValueError(f"No 'observations' key found in {obs_path}")

    case_id = obs_data.get("case_id") or er_data.get("case_id") or "UNKNOWN"

    return {
        "case_id": case_id,
        "obs_only": {
            "observations": observations
        },
        "entity_resolved": {
            "canonical_entities": er_data.get("canonical_entities", []),
            "clusters":           er_data.get("clusters", []),
            "conflicts_detected": er_data.get("conflicts_detected", 0),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ForenSynth Timeline Agent")
    parser.add_argument("combined", nargs="?", help="Combined input JSON (legacy single-file mode)")
    parser.add_argument("--obs", help="Path to obs_only JSON file")
    parser.add_argument("--er",  help="Path to entity resolution output JSON file")
    parser.add_argument("--json", action="store_true", dest="print_json", help="Print full JSON output")
    parser.add_argument("--output-dir", default="./output", help="Output directory (default: ./output)")
    args = parser.parse_args()

    # ── Determine input mode ──────────────────────────────────────────────────
    if args.obs and args.er:
        obs_path = Path(args.obs)
        er_path  = Path(args.er)
        for p in (obs_path, er_path):
            if not p.exists():
                print(f"[ERROR] File not found: {p}")
                sys.exit(1)
        payload = build_payload(obs_path, er_path)
        print(f"\n{'='*60}")
        print(f"  ForenSynth Timeline Agent")
        print(f"  Obs file : {obs_path.name}")
        print(f"  ER file  : {er_path.name}")
        print(f"  Case     : {payload['case_id']}")
        print(f"{'='*60}\n")

    elif args.combined:
        combined_path = Path(args.combined)
        if not combined_path.exists():
            print(f"[ERROR] File not found: {combined_path}")
            sys.exit(1)
        with open(combined_path, encoding="utf-8") as f:
            payload = json.load(f)
        print(f"\n{'='*60}")
        print(f"  ForenSynth Timeline Agent – Case: {payload.get('case_id','?')}")
        print(f"{'='*60}\n")

    else:
        # Default: use bundled example
        default = Path(__file__).parent / "example_input.json"
        if not default.exists():
            print("[ERROR] No input specified and example_input.json not found.")
            parser.print_help()
            sys.exit(1)
        with open(default, encoding="utf-8") as f:
            payload = json.load(f)
        print(f"\n{'='*60}")
        print(f"  ForenSynth Timeline Agent – Case: {payload.get('case_id','?')} (example)")
        print(f"{'='*60}\n")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = run_timeline_agent(payload, output_dir=args.output_dir, save_outputs=True)
    except ValidationError as exc:
        print(f"[VALIDATION ERROR] {exc}")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Timeline Version : {result['timeline_version']}")
    print(f"Generated At     : {result['generated_at']}")
    print(f"Total Events     : {len(result['events'])}")
    print(f"Causal Links     : {len(result['causal_links'])}")
    print(f"Uncertainties    : {len(result['uncertainties'])}")
    print(f"Conflicts        : {len(result['conflicts_summary'])}")
    print(f"Total Time       : {result['total_time_sec']:.3f}s")
    if result.get("validation_warnings"):
        print(f"Warnings         : {len(result['validation_warnings'])}")
    print()

    # ── Narrative ─────────────────────────────────────────────────────────────
    exp_layer = ExplainabilityLayer()
    narrative_lines = [
        NarrativeLine(
            timestamp=n["timestamp"], actor=n["actor"], action=n["action"],
            location=n["location"], evidence=n["evidence"],
            confidence=n["confidence"], event_id=n["event_id"],
        )
        for n in result["narrative"]
    ]
    print(exp_layer.format_text_narrative(narrative_lines))

    if args.print_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\nOutputs written to {args.output_dir}/")
    case = result['case_id']
    print(f"  • {case}_timeline_V1.json")
    print(f"  • timeline_graph.json")


if __name__ == "__main__":
    main()