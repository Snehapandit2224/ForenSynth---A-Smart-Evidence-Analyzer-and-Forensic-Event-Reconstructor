"""
ForenSynth-X+ Main Entry Point

Usage:
    # Basic — fully offline
    python main.py
    python main.py --domain ATM_Robbery
    python main.py --suspects 2 --witnesses 2
    python main.py --batch 5 --output ./cases/
    python main.py --seed 123 --validate

    # With Cohere enrichment (single call: rewrites FIR + all observations)
    python main.py --enrich --api-key co-...
    python main.py --domain Office_Theft --enrich --api-key co-... --output ./cases/
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  # non-standard stdout (e.g. some test runners) - fall back silently

# Ensure the package root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from config import GeneratorConfig, NoiseConfig
from generator import ForenSynthGenerator
from utils import export_case, export_observations_only, extract_observations_only, pretty_print_case, validate_case_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ForenSynth-X+: Synthetic Forensic Case Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --domain ATM_Robbery --suspects 2 --witnesses 2\n"
            "  python main.py --batch 10 --output ./cases/ --validate\n"
            "  python main.py --enrich --api-key co-... --output ./cases/\n"
        ),
    )
    parser.add_argument(
        "--domain",
        choices=["ATM_Robbery", "Office_Theft", "Communication"],
        default=None,
        help="Force a specific domain (default: random).",
    )
    parser.add_argument(
        "--suspects", type=int, default=None,
        help="Number of suspects (default: random 1–3).",
    )
    parser.add_argument(
        "--witnesses", type=int, default=None,
        help="Number of witnesses (default: random 1–2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Number of cases to generate (default: 1).",
    )
    parser.add_argument(
        "--start-index", type=int, default=1,
        help=(
            "Starting case index for file naming (default: 1). "
            "Use this to avoid overwriting existing files — e.g. if you "
            "already have 10 cases, pass --start-index 11 to generate "
            "CASE_XXX_011 onwards."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for JSON files (default: print to stdout).",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run schema validation and print results.",
    )

    # --- LLM enrichment flags ---
    parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help=(
            "Use Cohere API to enrich the full case in a single API call. "
            "Rewrites: fir.description, fir.location, observations[].content, observations[].source. "
            "Ground truth, timestamps, noise_tags, and aliases are never touched. "
            "Requires --api-key or COHERE_API_KEY env variable."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "Cohere API key (co-...). "
            "Can also be set via COHERE_API_KEY environment variable."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve API key: CLI flag takes priority, then environment variable
    api_key: str | None = args.api_key or os.environ.get("COHERE_API_KEY")

    if args.enrich and not api_key:
        print(
            "[ERROR] --enrich requires a Cohere API key.\n"
            "        Provide it via --api-key co-... or set COHERE_API_KEY env variable."
        )
        sys.exit(1)

    cfg = GeneratorConfig(
        seed=args.seed,
        noise=NoiseConfig(),
        enrich=args.enrich,
        cohere_api_key=api_key,
    )

    gen = ForenSynthGenerator(config=cfg, case_index=args.start_index)

    if args.batch == 1:
        cases = [gen.generate_case(
            domain=args.domain,
            suspect_count=args.suspects,
            witness_count=args.witnesses,
        )]
    else:
        cases = gen.generate_batch(n=args.batch, domain=args.domain)

    for case in cases:
        if args.validate:
            errors = validate_case_schema(case)
            if errors:
                print(f"[VALIDATION ERRORS] {case['case_id']}:")
                for e in errors:
                    print(f"  ✗ {e}")
            else:
                tag = " [enriched]" if case["fir"].get("enriched", False) else ""
                print(f"[VALID] {case['case_id']} — schema OK ✓{tag}")

        if args.output:
            out_dir = Path(args.output)
            out_path = out_dir / f"{case['case_id']}.json"
            obs_path = out_dir / f"{case['case_id']}_obs_only.json"
            export_case(case, out_path)
            export_observations_only(case, obs_path)
            print(f"Saved → {out_path}")
            print(f"Saved → {obs_path}  (observations only)")
        else:
            pretty_print_case(case)
            print("\n--- OBSERVATIONS ONLY (no event_ref / noise_tags) ---")
            import json as _json
            print(_json.dumps(extract_observations_only(case), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()