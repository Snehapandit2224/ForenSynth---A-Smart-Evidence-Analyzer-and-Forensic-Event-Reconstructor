"""
ForenSynth-X+ Main Entry Point

Usage:
    # Basic — fully offline
    python main.py
    python main.py --domain ATM_Robbery
    python main.py --suspects 2 --witnesses 2
    python main.py --batch 5 --output ./cases/
    python main.py --seed 123 --validate

    # Realistic field-agent output (see realism.py), into its own folder
    python main.py --batch 13 --realistic --output ../cases_realistic/
    python main.py --hide-roles --per-source-labels --appearance --output ./cases/

    # With Cohere enrichment (single call: rewrites FIR + all observations)
    python main.py --enrich --api-key co-...
    python main.py --domain Office_Theft --enrich --api-key co-... --output ./cases/
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the package root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from config import GeneratorConfig, NoiseConfig
from generator import ForenSynthGenerator
from realism import RealismConfig
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

    # --- Realism flags: make observations look like REAL field-agent output.
    # All off by default; with none set, output is unchanged.
    real = parser.add_argument_group(
        "realism", "Make observations look like real field-agent output (see realism.py)")
    real.add_argument("--appearance", action="store_true",
                      help="Every person gets a look (clothing, build); video observations carry it "
                           "as `appearance` and witnesses describe it.")
    real.add_argument("--appearance-noise", type=float, default=0.2, metavar="RATE",
                      help="Chance a description is partial or a colour slightly off (default 0.2).")
    real.add_argument("--alias-variants", action="store_true",
                      help="Text names people inconsistently: 'Ramesh Kumar' / 'Ramesh' / 'Kumar' / "
                           "'R. Kumar', and two people share a surname.")
    real.add_argument("--voice-prints", action="store_true",
                      help="Audio observations carry a per-speaker identity_embedding (noisy per recording).")
    real.add_argument("--hide-roles", action="store_true",
                      help="Every role 'unknown' plus a role_hint with a confidence - mostly right, "
                           "sometimes wrong, sometimes low confidence.")
    real.add_argument("--per-source-labels", action="store_true",
                      help="A new label per camera / recording (cam_Person_NN, rec_SPEAKER_00) "
                           "instead of one alias per person per modality.")
    real.add_argument("--reported-events", action="store_true",
                      help="Witnesses describe what a suspect did ('a man in a black hoodie came out "
                           "fast'): the actor is the suspect, the witness is reported_by.")
    real.add_argument("--undiarized-audio", type=float, default=0.0, metavar="RATE",
                      help="Chance a recording has no speaker separation: one shared Speaker_Unknown "
                           "label, extraction_status=undiarized (default 0 = off).")
    real.add_argument("--timestamp-gaps", type=float, default=0.0, metavar="RATE",
                      help="Chance a video/text observation has no timestamp (default 0 = off).")
    real.add_argument("--mixed-timezones", action="store_true",
                      help="Some sources write timestamps with 'Z' or '+05:30' (same instant).")
    real.add_argument("--fir-errors", action="store_true",
                      help="The FIR gets a head-count or the time window wrong (truth in "
                           "ground_truth.fir_errors).")
    real.add_argument("--action-labels", action="store_true",
                      help="Each observation names its action (APPROACH, ENTER...) as the extractors "
                           "do - mostly right, sometimes missing or wrong.")
    real.add_argument("--agent-metadata", action="store_true",
                      help="Time provenance as the agents report it (video time_source, text/audio "
                           "time_basis) and denial tags.")
    real.add_argument("--realistic", action="store_true",
                      help="All of the above at standard rates.")
    real.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium",
                      help="Scale every realism rate (x0.5 / x1 / x1.5; hard = 2 FIR errors).")

    return parser.parse_args()


def realism_config(args: argparse.Namespace) -> RealismConfig | None:
    """The RealismConfig the flags ask for, or None (clean output)."""
    if args.realistic:
        cfg = RealismConfig.realistic(args.difficulty)
        cfg.appearance_noise = args.appearance_noise
        return cfg
    cfg = RealismConfig(
        appearance=args.appearance, appearance_noise=args.appearance_noise,
        alias_variants=args.alias_variants, voice_prints=args.voice_prints,
        hide_roles=args.hide_roles, per_source_labels=args.per_source_labels,
        reported_events=args.reported_events, undiarized_audio=args.undiarized_audio,
        timestamp_gaps=args.timestamp_gaps, mixed_timezones=args.mixed_timezones,
        fir_errors=args.fir_errors, action_labels=args.action_labels,
        agent_metadata=args.agent_metadata, difficulty=args.difficulty,
    )
    return cfg if cfg.enabled() else None


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
        realism=realism_config(args),
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