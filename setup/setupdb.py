#!/usr/bin/env python3
"""
ForenSynth – setup_db.py
One-time database setup and migration tool.

Run once after setting DATABASE_URL in your .env:
    python setup_db.py

Then migrate all existing files:
    python setup_db.py --migrate

This scans:
  - Current folder (.) for obs_only JSON files
  - ./output/ for ER output and timeline JSON files
  - Any extra folders you pass with --dirs

Usage:
    python setup_db.py                              # setup schema only
    python setup_db.py --test                       # test connection only
    python setup_db.py --migrate                    # setup + migrate all files
    python setup_db.py --migrate --dirs ./cases ./output  # custom folders
    python setup_db.py --status                     # show all cases in DB
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
# Load .env from project root (one level up from setup/)
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

import sys as _sys
from pathlib import Path as _Path
# Add agents/ and memory/ to path so imports work from any subfolder
_root = _Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("forensynth.setup_db")



def test_connection() -> bool:
    try:
        from memory_store import ForenSynthMemory
        mem = ForenSynthMemory()
        cases = mem.list_cases()
        print(f"Connection OK — {len(cases)} cases currently in database")
        return True
    except Exception as exc:
        print(f"\nConnection FAILED: {exc}")
        print()
        print("Make sure your .env file has:")
        print("  DATABASE_URL=postgresql://postgres:password@db.xxx.supabase.co:5432/postgres")
        print()
        print("For Supabase: Settings → Database → Connection string → URI")
        return False


def _is_obs_only(data: dict) -> bool:
    """Check if a JSON file is an obs_only input file."""
    return (
        "observations" in data
        and "case_id" in data
        and "canonical_entities" not in data   # not an ER output
        and "events" not in data               # not a timeline output
    )

def _is_er_output(data: dict) -> bool:
    """Check if a JSON file is an ER output."""
    return (
        "canonical_entities" in data
        and "case_id" in data
        and "clusters" in data
    )

def _is_timeline_output(data: dict) -> bool:
    """Check if a JSON file is a timeline output."""
    return (
        "events" in data
        and "timeline_version" in data
        and "causal_links" in data
        and "case_id" in data
    )


def migrate_all(folders: list[str]) -> None:
    from memory_store import ForenSynthMemory
    mem = ForenSynthMemory()

    # Collect all JSON files from all folders
    all_json_files = []
    for folder_str in folders:
        folder = Path(folder_str)
        if not folder.exists():
            print(f"  Folder not found: {folder} — skipping")
            continue
        files = list(folder.glob("*.json"))
        print(f"  Found {len(files)} JSON files in {folder}")
        all_json_files.extend(files)

    if not all_json_files:
        print("No JSON files found.")
        return

    # Categorise files
    obs_files      = []
    er_files       = []
    timeline_files = []
    unknown_files  = []

    for f in all_json_files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            if _is_obs_only(data):
                obs_files.append((f, data))
            elif _is_er_output(data):
                er_files.append((f, data))
            elif _is_timeline_output(data):
                timeline_files.append((f, data))
            else:
                unknown_files.append(f)
        except Exception as exc:
            print(f"  Could not read {f.name}: {exc}")

    print(f"\nFound:")
    print(f"  {len(obs_files)} obs_only input files")
    print(f"  {len(er_files)} ER output files")
    print(f"  {len(timeline_files)} timeline output files")
    if unknown_files:
        print(f"  {len(unknown_files)} unrecognised files (skipped)")

    # ── Step 1: Migrate obs_only files ────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Step 1: Migrating {len(obs_files)} obs_only files...")
    print(f"{'─'*50}")

    for f, data in obs_files:
        try:
            mem.save_case(data)
            print(f"  ✓ {f.name}  →  case_id={data['case_id']}")
        except Exception as exc:
            print(f"  ✗ {f.name}: {exc}")

    # ── Step 2: Migrate ER outputs ────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Step 2: Migrating {len(er_files)} ER output files...")
    print(f"{'─'*50}")

    for f, data in er_files:
        try:
            # Detect run version from filename if possible
            name = f.stem
            run_ver = 1
            for part in name.split("_"):
                if part.startswith("v") and part[1:].isdigit():
                    run_ver = int(part[1:])
            mem.save_er_result(data, run_version=run_ver)
            print(f"  ✓ {f.name}  →  case_id={data['case_id']}  entities={data.get('entity_count',0)}")
        except Exception as exc:
            print(f"  ✗ {f.name}: {exc}")

    # ── Step 3: Migrate timeline outputs ──────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Step 3: Migrating {len(timeline_files)} timeline files...")
    print(f"{'─'*50}")

    for f, data in timeline_files:
        try:
            mem.save_timeline(data)
            print(f"  ✓ {f.name}  →  case_id={data['case_id']}  version={data['timeline_version']}  events={len(data.get('events',[]))}")
        except Exception as exc:
            print(f"  ✗ {f.name}: {exc}")

    # ── Final status ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("Migration complete. Database status:")
    print(f"{'='*50}")
    cases = mem.list_cases()
    if not cases:
        print("  No cases found — check if obs_only files were migrated correctly.")
    else:
        print(f"  {'Case':>16}  {'Obs':>4}  {'Entities':>8}  {'TL Versions':>11}  {'Status'}")
        print(f"  {'─'*16}  {'─'*4}  {'─'*8}  {'─'*11}  {'─'*20}")
        for c in cases:
            print(f"  {c['case_id']:>16}  {c['obs_count']:>4}  {c['entity_count']:>8}  "
                  f"{c['tl_versions']:>11}  {c['pipeline_status']}")
    print()


def show_status():
    from memory_store import ForenSynthMemory
    mem = ForenSynthMemory()
    cases = mem.list_cases()
    if not cases:
        print("No cases in database yet.")
        return
    print(f"\n{'='*70}")
    print(f"  ForenSynth Database — {len(cases)} cases")
    print(f"{'='*70}")
    print(f"  {'Case':>16}  {'Obs':>4}  {'Entities':>8}  {'TL':>4}  {'Critiques':>9}  Status")
    print(f"  {'─'*16}  {'─'*4}  {'─'*8}  {'─'*4}  {'─'*9}  {'─'*20}")
    for c in cases:
        print(f"  {c['case_id']:>16}  {c['obs_count']:>4}  {c['entity_count']:>8}  "
              f"{c['tl_versions']:>4}  {c['critique_count']:>9}  {c['pipeline_status']}")
    print()


def main():
    p = argparse.ArgumentParser(description="ForenSynth Database Setup")
    p.add_argument("--test",    action="store_true", help="Test connection only")
    p.add_argument("--migrate", action="store_true", help="Migrate all existing JSON files")
    p.add_argument("--status",  action="store_true", help="Show all cases in database")
    p.add_argument("--dirs",    nargs="+",           help="Extra folders to scan (default: . and ./output)")
    args = p.parse_args()

    print("\nForenSynth Database Setup")
    print("=" * 40)

    if args.test:
        test_connection()
        return

    if args.status:
        if test_connection():
            show_status()
        return

    # Always test connection first
    if not test_connection():
        sys.exit(1)

    if args.migrate:
        # Default: scan current folder + ./output
        folders = args.dirs if args.dirs else [".", "./output"]
        print(f"\nScanning folders: {folders}")
        migrate_all(folders)
    else:
        print("\nSchema is ready.")
        print()
        print("Next steps:")
        print("  1. Run migration:  python setup_db.py --migrate")
        print("  2. Check status:   python setup_db.py --status")
        print("  3. Run pipeline:   python pipeline.py --input CASE_ATM_001_obs_only.json")


if __name__ == "__main__":
    main()