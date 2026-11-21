#!/usr/bin/env python3
"""
ForenSynth - Field Agent Orchestrator

Coordinates the per-modality field agents (text_agent.py and any future
modality agent), merges their independently-numbered
obs_only outputs into one coherent case, and hands the merged case to the
memory store - the single funnel point before ER -> Timeline -> Critique
-> Showrunner ever run.

Why an orchestrator instead of each field agent writing to memory
directly: every field agent numbers its own observations from O1
independently, since each was designed to stand alone as its own case.
Under one merged case_id those numbers collide - and
memory_store.save_case()'s INSERT OR IGNORE means a collision is silently
DROPPED, not errored (see memory/memory_store.py). Only a component that
sees every modality's observations at once can safely renumber them; that
is the one job a bare "combiner" and three independently-writing agents
both fail to do safely. This module is that component. It also owns the
one other thing no single field agent can decide for itself: whether to
proceed when one modality's agent fails (partial-failure tolerant, by
design) and where the obs_only ground-truth-leak fields
(event_ref/noise_tags - see GENERATOR_FIXED/utils.py:
extract_observations_only) get stripped a final time before anything
reaches the memory store.

The existing ER -> Timeline -> Critique -> Showrunner loop already has its
own orchestrator (pipeline/run_case.py) and is left untouched - this
module only covers the gap in front of it (multi-modal evidence intake),
and can optionally hand off into that existing loop with --run-pipeline.

Usage:
    # Merge already-generated obs_only files:
    python agents/field_agent/orchestrator.py --case-id CASE_MULTI_001 \\
        --from-files cases/CASE_ENRON_001_obs_only.json cases/CASE_LIBRISPEECH_001_obs_only.json

    # Run every registered field agent fresh, then merge:
    python agents/field_agent/orchestrator.py --case-id CASE_MULTI_001 --run-agents

    # ...and immediately run the full ER->Timeline->Critique->Showrunner loop:
    python agents/field_agent/orchestrator.py --case-id CASE_MULTI_001 --run-agents --run-pipeline

    # FIR-anchored case: FIR + real text evidence:
    python agents/field_agent/orchestrator.py --case-id CASE_FIR_001 \\
        --fir samples/fir/FIR_001_chain_snatching.txt \\
        --text-inputs samples/fir/FIR_001_witness_statement.txt samples/fir/FIR_001_chat.json

With --fir, fir_agent.py processes the FIR: its record goes to the separate
fir_records table (partial ground truth), its summary becomes the merged
case's `fir` block, and its narrative observations (source "fir",
fir_derived: true) are merged in like any other modality.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parents[3]
for _p in (_ROOT / "memory", _ROOT):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("forensynth.orchestrator")

# Registered field agents: add a new entry (e.g. a video_agent.py, once one
# exists) and it merges automatically - nothing else in this file needs to
# change. Each agent is invoked as its own subprocess with its own
# sub-case-id, so one agent's internal numbering never touches another's;
# this orchestrator only ever looks at the resulting obs_only files.
FIELD_AGENTS: list[dict[str, Any]] = [
    {"script": "text_agent.py",  "modality": "text",  "extra_args": []},
]

# Ground-truth/bookkeeping fields the obs_only contract excludes (see
# GENERATOR_FIXED/utils.py:extract_observations_only). text_agent.py no longer
# emits these, but this is the last point before the memory store
# regardless - stripped again here as a safety net in case a source file
# came from somewhere else (hand-edited, an older agent version, a
# third-party export, etc).
_DROP_FIELDS = {"event_ref", "noise_tags"}


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=str(_ROOT)).returncode


def run_fir_agent(case_id: str, fir_path: Path, staging_dir: Path,
                  no_llm: bool = False) -> tuple[Path, Path]:
    """FIR failure IS fatal (unlike a modality agent): a case explicitly
    anchored on an FIR must not silently proceed without it."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    log.info("Running fir_agent.py on %s ...", fir_path)
    rc = _run([sys.executable, str(Path(__file__).parent / "fir_agent.py"), "--case-id", case_id,
               "--input", str(fir_path), "--output", str(staging_dir),
               *(["--no-llm"] if no_llm else [])])
    rec, obs = staging_dir / f"{case_id}_fir.json", staging_dir / f"{case_id}__fir_obs_only.json"
    if rc != 0 or not rec.exists() or not obs.exists():
        raise RuntimeError(f"fir_agent.py failed on {fir_path} (exit {rc}).")
    return rec, obs


def _text_args(inputs: list[Path] | None, reference_date: str | None) -> list[str]:
    """Extra text_agent.py args. The FIR's incident date is the reference for
    statements that give a day but no year ("Friday 14th March")."""
    args = ["--input", *map(str, inputs)] if inputs else []
    return args + (["--reference-date", reference_date] if reference_date else [])


def run_text_agent(case_id: str, inputs: list[Path], staging_dir: Path,
                   reference_date: str | None = None, no_llm: bool = False) -> Path | None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    sub_case_id = f"{case_id}__text"
    log.info("Running text_agent.py on %s ...", ", ".join(map(str, inputs)))
    rc = _run([sys.executable, str(Path(__file__).parent / "text_agent.py"), "--case-id", sub_case_id,
               "--output", str(staging_dir), *_text_args(inputs, reference_date),
               *(["--no-llm"] if no_llm else [])])
    out = staging_dir / f"{sub_case_id}_obs_only.json"
    if rc != 0 or not out.exists():
        log.warning("text_agent.py failed (exit %s) - continuing without text evidence.", rc)
        return None
    return out


AUDIO_AGENT = _ROOT / "agents" / "field_agents" / "audio_agent.py"


def run_audio_agent(case_id: str, manifest: Path | str | list, staging_dir: Path,
                    no_llm: bool = False) -> Path | None:
    """Audio recordings listed in one or more manifests ([{path, start_time,
    location, source?}], see audio_agent.py) - or folders of per-recording
    *manifest*.json - -> one obs_only file. Like text, a failure skips the
    modality rather than the case."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    sub_case_id = f"{case_id}__audio"
    manifests = [str(m) for m in (manifest if isinstance(manifest, (list, tuple)) else [manifest])]
    log.info("Running audio_agent.py on %s ...", ", ".join(manifests))
    rc = _run([sys.executable, str(AUDIO_AGENT), "--case-id", sub_case_id,
               "--manifest", *manifests, "--output", str(staging_dir),
               *(["--no-llm"] if no_llm else [])])
    out = staging_dir / f"{sub_case_id}_obs_only.json"
    if rc != 0 or not out.exists():
        log.warning("audio_agent.py failed (exit %s) - continuing without audio evidence.", rc)
        return None
    return out


def run_field_agents(case_id: str, staging_dir: Path, text_inputs: list[Path] | None = None,
                     reference_date: str | None = None) -> list[Path]:
    """Invoke every registered field agent as its own subprocess, each
    producing its own obs_only file under staging_dir, namespaced under
    its own sub-case-id (<case_id>__<modality>) so agents never see or
    collide with each other.

    A failing agent (bad sample slice, model load error, dataset hiccup,
    ...) is logged and skipped, not fatal - the merge proceeds with
    whatever modalities actually succeeded. Deciding that is exactly the
    kind of call a bare data-merge step has no place to make; it belongs
    here."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    for agent in FIELD_AGENTS:
        sub_case_id = f"{case_id}__{agent['modality']}"
        script = Path(__file__).parent / agent["script"]
        cmd = [sys.executable, str(script), "--case-id", sub_case_id,
               "--output", str(staging_dir), *agent["extra_args"]]
        if agent["modality"] == "text":
            cmd += _text_args(text_inputs, reference_date)
        log.info("Running %s (modality=%s) ...", agent["script"], agent["modality"])
        result = subprocess.run(cmd, cwd=str(_ROOT))
        out_path = staging_dir / f"{sub_case_id}_obs_only.json"
        if result.returncode != 0 or not out_path.exists():
            log.warning("%s failed (exit %s) - skipping this modality, continuing with the rest.",
                        agent["script"], result.returncode)
            continue
        produced.append(out_path)

    if not produced:
        raise RuntimeError("Every registered field agent failed - nothing to merge.")
    return produced


def _clean_observation(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if k not in _DROP_FIELDS}


def _parse_ts(value) -> datetime | None:
    """An observation timestamp as a naive datetime, or None if it has none.

    Agents disagree on format: text writes naive local times, audio and
    video may write an offset or "Z" - and Python refuses to compare naive
    with aware datetimes, which crashed the merge. Aware times are converted
    to UTC and made naive, the same reading shared.parse_timestamp() gives
    naive times (UTC) everywhere downstream."""
    s = str(value if value is not None else "").strip()
    if s.lower() in ("", "none", "null"):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _ts_key(obs: dict) -> datetime:
    # Unparseable/missing timestamp sorts last rather than crashing the merge.
    return _parse_ts(obs.get("timestamp")) or datetime.max


def adopt_role_hints(observations: list[dict], min_conf: float = 0.7) -> int:
    """Opt-in (--adopt-role-hints): promote confident LLM role hints into
    `role` at intake. Off by default - hints stay hints unless you ask."""
    n = 0
    for o in observations:
        if (o.get("role", "unknown") == "unknown" and o.get("role_hint") not in (None, "", "unknown")
                and float(o.get("role_hint_confidence", 0)) >= min_conf):
            o["role"] = o["role_hint"]
            o["role_source"] = "llm_hint"
            n += 1
    return n


def merge_cases(case_id: str, source_paths: list[Path], fir_summary: dict | None = None) -> dict:
    """Merge N per-modality obs_only case files into one case:
      1. concatenate observations from every source (dropping the
         ground-truth-leak fields a second time, see _DROP_FIELDS)
      2. sort by absolute timestamp - interleaves modalities chronologically
         instead of stacking one modality's whole block before the next
      3. renumber obs_id sequentially (O1..On) and recompute time_offset
         relative to the merged case's own earliest timestamp

    Step 3 is the actual fix for the numbering collision discussed above:
    it happens exactly once, here, with full visibility of every
    observation across every modality - which is precisely what none of
    the field agents, writing independently, can have.
    """
    all_obs: list[dict] = []
    modalities_seen: set[str] = set()
    sources_meta: list[str] = []

    for path in source_paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        sources_meta.append(f"{case.get('case_id', path.stem)} ({case.get('domain', '?')})")
        for obs in case.get("observations", []):
            cleaned = _clean_observation(obs)
            modalities_seen.add(cleaned.get("modality", "unknown"))
            all_obs.append(cleaned)

    if not all_obs:
        raise RuntimeError(f"No observations found across {len(source_paths)} source file(s).")

    raw_ts = [str(o.get("timestamp") or "") for o in all_obs if _parse_ts(o.get("timestamp"))]
    aware = sum(1 for t in raw_ts if t.endswith("Z") or re.search(r"[+-]\d{2}:?\d{2}$", t))
    if 0 < aware < len(raw_ts):
        log.warning("Merging %d timestamp(s) with a timezone and %d without: the ones without "
                    "are read as UTC. If they are local times, the modalities will be offset "
                    "from each other by the local UTC offset.", aware, len(raw_ts) - aware)

    all_obs.sort(key=_ts_key)
    dated = [ts for ts in (_parse_ts(o.get("timestamp")) for o in all_obs) if ts]
    base_ts = dated[0] if dated else None

    merged_obs = []
    for i, obs in enumerate(all_obs, start=1):
        ts = _parse_ts(obs.get("timestamp"))
        if ts is None:
            # Undated (e.g. a video clip whose clock couldn't be read): an
            # empty timestamp is the pipeline's "no time" - ER and Timeline
            # treat it as unknown, not as the case's first moment.
            merged_obs.append({**obs, "obs_id": f"O{i}", "timestamp": "",
                               "time_offset": max(0, int(obs.get("time_offset") or 0))})
            continue
        merged_obs.append({**obs, "obs_id": f"O{i}", "timestamp": ts.isoformat(timespec="seconds"),
                           "time_offset": max(0, int((ts - base_ts).total_seconds()))})

    if fir_summary:
        # FIR incident window, re-expressed as offsets from this merged case's base.
        def off(iso):
            ts = _parse_ts(iso)
            return max(0, int((ts - base_ts).total_seconds())) if ts and base_ts else None
        start, end = off(fir_summary.get("incident_datetime")), off(fir_summary.get("incident_datetime_end"))
        window = ([start, end] if start is not None and end is not None
                  else [0, max(o["time_offset"] for o in merged_obs)])
        return {
            "case_id": case_id,
            "domain": "FIR_Case",
            "template": "Field_Agent_Orchestrated",
            "fir": {**fir_summary, "time_window": window},
            "observations": merged_obs,
        }

    return {
        "case_id": case_id,
        "domain": "Multi_Modal",
        "template": "Field_Agent_Orchestrated",
        "fir": {
            "crime_type": "unknown - real multi-modal record (field-agent pilot)",
            "location": "multiple sources",
            "time_window": [0, max(o["time_offset"] for o in merged_obs)],
            "description": (
                f"Merged case from {len(source_paths)} field-agent source(s) "
                f"({'; '.join(sources_meta)}), spanning modalities: "
                f"{', '.join(sorted(modalities_seen))}. {len(merged_obs)} observations total. "
                "Confidence values are carried through as each field agent computed them "
                "(text: LLM-scored or heuristic fallback) - "
                "no forced cross-modal calibration is applied here; see agents/field_agent/orchestrator.py."
            ),
            "roles": {},
        },
        "observations": merged_obs,
    }


def save_to_memory(case: dict, db_path: str, fir_record: dict | None = None) -> None:
    from memory_store import ForenSynthMemory
    mem = ForenSynthMemory(db_path)
    mem.save_case(case)
    log.info("Saved merged case %s to memory store (%s)", case["case_id"], mem.db_path)
    if fir_record:
        mem.save_fir_record({**fir_record, "case_id": case["case_id"]}, case["fir"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge field-agent modality outputs into one case")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--from-files", nargs="+", type=Path,
                     help="Pre-generated obs_only files to merge (skips running agents)")
    ap.add_argument("--run-agents", action="store_true",
                     help="Run every registered field agent fresh before merging")
    ap.add_argument("--fir", type=Path, help="FIR text file - anchors the case (see fir_agent.py)")
    ap.add_argument("--text-inputs", "--text-dir", dest="text_inputs", nargs="+", type=Path,
                     help="Text evidence files, or folders of them, for text_agent.py "
                          "(instead of the Enron pilot)")
    ap.add_argument("--audio-manifest", "--audio-dir", dest="audio_manifest", nargs="+", type=Path,
                     help="Manifests for audio_agent.py - JSON lists of recordings "
                          '[{"path", "start_time", "location", "source"}, ...] - '
                          "or folders of per-recording *manifest*.json")
    ap.add_argument("--adopt-role-hints", action="store_true",
                     help="Copy LLM role hints with confidence >= 0.7 into `role` at intake")
    ap.add_argument("--staging", default="cases/_staging",
                     help="Where fresh field-agent outputs are written before merging")
    ap.add_argument("--output", default="cases")
    ap.add_argument("--db", default="./forensynth.db")
    ap.add_argument("--run-pipeline", action="store_true",
                     help="After saving, immediately run pipeline/run_case.py --case <id>")
    args = ap.parse_args()

    if not (args.from_files or args.run_agents or args.fir or args.text_inputs or args.audio_manifest):
        print("[ERROR] Provide --from-files, --run-agents, --fir, --text-inputs and/or --audio-manifest.")
        sys.exit(1)

    staging = Path(args.staging)
    source_paths: list[Path] = list(args.from_files) if args.from_files else []
    fir_record = fir_summary = None
    if args.fir:
        rec_path, fir_obs_path = run_fir_agent(args.case_id, args.fir, staging)
        fir_record = json.loads(rec_path.read_text(encoding="utf-8"))
        fir_summary = json.loads(fir_obs_path.read_text(encoding="utf-8"))["fir"]
        source_paths.append(fir_obs_path)
    ref = (fir_summary or {}).get("incident_datetime") or (fir_summary or {}).get("date_of_report")
    if args.run_agents:
        source_paths += run_field_agents(args.case_id, staging, args.text_inputs, ref)
    elif args.text_inputs:
        text_out = run_text_agent(args.case_id, args.text_inputs, staging, ref)
        if text_out:
            source_paths.append(text_out)
    if args.audio_manifest:
        audio_out = run_audio_agent(args.case_id, args.audio_manifest, staging)
        if audio_out:
            source_paths.append(audio_out)

    missing = [p for p in source_paths if not p.exists()]
    if missing:
        print(f"[ERROR] Missing source file(s): {missing}")
        sys.exit(1)

    merged = merge_cases(args.case_id, source_paths, fir_summary)
    if args.adopt_role_hints:
        log.info("Adopted %d LLM role hint(s) into `role`.", adopt_role_hints(merged["observations"]))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.case_id}_obs_only.json"
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    log.info("Merged %d source(s) -> %d observations -> %s",
              len(source_paths), len(merged["observations"]), out_path)

    save_to_memory(merged, db_path=args.db, fir_record=fir_record)

    if args.run_pipeline:
        log.info("Launching full pipeline for %s ...", args.case_id)
        subprocess.run(
            [sys.executable, str(_ROOT / "pipeline" / "run_case.py"),
             "--case", args.case_id, "--db", args.db],
            cwd=str(_ROOT),
        )
    else:
        print(f"Run:    python pipeline/run_case.py --case {args.case_id} --db {args.db}")


if __name__ == "__main__":
    main()
