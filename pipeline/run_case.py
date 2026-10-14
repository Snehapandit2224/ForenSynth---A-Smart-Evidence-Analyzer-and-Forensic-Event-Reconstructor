#!/usr/bin/env python3
r"""
ForenSynth - run_case.py
Runs the complete pipeline loop for a single case.

    ER -> Timeline -> Critique -> Showrunner -> (loop back if needed) -> done

The loop runs on the LangGraph orchestrator in graph.py; --legacy runs the
original for-loop below instead (kept for comparison). Both call the same
stage functions in steps.py.

Usage:
    python pipeline/run_case.py --input cases/CASE_ATM_001_obs_only.json
    python pipeline/run_case.py --case CASE_ATM_001
    python pipeline/run_case.py --input cases/CASE_ATM_001_obs_only.json --no-llm
    python pipeline/run_case.py --input cases/CASE_ATM_001_obs_only.json --legacy

    # raw evidence: field agents -> merge -> the same loop
    python pipeline/run_case.py --case CASE_FIR_001 --fir samples/fir/FIR_001_chain_snatching.txt \\
        --text-inputs samples/fir/FIR_001_witness_statement.txt samples/fir/FIR_001_chat.json
    python pipeline/run_case.py --case CASE_MULTI_001 \\
        --obs-files cases/CASE_ENRON_001_obs_only.json cases/CASE_LIBRISPEECH_001_obs_only.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Windows consoles default to a legacy codepage (cp1252) that can't encode
# the "→" characters used throughout the pipeline's print() output, which
# crashes every run with a UnicodeEncodeError. Force UTF-8 stdout/stderr.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
log = logging.getLogger("forensynth.run_case")

from steps import (  # noqa: E402
    MAX_LOOPS, _banner, run_critique, run_er, run_eval, run_report,
    run_scene_video, run_showrunner, run_timeline,
)


def _next_tl(v):
    try: return f"V{int(v.lstrip('V')) + 1}"
    except: return "V2"


def main():
    p = argparse.ArgumentParser(description="ForenSynth Full Pipeline Loop")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to obs_only JSON (first run)")
    group.add_argument("--case",  help="Case ID already in DB - or, with raw evidence below, "
                                       "the ID of the new case to build from it")
    ev = p.add_argument_group("raw evidence (runs the field agents first; needs --case)")
    ev.add_argument("--fir", help="FIR text file - anchors the case (fir_agent.py)")
    ev.add_argument("--text-inputs", nargs="+", help="Text evidence files (text_agent.py)")
    ev.add_argument("--audio-manifest", help="JSON list of recordings (audio_agent.py): "
                                             '[{"path", "start_time", "location", "source"}, ...]')
    ev.add_argument("--obs-files", nargs="+", help="Ready-made obs_only files to merge in")
    ev.add_argument("--adopt-role-hints", action="store_true",
                    help="Copy LLM role hints with confidence >= 0.7 into `role` at intake")
    p.add_argument("--no-llm",  action="store_true")
    p.add_argument("--must-merge", default=None,
                   help="JSON list of alias pairs the ER agent must cluster "
                        "into the same entity, applied on the first ER run, "
                        "e.g. '[[\"Person_14\", \"Speaker_C\"]]'")
    p.add_argument("--must-not-merge", default=None,
                   help="JSON list of alias pairs the ER agent must keep as "
                        "separate entities, applied on the first ER run")
    p.add_argument("--output",  default="./output")
    p.add_argument("--db",      default="./forensynth.db", help="Path to forensynth.db")
    p.add_argument("--skip-video",  action="store_true", default=False,
                   help="Skip scene-reconstruction video generation")
    p.add_argument("--skip-report", action="store_true", default=False,
                   help="Skip explainability PDF report generation")
    p.add_argument("--skip-eval",   action="store_true", default=False,
                   help="Skip ground-truth evaluation")
    p.add_argument("--legacy", action="store_true", default=False,
                   help="Run the original for-loop orchestrator instead of the LangGraph one (graph.py)")
    args = p.parse_args()

    llm = not args.no_llm and bool(os.environ.get("GROQ_API_KEY", ""))
    out = Path(args.output)
    evidence = {"fir": args.fir, "text": args.text_inputs,
                "audio_manifest": args.audio_manifest, "obs_files": args.obs_files}
    if any(evidence.values()):
        if not args.case:
            p.error("raw evidence needs --case <new case id>")
        if args.legacy:
            p.error("--legacy runs the loop only; evidence intake needs the graph")

    if not args.legacy:
        from graph import run_pipeline
        initial_constraints = None
        if args.must_merge or args.must_not_merge:
            initial_constraints = {
                "must_merge":     json.loads(args.must_merge) if args.must_merge else [],
                "must_not_merge": json.loads(args.must_not_merge) if args.must_not_merge else [],
            }
        try:
            run_pipeline(args.input, args.case, evidence=evidence,
                         adopt_role_hints=args.adopt_role_hints,
                         llm=llm, output=args.output, db=args.db,
                         initial_constraints=initial_constraints,
                         skip_video=args.skip_video, skip_report=args.skip_report,
                         skip_eval=args.skip_eval)
        except (FileNotFoundError, LookupError, RuntimeError) as exc:
            print(f"[ERROR] {exc}"); sys.exit(1)
        return

    from memory_store import ForenSynthMemory
    mem = ForenSynthMemory(args.db)

    # ── Load / register case ──────────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_path}"); sys.exit(1)
        with open(input_path) as f:
            obs_data = json.load(f)
        mem.save_case(obs_data)
        case_id = obs_data["case_id"]
    else:
        case_id  = args.case
        obs_data = mem.load_observations(case_id)
        if not obs_data:
            print(f"[ERROR] Case {case_id} not in DB. Use --input first.")
            sys.exit(1)

    _banner(f"ForenSynth Pipeline — {case_id}")
    print(f"  LLM     : {'ENABLED' if llm else 'DISABLED'}")
    print(f"  Output  : {out}")
    print(f"  Max loops: {MAX_LOOPS}")

    # ── Check existing state ──────────────────────────────────────────────────
    versions = mem.get_latest_versions(case_id)
    er_ver   = versions["er_version"] or 0

    # ── Step 1: ER ────────────────────────────────────────────────────────────
    initial_constraints = None
    if args.must_merge or args.must_not_merge:
        initial_constraints = {
            "must_merge":     json.loads(args.must_merge) if args.must_merge else [],
            "must_not_merge": json.loads(args.must_not_merge) if args.must_not_merge else [],
        }

    er_result = None
    if er_ver == 0:
        er_result = run_er(mem, obs_data, llm, run_version=1,
                           human_constraints=initial_constraints,
                           out_dir=out / "er")
        er_ver = 1

    # ── Main loop ─────────────────────────────────────────────────────────────
    tl_result    = None   # timeline result from this iteration
    tl_ver       = None   # current timeline version
    crit_ver     = None   # current critique version
    prev_decision = None  # full previous Showrunner result for convergence tracking
    critiques_by_round   = {}  # "critique_c{n}"   -> critique dict, for the explainability report
    showrunner_by_round  = {}  # "showrunner_c{n}" -> decision dict, for the explainability report

    for loop in range(1, MAX_LOOPS + 1):
        next_tl   = f"V{loop}"
        next_crit = f"C{loop}"

        # Timeline
        _banner_msg = f"Loop {loop}/{MAX_LOOPS}"
        print(f"\n  {_banner_msg}")
        tl_result = run_timeline(mem, case_id, er_ver, next_tl,
                                  out_dir=out / "timelines")
        tl_ver = next_tl

        # Critique — pass tl_result directly, no DB reload needed
        critique = run_critique(mem, case_id, tl_ver, next_crit,
                                tl_result=tl_result,
                                out_dir=out / "critiques")
        crit_ver = next_crit
        critiques_by_round[f"critique_c{loop}"] = critique

        # Showrunner — pass previous decision for convergence tracking
        decision = run_showrunner(mem, case_id, tl_ver, crit_ver,
                                  out_dir=out / "showrunner",
                                  prev_decision=prev_decision)
        prev_decision = decision  # carry forward for next loop
        showrunner_by_round[f"showrunner_c{loop}"] = decision
        action = decision["action"]

        # ── Decide next step ──────────────────────────────────────────────────
        if action == "no_action":
            _banner(f"PIPELINE COMPLETE — {case_id}")
            print(f"  Final timeline : {tl_ver}")
            print(f"  Score          : {critique.get('overall_score', 0):.2f}")
            print(f"  No revision needed.")
            break

        if action == "human_review":
            _banner(f"ESCALATED TO HUMAN REVIEW — {case_id}")
            print(f"  Reason: {decision['reasoning']}")
            break

        if action == "re_run_er":
            if loop == MAX_LOOPS:
                _banner(f"MAX REVISIONS REACHED — {case_id}")
                print(f"  Cannot run ER again — at loop {loop}/{MAX_LOOPS}")
                break
            er_ver += 1
            c = mem.load_er_constraints(case_id)
            human_constraints = {
                "must_not_merge": c.get("must_not_merge", []),
                "must_merge":     c.get("must_merge", []),
                "soft_hints":     c.get("soft_hints", {}),
            } if c else None
            run_er(mem, obs_data, llm, run_version=er_ver,
                   human_constraints=human_constraints,
                   out_dir=out / "er")
            tl_result = None  # will rebuild next iteration

        elif action == "re_run_timeline":
            if loop == MAX_LOOPS:
                _banner(f"MAX REVISIONS REACHED — {case_id}")
                break
            tl_result = None  # will rebuild next iteration
            # continue loop — next iteration produces next TL version

        if loop == MAX_LOOPS:
            _banner(f"MAX REVISIONS REACHED — {case_id}")
            print(f"  Stopped at V{loop}. Manual review recommended.")

    # ── Post-pipeline outputs: scene video + explainability report ─────────────
    # Runs once the loop above has stopped — no_action, human_review, or the
    # C3 version ceiling all fall through to here since none of those paths
    # re-enter the loop. Everything is passed in from memory; nothing is
    # re-read from the JSON files already written to disk.
    pipeline_outputs = {
        "er":          er_result or {},
        "timeline_v3": tl_result or {},
        **critiques_by_round,
        **showrunner_by_round,
    }

    if not args.skip_video:
        run_scene_video(case_id, out)
    if not args.skip_report:
        run_report(case_id, out, pipeline_outputs)
    if not args.skip_eval:
        run_eval(case_id, out, tl_ver)

    # ── Final status ──────────────────────────────────────────────────────────
    mem.print_status(case_id)


if __name__ == "__main__":
    main()