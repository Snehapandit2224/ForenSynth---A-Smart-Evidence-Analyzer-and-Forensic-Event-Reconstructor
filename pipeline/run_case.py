#!/usr/bin/env python3
"""
ForenSynth – run_case.py
Runs the complete pipeline loop for a single case.

    ER → Timeline → Critique → Showrunner → (loop back if needed) → done

Usage:
    python pipeline\run_case.py --input cases\cases_atm\CASE_ATM_001_obs_only.json
    python pipeline\run_case.py --case CASE_ATM_001
    python pipeline\run_case.py --input cases\cases_atm\CASE_ATM_001_obs_only.json --no-llm
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
log = logging.getLogger("forensynth.run_case")

MAX_LOOPS = 5


def _banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def run_er(mem, obs_data, llm, run_version, human_constraints=None, out_dir=None):
    from entity_resolution import resolve_entities
    _banner(f"ER v{run_version} — {obs_data['case_id']}")
    if human_constraints:
        print(f"  Constraints: {human_constraints.get('must_not_merge', [])}")
    result = resolve_entities(obs_data, llm_enabled=llm,
                              human_constraints=human_constraints)
    mem.save_er_result(result, run_version=run_version)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{obs_data['case_id']}_er_v{run_version}_output.json"
        p.write_text(json.dumps(result, indent=2))
        print(f"  Saved → {p}")
    print(f"  Entities: {result['entity_count']}  "
          f"Classification: {result['output_classification']}")
    return result


def run_timeline(mem, case_id, er_ver, tl_version, out_dir=None):
    from timeline_agent import run_timeline_agent
    _banner(f"Timeline {tl_version} — {case_id}  (ER v{er_ver})")
    payload = mem.load_for_timeline(case_id, er_version=er_ver)
    if not payload:
        raise RuntimeError(f"Cannot load timeline payload for {case_id}")
    tmp = str(out_dir) if out_dir else "./output/timelines"
    result = run_timeline_agent(payload, output_dir=tmp)
    result["timeline_version"] = tl_version
    mem.save_timeline(result)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{case_id}_timeline_{tl_version}.json"
        p.write_text(json.dumps(result, indent=2))
        gp = out_dir / f"{case_id}_timeline_{tl_version}_graph.json"
        if result.get("timeline_graph"):
            gp.write_text(json.dumps(result["timeline_graph"], indent=2))
        print(f"  Saved → {p}")
    print(f"  Events: {len(result['events'])}  "
          f"Causal: {len(result['causal_links'])}")
    return result


def run_critique(mem, case_id, tl_version, crit_version,
                 tl_result=None, out_dir=None):
    from critique_agent import run_critique_agent
    from memory_store import _load
    _banner(f"Critique {crit_version} — {case_id}  (Timeline {tl_version})")

    # Build payload — use tl_result directly to avoid DB timing issues
    if tl_result is not None:
        er_row = mem._query_one(
            "SELECT full_json FROM er_runs WHERE case_id=? "
            "ORDER BY run_version DESC LIMIT 1", (case_id,))
        obs_rows = mem._query(
            "SELECT raw_json FROM observations "
            "WHERE case_id=? ORDER BY time_offset", (case_id,))
        payload = {
            "timeline":     tl_result,
            "graph":        mem._build_nx_graph(
                                tl_result.get("timeline_graph", {})),
            "observations": [_load(r["raw_json"]) for r in obs_rows],
            "er_result":    _load(er_row["full_json"]) if er_row else {},
        }
    else:
        payload = mem.load_for_critique(case_id, tl_version=tl_version)
        if not payload:
            raise RuntimeError(
                f"Cannot load critique payload for {case_id} "
                f"tl_version={tl_version}")

    result = run_critique_agent(payload, critique_version=crit_version)
    mem.save_critique(result)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{case_id}_critique_{crit_version}.json"
        p.write_text(json.dumps(result, indent=2))
        print(f"  Saved → {p}")

    print(f"  Score: {result['overall_score']:.2f}  "
          f"Gaps: {len(result['gaps'])}  "
          f"requires_revision: {result['requires_revision']}  "
          f"Action: {result['recommended_action']}")
    return result


def run_showrunner(mem, case_id, tl_ver, crit_ver, out_dir=None, prev_decision=None):
    from showrunner_agent import run_showrunner as _sr
    _banner(f"Showrunner — {case_id}")
    payload = mem.load_for_showrunner(case_id,
                                       tl_version=tl_ver,
                                       crit_version=crit_ver)
    if not payload:
        raise RuntimeError(f"Cannot load showrunner payload for {case_id}")
    # FIX: inject full previous decision as previous_constraints
    # The DB only stores sparse er_constraints row, losing iter_log, belief_state etc.
    # Passing the full previous result preserves convergence detection state.
    if prev_decision:
        payload["previous_constraints"] = prev_decision.get("er_constraints", {})
    result = _sr(payload)

    if result.get("er_constraints"):
        c = result["er_constraints"]
        # Save constraints for both re_run_er and re_run_timeline
        # so recurrence detection works on subsequent loops
        ver = mem.save_er_constraints(
            case_id,
            constraints={
                "must_not_merge":     c.get("must_not_merge", []),
                "must_merge":         c.get("must_merge", []),
                "soft_hints":         c.get("soft_hints", {}),
                "previous_gap_types": c.get("previous_gap_types", [
                    g.get("gap_type") or g.get("check","")
                    for g in payload["critique"].get("gaps", [])
                ]),
            },
            reason=result.get("reasoning", "")[:500],
        )
        print(f"  Constraints v{ver} saved (action={result['action']})")

    mem.save_showrunner_run({
        **result,
        "action_taken":       result["action"],
        "input_tl_version":   tl_ver,
        "input_crit_version": crit_ver,
        "output_tl_version":  result["output_tl_version"],
    })

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{case_id}_showrunner_{crit_ver}.json"
        p.write_text(json.dumps(result, indent=2))
        print(f"  Saved → {p}")

    print(f"  Action: {result['action']}  "
          f"Next TL: {result['output_tl_version']}")
    print(f"  Reason: {result['reasoning'][:100]}")
    return result


def _next_tl(v):
    try: return f"V{int(v.lstrip('V')) + 1}"
    except: return "V2"


def main():
    p = argparse.ArgumentParser(description="ForenSynth Full Pipeline Loop")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to obs_only JSON (first run)")
    group.add_argument("--case",  help="Case ID already in DB")
    p.add_argument("--no-llm",  action="store_true")
    p.add_argument("--output",  default="./output")
    args = p.parse_args()

    llm = not args.no_llm and bool(os.environ.get("GROQ_API_KEY", ""))
    out = Path(args.output)

    from memory_store import ForenSynthMemory
    mem = ForenSynthMemory()

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
    if er_ver == 0:
        run_er(mem, obs_data, llm, run_version=1, out_dir=out / "er")
        er_ver = 1

    # ── Main loop ─────────────────────────────────────────────────────────────
    tl_result    = None   # timeline result from this iteration
    tl_ver       = None   # current timeline version
    crit_ver     = None   # current critique version
    prev_decision = None  # full previous Showrunner result for convergence tracking

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

        # Showrunner — pass previous decision for convergence tracking
        decision = run_showrunner(mem, case_id, tl_ver, crit_ver,
                                  out_dir=out / "showrunner",
                                  prev_decision=prev_decision)
        prev_decision = decision  # carry forward for next loop
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

    # ── Final status ──────────────────────────────────────────────────────────
    mem.print_status(case_id)


if __name__ == "__main__":
    main()