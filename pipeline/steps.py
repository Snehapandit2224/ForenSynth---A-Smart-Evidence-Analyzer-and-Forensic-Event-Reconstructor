#!/usr/bin/env python3
"""
ForenSynth - steps.py
One function per pipeline stage (ER, Timeline, Critique, Showrunner) plus the
lazy importers for the post-loop outputs. Each stage runs its agent, saves the
result to the memory store and writes the JSON output file.

Shared by both orchestrators - run_case.py (the original loop, kept as
--legacy) and graph.py (LangGraph) - so neither imports the other.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for _p in [str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

log = logging.getLogger("forensynth.steps")

MAX_LOOPS = 5


def _import_video():
    """Lazily import the scene-video generator; optional, heavy deps (moviepy/Pillow/numpy)."""
    try:
        from scene_reconstruction_v3 import generate_scene_video
        return generate_scene_video
    except ImportError as exc:
        log.warning("[run_case] Scene video generation unavailable: %s", exc)
        return None


def _import_scene_planner():
    """Lazily import the scene planner, which builds the render-safe scene
    spec (physical participants vs. report-only evidence sources,
    already separated) that generate_scene_video() renders."""
    try:
        from scene_planner import build_scene_specs, DEFAULT_DB_PATH
        return build_scene_specs, DEFAULT_DB_PATH
    except ImportError as exc:
        log.warning("[run_case] Scene planner unavailable: %s", exc)
        return None, None


def _import_report():
    """Lazily import the explainability report generator; optional dep (fpdf2)."""
    try:
        from explainability_report_v2 import generate_explainability_report
        return generate_explainability_report
    except ImportError as exc:
        log.warning("[run_case] Explainability report generation unavailable: %s", exc)
        return None


def _import_evaluator():
    """Lazily import the ground-truth evaluator; lives alongside run_case.py."""
    try:
        from evaluate import evaluate_case
        return evaluate_case
    except ImportError as exc:
        log.warning("[run_case] Ground-truth evaluation unavailable: %s", exc)
        return None


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


# ── Post-loop outputs ─────────────────────────────────────────────────────────
# Run once the revision loop has stopped (no_action, human_review or the
# MAX_LOOPS ceiling). Each is optional and returns None when its dependency
# is missing, so a failed output never fails the case.

def run_scene_video(case_id, out):
    generate_scene_video = _import_video()
    build_scene_specs, scene_db_path = _import_scene_planner()
    if not generate_scene_video:
        return None
    if not build_scene_specs:
        log.warning("[steps] Scene planner unavailable — skipping video for %s", case_id)
        return None
    _banner(f"Scene Reconstruction Video — {case_id}")
    # build_scene_specs re-reads the timeline/critique already saved to
    # forensynth.db -- a fresh, not stale, read -- so that the video is
    # driven by the same physical-vs-reported classification used
    # everywhere else, instead of re-deriving it from the timeline result.
    scene_spec = build_scene_specs(case_id, scene_db_path)
    video_path = generate_scene_video(scene_spec, str(out / "videos"), case_id)
    print(f"  Video  → {video_path}" if video_path else "  Video generation FAILED — see log.")
    return video_path


def run_report(case_id, out, pipeline_outputs):
    generate_explainability_report = _import_report()
    if not generate_explainability_report:
        return None
    _banner(f"Explainability Report — {case_id}")
    report_path = generate_explainability_report(pipeline_outputs, str(out / "reports"), case_id)
    print(f"  Report → {report_path}" if report_path else "  Report generation FAILED — see log.")
    return report_path


def run_eval(case_id, out, tl_ver):
    # Ground-truth evaluation only applies to synthetic cases generated
    # with a matching cases/{case_id}.json ground truth file -- skip
    # quietly for any case that doesn't have one.
    ground_truth_path = Path("cases") / f"{case_id}.json"
    if not ground_truth_path.exists():
        return None
    evaluate_case = _import_evaluator()
    if not evaluate_case:
        return None
    _banner(f"Ground Truth Evaluation — {case_id}")
    # NOTE: the parameter is named timeline_v3_path for interface
    # compatibility, but we pass tl_ver -- the case's actual final
    # version -- since most cases converge at V1 or V2 and never reach V3.
    eval_result = evaluate_case(
        case_id=case_id,
        ground_truth_path=str(ground_truth_path),
        timeline_v3_path=str(out / "timelines" / f"{case_id}_timeline_{tl_ver}.json"),
        output_dir=str(out / "evaluation"),
    )
    log.info("Evaluation: F1=%s result=%s", eval_result.get("f1_score", "N/A"), eval_result.get("result"))
    print(f"  Eval   → F1={eval_result.get('f1_score', 'N/A')}  result={eval_result.get('result')}")
    return eval_result
