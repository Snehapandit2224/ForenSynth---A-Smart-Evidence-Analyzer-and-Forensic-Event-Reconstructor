#!/usr/bin/env python3
"""
ForenSynth - graph.py
Evidence intake and the revision loop as one LangGraph state machine.

    raw evidence:  [fir_intake] -> text_intake  ---+
                                -> audio_intake ---+-> merge_evidence -> load_case
                                -> video_intake ---+    (video: wired, never selected yet)
    obs_only file / case in DB: ---------------------------------------> load_case

    load_case -> [ER] -> Timeline -> Critique -> Showrunner --+-> finalize -> scene_video -> report -> eval
                  ^          ^                                |
                  |          +------- re_run_timeline --------+
                  +------------------ re_run_er --------------+
                                      human_review -> human_review -> finalize

Intake reuses the field-agent orchestrator (agents/field_agents/text agent/
orchestrator.py) for every agent call and for the merge, so the graph and
that CLI produce the same merged case. The FIR runs first because the text
agent dates relative statements ("Friday 14th") from the FIR's incident
date; the modality agents then run in parallel.

Division of labour:
  - The Showrunner DECIDES what happens next (no_action / re_run_er /
    re_run_timeline / human_review). Nothing in this file second-guesses it.
  - This file only CARRIES OUT that decision: routing, the MAX_LOOPS ceiling,
    and the state passed between stages.
  - The memory store stays the record of every versioned result; the graph
    state only holds what the next node needs.

Every stage is a call into steps.py, the same functions run_case.py's
original loop (--legacy) uses, so the two orchestrators can be compared
case by case.

Usage:
    python pipeline/run_case.py --input cases/CASE_ATM_001_obs_only.json      # graph (default)
    python pipeline/run_case.py --input cases/CASE_ATM_001_obs_only.json --legacy
    python pipeline/run_case.py --case CASE_FIR_001 --fir samples/fir/FIR_001_chain_snatching.txt \\
        --text-inputs samples/fir/FIR_001_witness_statement.txt samples/fir/FIR_001_chat.json
    python pipeline/graph.py --diagram docs/pipeline_graph.md               # draw the graph
"""
from __future__ import annotations

import importlib.util
import json
import logging
import operator
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

_root = Path(__file__).parent.parent
for _p in [str(_root / "pipeline"), str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langgraph.graph import END, START, StateGraph  # noqa: E402

from steps import (  # noqa: E402
    MAX_LOOPS, _banner, run_critique, run_er, run_eval, run_report,
    run_scene_video, run_showrunner, run_timeline,
)


log = logging.getLogger("forensynth.graph")


class ForenSynthState(TypedDict, total=False):
    # raw evidence for intake: {"fir": path, "text": [paths],
    # "audio_manifest": path, "video": path, "obs_files": [paths]}
    evidence: Dict[str, Any]
    staging: str
    enable_video: bool
    adopt_role_hints: bool
    fir_record: Optional[Dict[str, Any]]
    fir_summary: Optional[Dict[str, Any]]
    # obs_only files produced by the intake nodes; parallel nodes append
    intake_files: Annotated[List[str], operator.add]

    # run settings (set once by run_pipeline)
    input_path: Optional[str]
    case_id: str
    llm: bool
    output: str
    db: str
    skip_video: bool
    skip_report: bool
    skip_eval: bool
    initial_constraints: Optional[Dict[str, Any]]

    # case + loop position
    obs_data: Dict[str, Any]
    er_ver: int
    loop: int
    tl_ver: Optional[str]
    crit_ver: Optional[str]

    # stage results
    er_result: Dict[str, Any]
    tl_result: Dict[str, Any]
    critique: Dict[str, Any]
    prev_decision: Optional[Dict[str, Any]]
    critiques_by_round: Dict[str, Any]
    showrunner_by_round: Dict[str, Any]
    actions: List[str]

    # outcome
    stop_reason: str
    outputs: Dict[str, Any]


_MEMORY: Dict[str, Any] = {}


def _mem(state: ForenSynthState):
    """One ForenSynthMemory per DB path, shared by every node of a run.
    Kept out of the state: it is a live connection, not data."""
    db = state["db"]
    if db not in _MEMORY:
        from memory_store import ForenSynthMemory
        _MEMORY[db] = ForenSynthMemory(db)
    return _MEMORY[db]


def _out(state: ForenSynthState) -> Path:
    return Path(state["output"])


# ── Intake nodes ──────────────────────────────────────────────────────────────

_FIELD_AGENTS_DIR = _root / "agents" / "field_agents"
_VIDEO_AGENT_DIR = _FIELD_AGENTS_DIR / "video_agent"
_orchestrator = None


_orchestrator_lock = threading.Lock()


def _orch():
    """The field-agent orchestrator module, loaded by path: its folder name
    ('text agent') has a space, so it cannot be imported as a package.

    The modality nodes run in parallel threads and can all ask for it at
    once, so loading is locked and the module is only published once fully
    executed - otherwise a second thread could get a half-loaded module
    (seen: text_intake failing with "no attribute 'run_text_agent'" while
    audio_intake was still loading it)."""
    global _orchestrator
    with _orchestrator_lock:
        if _orchestrator is None:
            path = _FIELD_AGENTS_DIR / "text agent" / "orchestrator.py"
            spec = importlib.util.spec_from_file_location("forensynth_field_orchestrator", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[spec.name] = module
            _orchestrator = module
    return _orchestrator


def _staging(state: ForenSynthState) -> Path:
    return Path(state["staging"])


def fir_intake_node(state: ForenSynthState) -> Dict[str, Any]:
    """FIR failure is fatal (the orchestrator raises): a case anchored on an
    FIR must not silently proceed without it."""
    rec_path, obs_path = _orch().run_fir_agent(state["case_id"], Path(state["evidence"]["fir"]),
                                               _staging(state))
    return {
        "fir_record": json.loads(rec_path.read_text(encoding="utf-8")),
        "fir_summary": json.loads(obs_path.read_text(encoding="utf-8"))["fir"],
        "intake_files": [str(obs_path)],
    }


def text_intake_node(state: ForenSynthState) -> Dict[str, Any]:
    fir = state.get("fir_summary") or {}
    out = _orch().run_text_agent(state["case_id"], [Path(p) for p in state["evidence"]["text"]],
                                 _staging(state),
                                 fir.get("incident_datetime") or fir.get("date_of_report"))
    return {"intake_files": [str(out)] if out else []}


def audio_intake_node(state: ForenSynthState) -> Dict[str, Any]:
    out = _orch().run_audio_agent(state["case_id"], Path(state["evidence"]["audio_manifest"]),
                                  _staging(state))
    return {"intake_files": [str(out)] if out else []}


def video_intake_node(state: ForenSynthState) -> Dict[str, Any]:
    """Wired in but never selected unless enable_video is set: the video
    agent is unverified (one matched observation in its own evaluation).

    Its observations assert `role`; ER treats differing roles as a hard
    "different people" rule, so an unverified model's guess could block a
    real cross-modal match. As with the text agent, the guess is kept as
    `role_hint` and `role` becomes "unknown"; every observation is marked
    extraction_status "unverified"."""
    case_id = state["case_id"]
    out_root = (_staging(state) / "video").resolve()
    cmd = [sys.executable, "-m", "video_agent.run", "--case-id", case_id,
           "--video", str(Path(state["evidence"]["video"]).resolve()), "--out", str(out_root)]
    rc = subprocess.run(cmd, cwd=str(_VIDEO_AGENT_DIR)).returncode
    src = out_root / case_id / f"{case_id}_video_obs.json"
    if rc != 0 or not src.exists():
        log.warning("video agent failed (exit %s) - continuing without video evidence.", rc)
        return {"intake_files": []}
    case = json.loads(src.read_text(encoding="utf-8"))
    for obs in case.get("observations", []):
        obs["role_hint"] = obs.get("role", "unknown")
        obs["role"] = "unknown"
        obs["extraction_status"] = "unverified"
    out = _staging(state) / f"{case_id}__video_obs_only.json"
    out.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"intake_files": [str(out)]}


def merge_evidence_node(state: ForenSynthState) -> Dict[str, Any]:
    """One case from every source: the orchestrator's merge (chronological
    interleave, single obs_id renumbering, ground-truth fields stripped),
    saved to the memory store with the FIR record. load_case then picks it
    up like any obs_only file."""
    orch = _orch()
    sources = [Path(p) for p in state["evidence"].get("obs_files", [])] + \
              [Path(p) for p in state.get("intake_files", [])]
    if not sources:
        raise RuntimeError("Every field agent failed - no evidence to merge.")
    merged = orch.merge_cases(state["case_id"], sources, state.get("fir_summary"))
    if state.get("adopt_role_hints"):
        log.info("Adopted %d role hint(s) into `role`.", orch.adopt_role_hints(merged["observations"]))
    # No agent may have run (premade obs files only), so the staging folder
    # might not exist yet.
    _staging(state).mkdir(parents=True, exist_ok=True)
    out = _staging(state) / f"{state['case_id']}_obs_only.json"
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    orch.save_to_memory(merged, db_path=state["db"], fir_record=state.get("fir_record"))
    print(f"  Intake : {len(sources)} source(s) -> {len(merged['observations'])} observations -> {out}")
    return {"input_path": str(out)}


# ── Analysis nodes ────────────────────────────────────────────────────────────

def load_case_node(state: ForenSynthState) -> Dict[str, Any]:
    """Register an obs_only file, or load a case already in the DB."""
    mem = _mem(state)
    if state.get("input_path"):
        input_path = Path(state["input_path"])
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        with open(input_path) as f:
            obs_data = json.load(f)
        mem.save_case(obs_data)
        case_id = obs_data["case_id"]
    else:
        case_id = state["case_id"]
        obs_data = mem.load_observations(case_id)
        if not obs_data:
            raise LookupError(f"Case {case_id} not in DB. Use --input first.")

    _banner(f"ForenSynth Pipeline — {case_id}")
    print(f"  LLM     : {'ENABLED' if state['llm'] else 'DISABLED'}")
    print(f"  Output  : {state['output']}")
    print(f"  Max loops: {MAX_LOOPS}")

    versions = mem.get_latest_versions(case_id)
    return {
        "case_id": case_id,
        "obs_data": obs_data,
        "er_ver": versions["er_version"] or 0,
        "loop": 0,
        "critiques_by_round": {},
        "showrunner_by_round": {},
        "actions": [],
        "prev_decision": None,
    }


def er_node(state: ForenSynthState) -> Dict[str, Any]:
    """First ER run uses the CLI's constraints; a re-run uses the ones the
    Showrunner just saved to the memory store."""
    mem = _mem(state)
    er_ver = state["er_ver"] + 1
    if er_ver == 1:
        constraints = state.get("initial_constraints")
    else:
        c = mem.load_er_constraints(state["case_id"])
        constraints = {
            "must_not_merge": c.get("must_not_merge", []),
            "must_merge":     c.get("must_merge", []),
            "soft_hints":     c.get("soft_hints", {}),
        } if c else None
    result = run_er(mem, state["obs_data"], state["llm"], run_version=er_ver,
                    human_constraints=constraints, out_dir=_out(state) / "er")
    return {"er_ver": er_ver, "er_result": result}


def timeline_node(state: ForenSynthState) -> Dict[str, Any]:
    loop = state["loop"] + 1
    tl_ver = f"V{loop}"
    print(f"\n  Loop {loop}/{MAX_LOOPS}")
    result = run_timeline(_mem(state), state["case_id"], state["er_ver"], tl_ver,
                          out_dir=_out(state) / "timelines")
    return {"loop": loop, "tl_ver": tl_ver, "tl_result": result}


def critique_node(state: ForenSynthState) -> Dict[str, Any]:
    crit_ver = f"C{state['loop']}"
    critique = run_critique(_mem(state), state["case_id"], state["tl_ver"], crit_ver,
                            tl_result=state["tl_result"], out_dir=_out(state) / "critiques")
    return {
        "crit_ver": crit_ver,
        "critique": critique,
        "critiques_by_round": {**state["critiques_by_round"], f"critique_c{state['loop']}": critique},
    }


def showrunner_node(state: ForenSynthState) -> Dict[str, Any]:
    decision = run_showrunner(_mem(state), state["case_id"], state["tl_ver"], state["crit_ver"],
                              out_dir=_out(state) / "showrunner",
                              prev_decision=state.get("prev_decision"))
    return {
        "prev_decision": decision,
        "showrunner_by_round": {**state["showrunner_by_round"], f"showrunner_c{state['loop']}": decision},
        "actions": state["actions"] + [decision["action"]],
    }


def human_review_node(state: ForenSynthState) -> Dict[str, Any]:
    """Escalation. For now this ends the loop, as run_case.py always has;
    pausing here for an investigator's constraints comes next."""
    _banner(f"ESCALATED TO HUMAN REVIEW — {state['case_id']}")
    print(f"  Reason: {state['prev_decision']['reasoning']}")
    return {"stop_reason": "human_review"}


def finalize_node(state: ForenSynthState) -> Dict[str, Any]:
    """Report why the loop stopped."""
    case_id = state["case_id"]
    reason = state.get("stop_reason") or _loop_outcome(state)
    if reason == "no_action":
        _banner(f"PIPELINE COMPLETE — {case_id}")
        print(f"  Final timeline : {state['tl_ver']}")
        print(f"  Score          : {state['critique'].get('overall_score', 0):.2f}")
        print(f"  No revision needed.")
    elif reason == "max_loops":
        _banner(f"MAX REVISIONS REACHED — {case_id}")
        print(f"  Stopped at {state['tl_ver']}. Manual review recommended.")
    return {"stop_reason": reason, "outputs": {}}


def scene_video_node(state: ForenSynthState) -> Dict[str, Any]:
    if state.get("skip_video"):
        return {}
    path = run_scene_video(state["case_id"], _out(state))
    return {"outputs": {**state["outputs"], "video": path}}


def report_node(state: ForenSynthState) -> Dict[str, Any]:
    if state.get("skip_report"):
        return {}
    pipeline_outputs = {
        "er":          state.get("er_result") or {},
        "timeline_v3": state.get("tl_result") or {},
        **state["critiques_by_round"],
        **state["showrunner_by_round"],
    }
    path = run_report(state["case_id"], _out(state), pipeline_outputs)
    return {"outputs": {**state["outputs"], "report": path}}


def eval_node(state: ForenSynthState) -> Dict[str, Any]:
    outputs = dict(state["outputs"])
    if not state.get("skip_eval"):
        outputs["eval"] = run_eval(state["case_id"], _out(state), state["tl_ver"])
    _mem(state).print_status(state["case_id"])
    return {"outputs": outputs}


# ── Routers ───────────────────────────────────────────────────────────────────

def _loop_outcome(state: ForenSynthState) -> str:
    """What the Showrunner's latest action means for the loop: carry it out,
    unless this was the last allowed pass."""
    action = state["prev_decision"]["action"]
    if action in ("no_action", "human_review"):
        return action
    if state["loop"] >= MAX_LOOPS:
        return "max_loops"
    return "re_run_er" if action == "re_run_er" else "re_run_timeline"


def route_modalities(state: ForenSynthState) -> List[str]:
    """The modality agents that have evidence, run in parallel. Video only
    when explicitly enabled. Nothing to run (only premade obs files, or only
    an FIR) goes straight to the merge."""
    ev = state.get("evidence") or {}
    nodes = []
    if ev.get("text"):
        nodes.append("text_intake")
    if ev.get("audio_manifest"):
        nodes.append("audio_intake")
    if ev.get("video") and state.get("enable_video"):
        nodes.append("video_intake")
    return nodes or ["merge_evidence"]


def route_start(state: ForenSynthState):
    """Raw evidence goes through intake; an obs_only file or a case already
    in the DB goes straight to load_case."""
    if not state.get("evidence"):
        return "load_case"
    return "fir_intake" if state["evidence"].get("fir") else route_modalities(state)


def route_after_load(state: ForenSynthState) -> str:
    return "er" if state["er_ver"] == 0 else "timeline"


def route_after_showrunner(state: ForenSynthState) -> str:
    return {
        "re_run_er":       "er",
        "re_run_timeline": "timeline",
        "human_review":    "human_review",
        "no_action":       "finalize",
        "max_loops":       "finalize",
    }[_loop_outcome(state)]


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    g = StateGraph(ForenSynthState)
    g.add_node("fir_intake", fir_intake_node)
    g.add_node("text_intake", text_intake_node)
    g.add_node("audio_intake", audio_intake_node)
    g.add_node("video_intake", video_intake_node)
    g.add_node("merge_evidence", merge_evidence_node)
    g.add_node("load_case", load_case_node)
    g.add_node("er", er_node)
    g.add_node("timeline", timeline_node)
    g.add_node("critique", critique_node)
    g.add_node("showrunner", showrunner_node)
    g.add_node("human_review", human_review_node)
    g.add_node("finalize", finalize_node)
    g.add_node("scene_video", scene_video_node)
    g.add_node("report", report_node)
    g.add_node("eval", eval_node)

    modality_nodes = ["text_intake", "audio_intake", "video_intake", "merge_evidence"]
    g.add_conditional_edges(START, route_start, ["fir_intake", "load_case"] + modality_nodes)
    g.add_conditional_edges("fir_intake", route_modalities, modality_nodes)
    for node in ("text_intake", "audio_intake", "video_intake"):
        g.add_edge(node, "merge_evidence")
    g.add_edge("merge_evidence", "load_case")
    g.add_conditional_edges("load_case", route_after_load, ["er", "timeline"])
    g.add_edge("er", "timeline")
    g.add_edge("timeline", "critique")
    g.add_edge("critique", "showrunner")
    g.add_conditional_edges("showrunner", route_after_showrunner,
                            ["er", "timeline", "human_review", "finalize"])
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", "scene_video")
    g.add_edge("scene_video", "report")
    g.add_edge("report", "eval")
    g.add_edge("eval", END)
    return g.compile(checkpointer=checkpointer)


def run_pipeline(
    input_path: Optional[str] = None,
    case_id: Optional[str] = None,
    *,
    evidence: Optional[Dict[str, Any]] = None,
    enable_video: bool = False,
    adopt_role_hints: bool = False,
    llm: bool = True,
    output: str = "./output",
    db: str = "./forensynth.db",
    initial_constraints: Optional[Dict[str, Any]] = None,
    skip_video: bool = False,
    skip_report: bool = False,
    skip_eval: bool = False,
) -> ForenSynthState:
    """Run one case through the graph and return its final state.

    Give exactly one of: input_path (an obs_only file), case_id alone (a
    case already in the DB), or case_id + evidence (raw evidence for the
    field agents; keys fir, text, audio_manifest, video, obs_files)."""
    evidence = {k: v for k, v in (evidence or {}).items() if v}
    if evidence and not case_id:
        raise ValueError("Raw evidence needs a case_id for the new case")
    if not input_path and not case_id:
        raise ValueError("Give input_path or case_id")
    state: ForenSynthState = {
        "input_path": input_path, "case_id": case_id or "", "llm": llm,
        "output": output, "db": db, "initial_constraints": initial_constraints,
        "skip_video": skip_video, "skip_report": skip_report, "skip_eval": skip_eval,
        "evidence": evidence, "enable_video": enable_video, "adopt_role_hints": adopt_role_hints,
        "staging": str(Path(output) / "_intake" / (case_id or "case")),
    }
    # Each pass is 3-4 node steps; MAX_LOOPS passes plus intake and outputs
    # must fit under LangGraph's step limit.
    return build_graph().invoke(state, {"recursion_limit": 10 * MAX_LOOPS + 20})


def export_diagram(path: str = "docs/pipeline_graph.md") -> Path:
    """Write the graph as a Mermaid diagram (renders on GitHub and in most
    Markdown viewers). Drawn from the compiled graph itself, so it can't
    drift from the code."""
    mermaid = build_graph().get_graph().draw_mermaid()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".md":
        out.write_text(
            "# ForenSynth pipeline graph\n\n"
            "Generated by `python pipeline/graph.py --diagram " + path.replace("\\", "/") + "` "
            "from `pipeline/graph.py`. Dashed edges are conditional (chosen by a router); "
            "solid edges always run.\n\n```mermaid\n" + mermaid + "```\n",
            encoding="utf-8")
    else:
        out.write_text(mermaid, encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ForenSynth pipeline graph")
    ap.add_argument("--diagram", default="docs/pipeline_graph.md",
                    help="Where to write the Mermaid diagram (.md or .mmd)")
    print(f"Diagram -> {export_diagram(ap.parse_args().diagram)}")
