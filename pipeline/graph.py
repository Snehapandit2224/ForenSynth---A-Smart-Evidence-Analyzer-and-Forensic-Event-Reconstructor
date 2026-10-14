#!/usr/bin/env python3
"""
ForenSynth - graph.py
The revision loop as a LangGraph state machine.

    load_case -> [ER] -> Timeline -> Critique -> Showrunner --+-> finalize -> scene_video -> report -> eval
                  ^          ^                                |
                  |          +------- re_run_timeline --------+
                  +------------------ re_run_er --------------+
                                      human_review -> human_review -> finalize

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
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

_root = Path(__file__).parent.parent
for _p in [str(_root / "pipeline"), str(_root / "agents"), str(_root / "memory"), str(_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langgraph.graph import END, START, StateGraph  # noqa: E402

from steps import (  # noqa: E402
    MAX_LOOPS, _banner, run_critique, run_er, run_eval, run_report,
    run_scene_video, run_showrunner, run_timeline,
)


class ForenSynthState(TypedDict, total=False):
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


# ── Nodes ─────────────────────────────────────────────────────────────────────

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

    g.add_edge(START, "load_case")
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
    llm: bool = True,
    output: str = "./output",
    db: str = "./forensynth.db",
    initial_constraints: Optional[Dict[str, Any]] = None,
    skip_video: bool = False,
    skip_report: bool = False,
    skip_eval: bool = False,
) -> ForenSynthState:
    """Run one case through the graph and return its final state."""
    if not input_path and not case_id:
        raise ValueError("Give input_path or case_id")
    state: ForenSynthState = {
        "input_path": input_path, "case_id": case_id or "", "llm": llm,
        "output": output, "db": db, "initial_constraints": initial_constraints,
        "skip_video": skip_video, "skip_report": skip_report, "skip_eval": skip_eval,
    }
    # Each pass is 3-4 node steps; MAX_LOOPS passes plus intake and outputs
    # must fit under LangGraph's step limit.
    return build_graph().invoke(state, {"recursion_limit": 10 * MAX_LOOPS + 20})
