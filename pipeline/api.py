#!/usr/bin/env python3
"""
ForenSynth - Dashboard API
Thin FastAPI layer connecting forensynth_dashboard.html to the real
pipeline. Runs pipeline/run_case.py as a subprocess (same pattern
evaluate_all.py uses), then reads the fresh per-round rows back out of
forensynth.db and reshapes them into the JSON contract the dashboard's
JS (buildMockResult()'s old shape) already renders.

Usage:
    uvicorn pipeline.api:app --reload --port 8000
    then open http://localhost:8000/
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_root = Path(__file__).resolve().parent.parent

PYTHON    = sys.executable
RUN_CASE  = str(_root / "pipeline" / "run_case.py")
DB_PATH   = str(_root / "forensynth.db")
UPLOAD_DIR = _root / "output" / "_uploads"

TABLES_WITH_CASE_ID = [
    "cases", "observations", "er_runs", "er_canonical", "er_clusters",
    "er_constraints", "timeline_runs", "timeline_events", "timeline_edges",
    "critique_runs", "critique_issues", "showrunner_runs", "pipeline_runs",
]

app = FastAPI(title="ForenSynth Dashboard API")


# ── Static hosting ───────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(_root / "forensynth_dashboard.html"))


app.mount("/output", StaticFiles(directory=str(_root / "output")), name="output")


# ── Request/response models ──────────────────────────────────────────────────

class RunPipelineRequest(BaseModel):
    obs_data: dict[str, Any]
    no_llm: bool = False
    must_merge: Optional[list[list[str]]] = None
    must_not_merge: Optional[list[list[str]]] = None


# ── DB helpers ────────────────────────────────────────────────────────────────

def _wipe_case(case_id: str) -> None:
    """Delete any existing DB rows for this case_id so a dashboard re-run is
    genuinely from scratch, not silently resuming a previous run's state."""
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    for t in TABLES_WITH_CASE_ID:
        try:
            conn.execute(f"DELETE FROM {t} WHERE case_id=?", (case_id,))
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def _build_result(case_id: str, obs_data: dict, no_llm: bool) -> dict:
    """Read the real per-round rows for case_id out of forensynth.db and
    reshape them into the JSON contract forensynth_dashboard.html's render
    functions already expect (the shape buildMockResult() used to fake)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    er_row = conn.execute(
        "SELECT full_json FROM er_runs WHERE case_id=? ORDER BY id DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    tl_row = conn.execute(
        "SELECT full_json FROM timeline_runs WHERE case_id=? ORDER BY id DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    critique_rows = conn.execute(
        "SELECT full_json FROM critique_runs WHERE case_id=? ORDER BY id",
        (case_id,),
    ).fetchall()
    showrunner_rows = conn.execute(
        "SELECT full_json FROM showrunner_runs WHERE case_id=? ORDER BY id",
        (case_id,),
    ).fetchall()
    conn.close()

    if not tl_row or not showrunner_rows:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline ran but no timeline/showrunner rows were found "
                    f"in the database for {case_id}.",
        )

    er = json.loads(er_row["full_json"]) if er_row else {}
    tl = json.loads(tl_row["full_json"])
    critiques = [json.loads(r["full_json"]) for r in critique_rows]
    showrunners = [json.loads(r["full_json"]) for r in showrunner_rows]

    last_critique = critiques[-1] if critiques else {}
    last_showrunner = showrunners[-1]

    er_entities = []
    for ent in er.get("canonical_entities", []):
        roles = ent.get("roles") or []
        er_entities.append({**ent, "role": roles[0] if roles else "unknown"})

    return {
        "case_id":  case_id,
        "template": obs_data.get("template", ""),
        "fir":      obs_data.get("fir", {}),
        "timeline_events": tl.get("events", []),
        "causal_links":    tl.get("causal_links", []),
        "er_entities":      er_entities,
        "issues":   last_critique.get("gaps", []),
        "score":    last_critique.get("overall_score", 0),
        "verdict":  last_critique.get("verdict", "ACCEPT"),
        "action":   last_showrunner.get("action", "no_action"),
        "output_case": last_showrunner.get("output_case", "AMBIGUOUS"),
        "iter_log": last_showrunner.get("iter_log", []),
        "loops":    len(showrunners),
        "reasoning": last_showrunner.get("reasoning", ""),
        "llm_used": not no_llm,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/run-pipeline")
def run_pipeline(req: RunPipelineRequest):
    case_id = req.obs_data.get("case_id")
    if not case_id:
        raise HTTPException(status_code=400, detail="obs_data.case_id is required")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    input_path = UPLOAD_DIR / f"{case_id}_obs_only.json"
    input_path.write_text(json.dumps(req.obs_data, indent=2), encoding="utf-8")

    _wipe_case(case_id)

    cmd = [PYTHON, RUN_CASE, "--input", str(input_path), "--output", str(_root / "output")]
    if req.no_llm:
        cmd.append("--no-llm")
    if req.must_merge:
        cmd += ["--must-merge", json.dumps(req.must_merge)]
    if req.must_not_merge:
        cmd += ["--must-not-merge", json.dumps(req.must_not_merge)]

    child_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if req.no_llm:
        child_env["GROQ_API_KEY"] = ""
        child_env["Timeline_Key"] = ""

    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        cwd=str(_root), env=child_env,
    )

    if proc.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"run_case.py exited with code {proc.returncode}",
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            },
        )

    result = _build_result(case_id, req.obs_data, req.no_llm)
    return result


@app.get("/api/case/{case_id}")
def get_case(case_id: str):
    """Re-fetch a previously-run case's result fresh from the DB, without
    re-running the pipeline."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    case_meta = conn.execute(
        "SELECT * FROM cases WHERE case_id=?", (case_id,)
    ).fetchone()
    conn.close()
    if not case_meta:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    meta = dict(case_meta)
    obs_data = {
        "case_id": case_id,
        "template": meta.get("template", "") or "",
        "fir": json.loads(meta["fir_json"]) if meta.get("fir_json") else {},
    }
    return _build_result(case_id, obs_data, no_llm=False)
