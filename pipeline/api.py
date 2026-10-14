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
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_root = Path(__file__).resolve().parent.parent

PYTHON    = sys.executable
RUN_CASE  = str(_root / "pipeline" / "run_case.py")
DB_PATH   = str(_root / "forensynth.db")
UPLOAD_DIR = _root / "output" / "_uploads"
AUDIO_AGENT = str(_root / "agents" / "field_agents" / "audio_agent.py")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
# Evidence the dashboard can upload for one case, by kind -> allowed extensions
EVIDENCE_KINDS = {
    "obs":   {".json"},                          # ready-made obs_only files
    "text":  {".txt", ".md", ".eml", ".json"},   # text_agent.py inputs
    "audio": AUDIO_EXTENSIONS,                   # audio_agent.py
    "video": {".mp4", ".mov", ".webm", ".mkv"},  # video agent (unverified)
}
_SAFE_ID       = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9 ._()-]{1,128}")

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


class AudioClip(BaseModel):
    filename: str
    start_time: str          # ISO 8601 wall-clock time the recording starts
    location: str
    source: Optional[str] = None


class ExtractAudioRequest(BaseModel):
    case_id: str
    clips: list[AudioClip]
    language: str = "en"


class RunCaseRequest(BaseModel):
    """One case from any mix of uploaded evidence (see /api/upload)."""
    case_id: str
    obs_files: list[str] = []
    text_files: list[str] = []
    audio: list[AudioClip] = []
    video_files: list[str] = []
    analyse_video: bool = False      # video agent is unverified: off unless asked
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


def _audio_dir(case_id: str) -> Path:
    if not _SAFE_ID.fullmatch(case_id or ""):
        raise HTTPException(status_code=400, detail="case_id may only contain letters, digits, '_' and '-'")
    return UPLOAD_DIR / "audio" / case_id


@app.post("/api/audio-upload")
async def audio_upload(case_id: str, filename: str, request: Request):
    """Store one raw audio file (request body = file bytes) for a case.
    Raw body rather than multipart so the API needs no extra dependency."""
    name = Path(filename).name
    if Path(name).suffix.lower() not in AUDIO_EXTENSIONS or not _SAFE_FILENAME.fullmatch(name):
        raise HTTPException(status_code=400, detail=f"Unsupported audio file name: {filename}")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    target_dir = _audio_dir(case_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / name).write_bytes(data)
    return {"filename": name, "bytes": len(data)}


@app.post("/api/extract-audio")
def extract_audio(req: ExtractAudioRequest):
    """Run the Audio Field Agent over a case's uploaded clips and return the
    resulting obs_only JSON - the same shape /api/run-pipeline takes as
    obs_data."""
    audio_dir = _audio_dir(req.case_id)
    if not req.clips:
        raise HTTPException(status_code=400, detail="No audio clips given")

    manifest = []
    for clip in req.clips:
        name = Path(clip.filename).name
        if not (audio_dir / name).is_file():
            raise HTTPException(status_code=400, detail=f"{name} has not been uploaded for {req.case_id}")
        if not clip.start_time.strip() or not clip.location.strip():
            raise HTTPException(status_code=400, detail=f"{name}: start time and location are required")
        manifest.append({
            "path": name,
            "start_time": clip.start_time.strip(),
            "location": clip.location.strip(),
            "source": (clip.source or "").strip() or None,
        })
    manifest_path = audio_dir / "clips.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cmd = [PYTHON, AUDIO_AGENT, "--manifest", str(manifest_path),
           "--case-id", req.case_id, "--output", str(audio_dir),
           "--language", req.language]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        cwd=str(_root), env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
    )
    out_path = audio_dir / f"{req.case_id}_obs_only.json"
    if proc.returncode != 0 or not out_path.is_file():
        return JSONResponse(
            status_code=500,
            content={
                "error": f"audio_agent.py exited with code {proc.returncode}",
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            },
        )

    obs_data = json.loads(out_path.read_text(encoding="utf-8"))
    if not obs_data.get("observations"):
        return JSONResponse(
            status_code=422,
            content={"error": "No speech was transcribed from the uploaded audio."},
        )
    return obs_data


def _case_meta(case_id: str) -> Optional[dict]:
    """{case_id, template, fir} for a case already in the DB, or None."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
    conn.close()
    if not row:
        return None
    meta = dict(row)
    return {
        "case_id": case_id,
        "template": meta.get("template", "") or "",
        "fir": json.loads(meta["fir_json"]) if meta.get("fir_json") else {},
    }


@app.get("/api/case/{case_id}")
def get_case(case_id: str):
    """Re-fetch a previously-run case's result fresh from the DB, without
    re-running the pipeline."""
    meta = _case_meta(case_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
    return _build_result(case_id, meta, no_llm=False)


# ── Multi-evidence cases (obs_only + text + audio + video) ──────────────────

def _evidence_dir(case_id: str, kind: str) -> Path:
    if not _SAFE_ID.fullmatch(case_id or ""):
        raise HTTPException(status_code=400, detail="case_id may only contain letters, digits, '_' and '-'")
    if kind not in EVIDENCE_KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {sorted(EVIDENCE_KINDS)}")
    return UPLOAD_DIR / "cases" / case_id / kind


def _evidence_path(case_id: str, kind: str, filename: str) -> Path:
    name = Path(filename).name
    path = _evidence_dir(case_id, kind) / name
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"{kind} file {name} has not been uploaded for {case_id}")
    return path


@app.post("/api/upload")
async def upload_evidence(case_id: str, kind: str, filename: str, request: Request):
    """Store one evidence file for a case (request body = file bytes).
    kind: obs | text | audio | video."""
    target_dir = _evidence_dir(case_id, kind)
    name = Path(filename).name
    if Path(name).suffix.lower() not in EVIDENCE_KINDS[kind] or not _SAFE_FILENAME.fullmatch(name):
        raise HTTPException(status_code=400, detail=f"Unsupported {kind} file: {filename} "
                                                    f"(allowed: {', '.join(sorted(EVIDENCE_KINDS[kind]))})")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if kind == "obs":
        try:
            parsed = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HTTPException(status_code=400, detail=f"{name} is not valid JSON")
        if not isinstance(parsed, dict) or not isinstance(parsed.get("observations"), list):
            raise HTTPException(status_code=400, detail=f"{name} is not an obs_only file "
                                                        "(needs an 'observations' list)")
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / name).write_bytes(data)
    return {"kind": kind, "filename": name, "bytes": len(data)}


@app.post("/api/run-case")
def run_case(req: RunCaseRequest):
    """Run one case from any mix of uploaded evidence.

    A single obs_only file with nothing else goes straight to the analysis
    loop (run_case.py --input), unchanged - it keeps its own case_id, FIR
    block and obs_ids, so ground-truth evaluation still applies. Anything
    else goes through the graph's evidence intake (run_case.py --case):
    text -> text_agent, audio -> audio_agent, obs files merged in, all into
    one case under req.case_id. Video clips are stored either way but only
    analysed when analyse_video is set - the video agent is unverified."""
    if not _SAFE_ID.fullmatch(req.case_id or ""):
        raise HTTPException(status_code=400, detail="case_id may only contain letters, digits, '_' and '-'")
    obs_paths = [_evidence_path(req.case_id, "obs", f) for f in req.obs_files]
    text_paths = [_evidence_path(req.case_id, "text", f) for f in req.text_files]
    video_paths = [_evidence_path(req.case_id, "video", f) for f in req.video_files]
    for clip in req.audio:
        _evidence_path(req.case_id, "audio", clip.filename)
        if not clip.start_time.strip() or not clip.location.strip():
            raise HTTPException(status_code=400, detail=f"{clip.filename}: start time and location are required")

    run_video = bool(video_paths) and req.analyse_video
    warnings: list[str] = []
    if video_paths and not run_video:
        warnings.append(f"{len(video_paths)} video clip(s) stored but not analysed "
                        "(video agent is unverified - tick 'Analyse video' to run it).")
    if not (obs_paths or text_paths or req.audio or run_video):
        raise HTTPException(status_code=400, detail="No evidence to analyse: add an obs_only file, "
                                                    "text documents, audio recordings, or enable video analysis.")

    cmd = [PYTHON, RUN_CASE, "--output", str(_root / "output")]
    raw_evidence = bool(text_paths or req.audio or run_video)
    if len(obs_paths) == 1 and not raw_evidence:
        case_id = json.loads(obs_paths[0].read_text(encoding="utf-8-sig")).get("case_id") or req.case_id
        if case_id != req.case_id:
            warnings.append(f"Used the case ID inside {obs_paths[0].name} ({case_id}).")
        cmd += ["--input", str(obs_paths[0])]
    else:
        case_id = req.case_id
        cmd += ["--case", case_id]
        if obs_paths:
            cmd += ["--obs-files", *map(str, obs_paths)]
        if text_paths:
            cmd += ["--text-inputs", *map(str, text_paths)]
        if req.audio:
            audio_dir = _evidence_dir(case_id, "audio")
            manifest = [{
                "path": str((audio_dir / Path(c.filename).name).resolve()),
                "start_time": c.start_time.strip(),
                "location": c.location.strip(),
                "source": (c.source or "").strip() or None,
            } for c in req.audio]
            manifest_path = audio_dir / "clips.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            cmd += ["--audio-manifest", str(manifest_path)]
        if video_paths:
            cmd += ["--video", str(_evidence_dir(case_id, "video").resolve())]
            if run_video:
                cmd.append("--enable-video")

    if req.no_llm:
        cmd.append("--no-llm")
    if req.must_merge:
        cmd += ["--must-merge", json.dumps(req.must_merge)]
    if req.must_not_merge:
        cmd += ["--must-not-merge", json.dumps(req.must_not_merge)]

    _wipe_case(case_id)
    child_env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    if req.no_llm:
        child_env["GROQ_API_KEY"] = ""
        child_env["Timeline_Key"] = ""
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          cwd=str(_root), env=child_env)
    if proc.returncode != 0:
        return JSONResponse(status_code=500, content={
            "error": f"run_case.py exited with code {proc.returncode}",
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        })

    meta = _case_meta(case_id) or {"case_id": case_id, "template": "", "fir": {}}
    result = _build_result(case_id, meta, req.no_llm)
    result["warnings"] = warnings
    result["evidence"] = {
        "obs_files": [p.name for p in obs_paths],
        "text_files": [p.name for p in text_paths],
        "audio_files": [Path(c.filename).name for c in req.audio],
        "video_files": [p.name for p in video_paths],
        "video_analysed": run_video,
    }
    return result
