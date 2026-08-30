"""
ForenSynth – memory_store.py  (SQLite backend)
Identical interface to the PostgreSQL version — swap files, nothing else changes.

Single file database: forensynth.db in your project root.
Zero setup, zero install, works offline.

When ready for production / multi-user:
    Replace this file with the PostgreSQL version.
    Everything else stays the same.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional

import networkx as nx

log = logging.getLogger("forensynth.memory_store")

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    domain TEXT DEFAULT '', template TEXT DEFAULT '',
    crime_type TEXT DEFAULT '', location TEXT DEFAULT '',
    description TEXT DEFAULT '', fir_json TEXT DEFAULT '{}',
    pipeline_status TEXT DEFAULT 'new',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    obs_id TEXT NOT NULL, entity TEXT DEFAULT '', role TEXT DEFAULT '',
    modality TEXT DEFAULT '', source TEXT DEFAULT '', location TEXT DEFAULT '',
    content TEXT DEFAULT '', timestamp_str TEXT DEFAULT '',
    time_offset INTEGER DEFAULT 0, confidence REAL DEFAULT 0.5,
    noise_tags TEXT DEFAULT '[]', raw_json TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(case_id, obs_id)
);
CREATE TABLE IF NOT EXISTS er_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    run_version INTEGER DEFAULT 1, status TEXT DEFAULT 'success',
    output_classification TEXT DEFAULT '', entity_count INTEGER DEFAULT 0,
    conflicts_detected INTEGER DEFAULT 0, llm_calls_made INTEGER DEFAULT 0,
    total_processing_time REAL DEFAULT 0.0, full_json TEXT NOT NULL,
    ran_at TEXT DEFAULT (datetime('now')),
    UNIQUE(case_id, run_version)
);
CREATE TABLE IF NOT EXISTS er_canonical (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    er_run_version INTEGER DEFAULT 1, entity_id TEXT NOT NULL,
    primary_alias TEXT DEFAULT '', aliases TEXT DEFAULT '[]',
    confidence_score REAL DEFAULT 0.0, sources TEXT DEFAULT '[]',
    modalities TEXT DEFAULT '[]', locations TEXT DEFAULT '[]',
    roles TEXT DEFAULT '[]', earliest_ts TEXT DEFAULT '',
    latest_ts TEXT DEFAULT '', time_span_sec INTEGER DEFAULT 0,
    raw_json TEXT NOT NULL,
    UNIQUE(case_id, er_run_version, entity_id)
);
CREATE TABLE IF NOT EXISTS er_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    er_run_version INTEGER DEFAULT 1, cluster_id TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    UNIQUE(case_id, er_run_version, cluster_id)
);
CREATE TABLE IF NOT EXISTS er_constraints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    constraint_version INTEGER DEFAULT 1,
    must_not_merge TEXT DEFAULT '[]', must_merge TEXT DEFAULT '[]',
    soft_hints TEXT DEFAULT '{}', reason TEXT DEFAULT '',
    injected_by TEXT DEFAULT 'showrunner',
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(case_id, constraint_version)
);
CREATE TABLE IF NOT EXISTS timeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    version TEXT NOT NULL DEFAULT 'V1', schema_version TEXT DEFAULT '',
    output_classification TEXT DEFAULT '', event_count INTEGER DEFAULT 0,
    causal_count INTEGER DEFAULT 0, conflict_count INTEGER DEFAULT 0,
    llm_calls_made INTEGER DEFAULT 0, total_time_sec REAL DEFAULT 0.0,
    full_json TEXT NOT NULL, graph_json TEXT DEFAULT '{}',
    generated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(case_id, version)
);
CREATE TABLE IF NOT EXISTS timeline_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    tl_version TEXT NOT NULL DEFAULT 'V1', event_id TEXT NOT NULL,
    obs_ids TEXT DEFAULT '[]', timestamp_str TEXT DEFAULT '',
    ts_epoch REAL DEFAULT 0.0, location TEXT DEFAULT '',
    entity_id TEXT DEFAULT '', primary_alias TEXT DEFAULT '',
    modality TEXT DEFAULT '', role TEXT DEFAULT '', content TEXT DEFAULT '',
    action_tags TEXT DEFAULT '[]', confidence REAL DEFAULT 0.0,
    conflict_flag INTEGER DEFAULT 0, reasoning TEXT DEFAULT '[]',
    UNIQUE(case_id, tl_version, event_id)
);
CREATE TABLE IF NOT EXISTS timeline_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    tl_version TEXT NOT NULL DEFAULT 'V1',
    source TEXT NOT NULL, target TEXT NOT NULL,
    edge_type TEXT DEFAULT '', relation TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0, label TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS critique_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    timeline_version TEXT NOT NULL DEFAULT 'V1',
    critique_version TEXT NOT NULL DEFAULT 'C1',
    overall_score REAL DEFAULT 0.0, total_issues INTEGER DEFAULT 0,
    critical_issues INTEGER DEFAULT 0, requires_revision INTEGER DEFAULT 0,
    revision_target TEXT DEFAULT '', full_json TEXT NOT NULL,
    generated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(case_id, timeline_version, critique_version)
);
CREATE TABLE IF NOT EXISTS critique_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    critique_version TEXT NOT NULL DEFAULT 'C1',
    event_id TEXT DEFAULT '', issue_type TEXT DEFAULT '',
    severity TEXT DEFAULT '', description TEXT DEFAULT '',
    recommendation TEXT DEFAULT '', showrunner_action TEXT DEFAULT '',
    must_not_merge TEXT DEFAULT '[]', affected_obs TEXT DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS showrunner_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(case_id),
    run_number INTEGER DEFAULT 1, input_tl_version TEXT DEFAULT 'V1',
    input_crit_version TEXT DEFAULT 'C1', output_tl_version TEXT DEFAULT 'V2',
    action_taken TEXT DEFAULT '', er_constraints_ver INTEGER DEFAULT NULL,
    status TEXT DEFAULT 'pending', full_json TEXT NOT NULL,
    ran_at TEXT DEFAULT (datetime('now')),
    UNIQUE(case_id, run_number)
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL, agent TEXT NOT NULL, action TEXT NOT NULL,
    version TEXT DEFAULT '', duration_sec REAL DEFAULT 0.0,
    status TEXT DEFAULT 'success', error_msg TEXT DEFAULT '',
    meta_json TEXT DEFAULT '{}', ran_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_obs_case      ON observations(case_id);
CREATE INDEX IF NOT EXISTS idx_er_run_case   ON er_runs(case_id);
CREATE INDEX IF NOT EXISTS idx_er_canon_case ON er_canonical(case_id, er_run_version);
CREATE INDEX IF NOT EXISTS idx_tl_run_case   ON timeline_runs(case_id);
CREATE INDEX IF NOT EXISTS idx_tl_ev_case    ON timeline_events(case_id, tl_version);
CREATE INDEX IF NOT EXISTS idx_tl_edge_case  ON timeline_edges(case_id, tl_version);
CREATE INDEX IF NOT EXISTS idx_crit_case     ON critique_runs(case_id);
CREATE INDEX IF NOT EXISTS idx_show_case     ON showrunner_runs(case_id);
CREATE INDEX IF NOT EXISTS idx_pipe_case     ON pipeline_runs(case_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def _load(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


class ForenSynthMemory:
    """SQLite memory store — identical interface to PostgreSQL version."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_path = str(_Path(__file__).parent.parent / "forensynth.db")
        self.db_path = str(_Path(db_path).resolve())
        self._init_schema()
        log.info("ForenSynthMemory (SQLite) → %s", self.db_path)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._conn() as con:
            con.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> List[Dict]:
        with self._conn() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def _query_one(self, sql: str, params: tuple = ()) -> Optional[Dict]:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.executescript(_SCHEMA)

    # ── Generator ──────────────────────────────────────────────────────────────
    def save_case(self, case: Dict[str, Any]) -> None:
        case_id = case["case_id"]
        fir = case.get("fir", {})
        with self._conn() as con:
            con.execute("""
                INSERT INTO cases (case_id,domain,template,crime_type,location,
                    description,fir_json,pipeline_status)
                VALUES (?,?,?,?,?,?,?,'generated')
                ON CONFLICT(case_id) DO UPDATE SET
                    updated_at=datetime('now'),pipeline_status=excluded.pipeline_status
            """, (case_id, case.get("domain",""), case.get("template",""),
                  fir.get("crime_type",""), fir.get("location",""),
                  fir.get("description",""), _j(fir)))
            for obs in case.get("observations", []):
                clean = {k: v for k, v in obs.items() if k != "event_ref"}
                con.execute("""
                    INSERT OR IGNORE INTO observations
                        (case_id,obs_id,entity,role,modality,source,location,
                         content,timestamp_str,time_offset,confidence,noise_tags,raw_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (case_id, obs.get("obs_id",""), obs.get("entity",""),
                      obs.get("role",""), obs.get("modality",""), obs.get("source",""),
                      obs.get("location",""), obs.get("content",""),
                      obs.get("timestamp",""), obs.get("time_offset",0),
                      obs.get("confidence",0.5), _j(obs.get("noise_tags",[])), _j(clean)))
        self._log_pipeline(case_id, "generator", "save_case", "")
        log.info("[Generator] Saved %s (%d obs)", case_id, len(case.get("observations",[])))

    # ── Entity Resolution ──────────────────────────────────────────────────────
    def load_observations(self, case_id: str) -> Optional[Dict[str, Any]]:
        rows = self._query(
            "SELECT raw_json FROM observations WHERE case_id=? ORDER BY time_offset, obs_id",
            (case_id,))
        if not rows:
            return None
        case_row = self._query_one(
            "SELECT * FROM cases WHERE case_id=?", (case_id,))
        if not case_row:
            return None
        return {
            "case_id": case_id,
            "domain": case_row.get("domain",""),
            "template": case_row.get("template",""),
            "fir": _load(case_row.get("fir_json","{}")) or {},
            "observations": [_load(r["raw_json"]) for r in rows],
        }

    def save_er_result(self, er_result: Dict[str, Any], run_version: int = 1) -> None:
        case_id = er_result["case_id"]
        with self._conn() as con:
            con.execute("UPDATE cases SET pipeline_status='er_complete',updated_at=datetime('now') WHERE case_id=?", (case_id,))
            con.execute("""
                INSERT OR REPLACE INTO er_runs
                    (case_id,run_version,status,output_classification,entity_count,
                     conflicts_detected,llm_calls_made,total_processing_time,full_json)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (case_id, run_version, er_result.get("status","success"),
                  er_result.get("output_classification",""),
                  er_result.get("entity_count",0),
                  er_result.get("conflicts_detected",0),
                  er_result.get("llm_calls_made",0),
                  er_result.get("total_processing_time_sec",0.0), _j(er_result)))
            con.execute("DELETE FROM er_canonical WHERE case_id=? AND er_run_version=?", (case_id, run_version))
            for ent in er_result.get("canonical_entities", []):
                con.execute("""
                    INSERT OR REPLACE INTO er_canonical
                        (case_id,er_run_version,entity_id,primary_alias,aliases,
                         confidence_score,sources,modalities,locations,roles,
                         earliest_ts,latest_ts,time_span_sec,raw_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (case_id, run_version, ent.get("entity_id",""),
                      ent.get("primary_alias",""), _j(ent.get("aliases",[])),
                      ent.get("confidence_score",0.0), _j(ent.get("sources",[])),
                      _j(ent.get("modalities",[])), _j(ent.get("locations",[])),
                      _j(ent.get("roles",[])), ent.get("earliest_timestamp",""),
                      ent.get("latest_timestamp",""), ent.get("time_span_seconds",0), _j(ent)))
            con.execute("DELETE FROM er_clusters WHERE case_id=? AND er_run_version=?", (case_id, run_version))
            for cluster in er_result.get("clusters", []):
                cid = cluster.get("cluster_id","")
                if cid:
                    con.execute("INSERT OR REPLACE INTO er_clusters (case_id,er_run_version,cluster_id,raw_json) VALUES (?,?,?,?)",
                                (case_id, run_version, cid, _j(cluster)))
        self._log_pipeline(case_id, "entity_resolution", "save_er_result", f"v{run_version}")
        log.info("[ER] Saved v%d for %s (%d entities)", run_version, case_id, er_result.get("entity_count",0))

    # ── Timeline Agent ─────────────────────────────────────────────────────────
    def load_for_timeline(self, case_id: str, er_version: int = 1) -> Optional[Dict[str, Any]]:
        obs_rows = self._query(
            "SELECT raw_json FROM observations WHERE case_id=? ORDER BY time_offset, obs_id", (case_id,))
        if not obs_rows:
            return None
        er_row = self._query_one(
            "SELECT full_json FROM er_runs WHERE case_id=? AND run_version=?", (case_id, er_version))
        if not er_row:
            return None
        er = _load(er_row["full_json"])
        return {
            "case_id": case_id,
            "obs_only": {"observations": [_load(r["raw_json"]) for r in obs_rows]},
            "entity_resolved": {
                "canonical_entities": er.get("canonical_entities",[]),
                "clusters": er.get("clusters",[]),
                "conflicts_detected": er.get("conflicts_detected",0),
                "conflicts": er.get("conflicts",[]),
            },
        }

    def save_timeline(self, timeline: Dict[str, Any]) -> None:
        case_id = timeline["case_id"]
        version = timeline.get("timeline_version","V1")
        graph_data = timeline.get("timeline_graph",{})
        G = self._build_nx_graph(graph_data)
        with self._conn() as con:
            con.execute("UPDATE cases SET pipeline_status='timeline_complete',updated_at=datetime('now') WHERE case_id=?", (case_id,))
            con.execute("""
                INSERT OR REPLACE INTO timeline_runs
                    (case_id,version,schema_version,output_classification,event_count,
                     causal_count,conflict_count,llm_calls_made,total_time_sec,full_json,graph_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (case_id, version, timeline.get("schema_version",""),
                  timeline.get("output_classification",""),
                  len(timeline.get("events",[])), len(timeline.get("causal_links",[])),
                  len(timeline.get("conflicts_summary",[])),
                  timeline.get("llm_calls_made",0), timeline.get("total_time_sec",0.0),
                  _j(timeline), _j(graph_data)))
            con.execute("DELETE FROM timeline_events WHERE case_id=? AND tl_version=?", (case_id, version))
            for ev in timeline.get("events",[]):
                con.execute("""
                    INSERT OR IGNORE INTO timeline_events
                        (case_id,tl_version,event_id,obs_ids,timestamp_str,ts_epoch,
                         location,entity_id,primary_alias,modality,role,content,
                         action_tags,confidence,conflict_flag,reasoning)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (case_id, version, ev.get("event_id",""), _j(ev.get("obs_ids",[])),
                      ev.get("timestamp",""), ev.get("ts_epoch",0.0), ev.get("location",""),
                      ev.get("entity_id",""), ev.get("primary_alias",""), ev.get("modality",""),
                      ev.get("role",""), ev.get("content",""), _j(ev.get("action_tags",[])),
                      ev.get("confidence",0.0), int(bool(ev.get("conflict_flag",False))),
                      _j(ev.get("reasoning",[]))))
            con.execute("DELETE FROM timeline_edges WHERE case_id=? AND tl_version=?", (case_id, version))
            for edge in graph_data.get("edges",[]):
                con.execute("""
                    INSERT INTO timeline_edges
                        (case_id,tl_version,source,target,edge_type,relation,confidence,label)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (case_id, version, edge.get("source",""), edge.get("target",""),
                      edge.get("edge_type",""), edge.get("relation",""),
                      edge.get("confidence",0.0), edge.get("label","")))
        self._log_pipeline(case_id, "timeline_agent", "save_timeline", version)
        log.info("[Timeline] Saved %s v%s (%d events)", case_id, version, len(timeline.get("events",[])))
        return G

    # ── Critique Agent ─────────────────────────────────────────────────────────
    def load_for_critique(self, case_id: str, tl_version: str = "V1") -> Optional[Dict[str, Any]]:
        tl_row = self._query_one(
            "SELECT full_json,graph_json FROM timeline_runs WHERE case_id=? AND version=?",
            (case_id, tl_version))
        if not tl_row:
            return None
        obs_rows = self._query(
            "SELECT raw_json FROM observations WHERE case_id=? ORDER BY time_offset", (case_id,))
        er_row = self._query_one(
            "SELECT full_json FROM er_runs WHERE case_id=? ORDER BY run_version DESC LIMIT 1", (case_id,))
        graph_data = _load(tl_row["graph_json"]) or {}
        return {
            "timeline": _load(tl_row["full_json"]),
            "graph": self._build_nx_graph(graph_data),
            "observations": [_load(r["raw_json"]) for r in obs_rows],
            "er_result": _load(er_row["full_json"]) if er_row else {},
        }

    def save_critique(self, critique: Dict[str, Any]) -> None:
        case_id = critique["case_id"]
        tl_version = critique.get("timeline_version","V1")
        crit_version = critique.get("critique_version","C1")
        issues = critique.get("issues",[])
        critical = sum(1 for i in issues if i.get("severity","").lower() == "critical")
        with self._conn() as con:
            con.execute("UPDATE cases SET pipeline_status='critique_complete',updated_at=datetime('now') WHERE case_id=?", (case_id,))
            con.execute("""
                INSERT OR REPLACE INTO critique_runs
                    (case_id,timeline_version,critique_version,overall_score,
                     total_issues,critical_issues,requires_revision,revision_target,full_json)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (case_id, tl_version, crit_version, critique.get("overall_score",0.0),
                  len(issues), critical, int(bool(critique.get("requires_revision",False))),
                  critique.get("revision_target",""), _j(critique)))
            con.execute("DELETE FROM critique_issues WHERE case_id=? AND critique_version=?", (case_id, crit_version))
            for issue in issues:
                con.execute("""
                    INSERT INTO critique_issues
                        (case_id,critique_version,event_id,issue_type,severity,
                         description,recommendation,showrunner_action,must_not_merge,affected_obs)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (case_id, crit_version, issue.get("event_id",""), issue.get("issue_type",""),
                      issue.get("severity",""), issue.get("description",""),
                      issue.get("recommendation",""), issue.get("showrunner_action",""),
                      _j(issue.get("must_not_merge",[])), _j(issue.get("affected_obs",[]))))
        self._log_pipeline(case_id, "critique_agent", "save_critique", crit_version)
        log.info("[Critique] Saved %s v%s (%d issues)", case_id, crit_version, len(issues))

    # ── Showrunner Agent ───────────────────────────────────────────────────────
    def load_for_showrunner(
        self, case_id: str, tl_version: str = "V1", crit_version: str = "C1"
    ) -> Optional[Dict[str, Any]]:
        crit_row = self._query_one("""
            SELECT full_json FROM critique_runs
            WHERE case_id=? AND timeline_version=? AND critique_version=?
        """, (case_id, tl_version, crit_version))
        if not crit_row:
            return None
        tl_row = self._query_one(
            "SELECT full_json FROM timeline_runs WHERE case_id=? AND version=?", (case_id, tl_version))
        obs_rows = self._query(
            "SELECT raw_json FROM observations WHERE case_id=? ORDER BY time_offset", (case_id,))
        constraint_row = self._query_one(
            "SELECT * FROM er_constraints WHERE case_id=? ORDER BY constraint_version DESC LIMIT 1", (case_id,))
        prev = {}
        if constraint_row:
            prev = dict(constraint_row)
            for k in ("must_not_merge","must_merge","soft_hints"):
                if k in prev:
                    prev[k] = _load(prev[k])
        return {
            "critique": _load(crit_row["full_json"]),
            "timeline": _load(tl_row["full_json"]) if tl_row else {},
            "observations": [_load(r["raw_json"]) for r in obs_rows],
            "previous_constraints": prev,
        }

    def save_er_constraints(self, case_id: str, constraints: Dict[str, Any], reason: str = "") -> int:
        row = self._query_one(
            "SELECT MAX(constraint_version) AS max_ver FROM er_constraints WHERE case_id=?", (case_id,))
        next_ver = (row["max_ver"] or 0) + 1 if row and row["max_ver"] else 1
        self._exec("""
            INSERT INTO er_constraints (case_id,constraint_version,must_not_merge,must_merge,soft_hints,reason)
            VALUES (?,?,?,?,?,?)
        """, (case_id, next_ver, _j(constraints.get("must_not_merge",[])),
              _j(constraints.get("must_merge",[])), _j(constraints.get("soft_hints",{})), reason))
        log.info("[Showrunner] Saved ER constraints v%d for %s", next_ver, case_id)
        return next_ver

    def load_er_constraints(self, case_id: str, version: int = None) -> Optional[Dict]:
        if version:
            row = self._query_one(
                "SELECT * FROM er_constraints WHERE case_id=? AND constraint_version=?", (case_id, version))
        else:
            row = self._query_one(
                "SELECT * FROM er_constraints WHERE case_id=? ORDER BY constraint_version DESC LIMIT 1", (case_id,))
        if not row:
            return None
        d = dict(row)
        for k in ("must_not_merge","must_merge","soft_hints"):
            if k in d:
                d[k] = _load(d[k])
        return d

    def save_showrunner_run(self, run: Dict[str, Any]) -> None:
        case_id = run["case_id"]
        row = self._query_one(
            "SELECT MAX(run_number) AS max_run FROM showrunner_runs WHERE case_id=?", (case_id,))
        next_run = (row["max_run"] or 0) + 1 if row and row["max_run"] else 1
        with self._conn() as con:
            con.execute("UPDATE cases SET pipeline_status='showrunner_complete',updated_at=datetime('now') WHERE case_id=?", (case_id,))
            con.execute("""
                INSERT OR REPLACE INTO showrunner_runs
                    (case_id,run_number,input_tl_version,input_crit_version,
                     output_tl_version,action_taken,er_constraints_ver,status,full_json)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (case_id, next_run, run.get("input_tl_version","V1"),
                  run.get("input_crit_version","C1"), run.get("output_tl_version","V2"),
                  run.get("action_taken",""), run.get("er_constraints_ver"),
                  run.get("status","complete"), _j(run)))
        self._log_pipeline(case_id, "showrunner", "save_showrunner_run", f"run_{next_run}")
        log.info("[Showrunner] Run %d saved for %s: %s", next_run, case_id, run.get("action_taken",""))

    # ── NetworkX ───────────────────────────────────────────────────────────────
    def _build_nx_graph(self, graph_data: Dict[str, Any]) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in graph_data.get("nodes",[]):
            nid = node.get("id","")
            if nid:
                G.add_node(nid, **{k: v for k, v in node.items() if k != "id"})
        for edge in graph_data.get("edges",[]):
            src, tgt = edge.get("source",""), edge.get("target","")
            if src and tgt:
                G.add_edge(src, tgt, **{k: v for k, v in edge.items() if k not in ("source","target")})
        return G

    def load_graph(self, case_id: str, version: str = "V1") -> Optional[nx.DiGraph]:
        row = self._query_one(
            "SELECT graph_json FROM timeline_runs WHERE case_id=? AND version=?", (case_id, version))
        if not row or not row.get("graph_json"):
            return None
        return self._build_nx_graph(_load(row["graph_json"]) or {})

    # ── Shared helpers ─────────────────────────────────────────────────────────
    def list_cases(self) -> List[Dict]:
        return self._query("""
            SELECT c.case_id, c.domain, c.template, c.pipeline_status, c.created_at,
                COUNT(DISTINCT o.obs_id) AS obs_count,
                COUNT(DISTINCT er.entity_id) AS entity_count,
                COUNT(DISTINCT tr.version) AS tl_versions,
                COUNT(DISTINCT cr.id) AS critique_count
            FROM cases c
            LEFT JOIN observations  o  ON o.case_id  = c.case_id
            LEFT JOIN er_canonical  er ON er.case_id = c.case_id
            LEFT JOIN timeline_runs tr ON tr.case_id = c.case_id
            LEFT JOIN critique_runs cr ON cr.case_id = c.case_id
            GROUP BY c.case_id ORDER BY c.created_at DESC
        """)

    def get_pipeline_status(self, case_id: str) -> Dict:
        case = self._query_one("SELECT * FROM cases WHERE case_id=?", (case_id,))
        if not case:
            return {}
        return {
            "case":        dict(case),
            "er_runs":     self._query("SELECT run_version,status,entity_count,ran_at FROM er_runs WHERE case_id=? ORDER BY run_version", (case_id,)),
            "tl_runs":     self._query("SELECT version,output_classification,event_count,generated_at FROM timeline_runs WHERE case_id=? ORDER BY generated_at", (case_id,)),
            "critiques":   self._query("SELECT critique_version,overall_score,total_issues,critical_issues FROM critique_runs WHERE case_id=?", (case_id,)),
            "showrunner":  self._query("SELECT run_number,action_taken,status FROM showrunner_runs WHERE case_id=? ORDER BY run_number", (case_id,)),
            "constraints": self._query("SELECT constraint_version,reason,created_at FROM er_constraints WHERE case_id=? ORDER BY constraint_version", (case_id,)),
        }

    def get_events(self, case_id: str, version: str = "V1", role: str = None, min_confidence: float = 0.0) -> List[Dict]:
        sql = "SELECT * FROM timeline_events WHERE case_id=? AND tl_version=? AND confidence >= ?"
        params = [case_id, version, min_confidence]
        if role:
            sql += " AND role=?"
            params.append(role)
        sql += " ORDER BY ts_epoch"
        rows = self._query(sql, tuple(params))
        for r in rows:
            for k in ("obs_ids","action_tags","reasoning"):
                if k in r:
                    r[k] = _load(r[k])
        return rows

    def get_causal_edges(self, case_id: str, version: str = "V1") -> List[Dict]:
        return self._query("""
            SELECT * FROM timeline_edges WHERE case_id=? AND tl_version=? AND edge_type='CAUSAL'
            ORDER BY confidence DESC
        """, (case_id, version))

    def get_critique_issues(self, case_id: str, crit_version: str = "C1", severity: str = None) -> List[Dict]:
        sql = "SELECT * FROM critique_issues WHERE case_id=? AND critique_version=?"
        params = [case_id, crit_version]
        if severity:
            sql += " AND severity=?"
            params.append(severity)
        return self._query(sql, tuple(params))

    def print_status(self, case_id: str) -> None:
        status = self.get_pipeline_status(case_id)
        if not status:
            print(f"Case {case_id} not found."); return
        c = status["case"]
        print(f"\n{'='*55}")
        print(f"  {c['case_id']}  [{c['pipeline_status']}]")
        print(f"  {c.get('domain','')} | {c.get('template','')}")
        print(f"{'='*55}")
        for r in status["er_runs"]:
            print(f"  ER v{r['run_version']}: {r['status']} — {r['entity_count']} entities")
        for r in status["tl_runs"]:
            print(f"  TL {r['version']}: {r['output_classification']} — {r['event_count']} events")
        for r in status["critiques"]:
            print(f"  Critique {r['critique_version']}: score={r['overall_score']:.2f}  issues={r['total_issues']}  critical={r['critical_issues']}")
        for r in status["showrunner"]:
            print(f"  Showrunner run {r['run_number']}: {r['action_taken']} [{r['status']}]")
        print(f"{'='*55}\n")


    def get_latest_versions(self, case_id: str) -> Dict[str, Any]:
        """
        Returns the latest version of every pipeline stage for a case.
        Used by all agents to auto-detect what to build on top of.

        Returns:
            {
                "er_version":      int  (latest ER run version, 0 if none)
                "tl_version":      str  (latest timeline version, None if none)
                "crit_version":    str  (latest critique version, None if none)
                "next_tl_version": str  (what the next timeline version should be)
                "next_crit_version": str
            }
        """
        er_rows   = self._query(
            "SELECT run_version FROM er_runs WHERE case_id=? ORDER BY run_version DESC LIMIT 1",
            (case_id,))
        tl_rows   = self._query(
            "SELECT version FROM timeline_runs WHERE case_id=? ORDER BY generated_at DESC LIMIT 1",
            (case_id,))
        crit_rows = self._query(
            "SELECT critique_version FROM critique_runs WHERE case_id=? ORDER BY generated_at DESC LIMIT 1",
            (case_id,))

        er_ver   = er_rows[0]["run_version"] if er_rows else 0
        tl_ver   = tl_rows[0]["version"]     if tl_rows else None
        crit_ver = crit_rows[0]["critique_version"] if crit_rows else None

        # Compute next versions
        def next_tl(v):
            if not v: return "V1"
            try: return f"V{int(v.lstrip('V')) + 1}"
            except: return "V1"

        def next_crit(v):
            if not v: return "C1"
            try: return f"C{int(v.lstrip('C')) + 1}"
            except: return "C1"

        return {
            "er_version":        er_ver,
            "tl_version":        tl_ver,
            "crit_version":      crit_ver,
            "next_tl_version":   next_tl(tl_ver),
            "next_crit_version": next_crit(crit_ver),
        }

    def _log_pipeline(self, case_id: str, agent: str, action: str,
                      version: str = "", duration: float = 0.0, status: str = "success") -> None:
        try:
            self._exec("""
                INSERT INTO pipeline_runs (case_id,agent,action,version,duration_sec,status)
                VALUES (?,?,?,?,?,?)
            """, (case_id, agent, action, version, duration, status))
        except Exception as exc:
            log.debug("Pipeline log failed: %s", exc)