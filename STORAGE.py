from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


class ForenSynthDB:
    """Minimal local SQLite storage used when the project is run without a full DB backend."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS er_results (
                    case_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS timeline_results (
                    case_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def load_observations_for_er(self, case_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT payload FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def save_case_input(self, obs_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            payload = {"obs_data": obs_data, "result": result}
            conn.execute(
                "INSERT OR REPLACE INTO cases(case_id, payload) VALUES(?, ?)",
                (obs_data.get("case_id", "UNKNOWN"), json.dumps(payload)),
            )
            conn.commit()

    def save_er_result(self, result: Dict[str, Any]) -> None:
        case_id = result.get("case_id", "UNKNOWN")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO er_results(case_id, payload) VALUES(?, ?)",
                (case_id, json.dumps(result)),
            )
            conn.commit()

    def load_case_for_timeline(self, case_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT payload FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if not row:
            return None
        payload = json.loads(row[0])
        if isinstance(payload, dict) and "obs_data" in payload and "result" in payload:
            obs_data = payload["obs_data"]
            er_result = payload["result"]
            return {
                "case_id": obs_data.get("case_id") or er_result.get("case_id", case_id),
                "obs_only": {"observations": obs_data.get("observations", [])},
                "entity_resolved": {
                    "canonical_entities": er_result.get("canonical_entities", []),
                    "clusters": er_result.get("clusters", []),
                    "conflicts_detected": er_result.get("conflicts_detected", 0),
                    "conflicts": er_result.get("conflicts", []),
                },
            }
        return None

    def save_timeline(self, result: Dict[str, Any]) -> None:
        case_id = result.get("case_id", "UNKNOWN")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO timeline_results(case_id, payload) VALUES(?, ?)",
                (case_id, json.dumps(result)),
            )
            conn.commit()
