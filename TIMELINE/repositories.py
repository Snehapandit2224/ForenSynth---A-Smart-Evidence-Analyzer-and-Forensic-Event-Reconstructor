"""
ForenSynth – Timeline Agent
repositories.py: clean repository abstractions for JSON-backed storage.

Design contract: replace JSON implementations with PostgreSQL-backed ones
without touching any other Timeline Agent file.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import CanonicalEntity, RawObservation, TimelineVersion

log = logging.getLogger("forensynth.timeline.repositories")


# ── Abstract base classes ─────────────────────────────────────────────────────

class ObservationRepository(ABC):
    """Provides raw observations to the Timeline Agent."""

    @abstractmethod
    def get_all(self, case_id: str) -> List[RawObservation]:
        ...

    @abstractmethod
    def get_by_id(self, case_id: str, obs_id: str) -> Optional[RawObservation]:
        ...


class EntityRepository(ABC):
    """Provides canonical entity data to the Timeline Agent."""

    @abstractmethod
    def get_all(self, case_id: str) -> List[CanonicalEntity]:
        ...

    @abstractmethod
    def get_by_id(self, case_id: str, entity_id: str) -> Optional[CanonicalEntity]:
        ...

    @abstractmethod
    def get_by_obs_id(self, case_id: str, obs_id: str) -> Optional[CanonicalEntity]:
        ...


class TimelineRepository(ABC):
    """Persists and retrieves Timeline Agent output."""

    @abstractmethod
    def save(self, timeline: TimelineVersion) -> None:
        ...

    @abstractmethod
    def load(self, case_id: str, version: str) -> Optional[TimelineVersion]:
        ...


# ── JSON-backed implementations ───────────────────────────────────────────────

def _parse_obs(item: Dict[str, Any]) -> RawObservation:
    return RawObservation(
        obs_id=str(item["obs_id"]),
        entity=str(item.get("entity", "")),
        role=str(item.get("role", "unknown")),
        modality=str(item.get("modality", "unknown")),
        location=str(item.get("location", "")),
        content=str(item.get("content", "")),
        timestamp=str(item.get("timestamp", "")),
        confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
        entity_norm=str(item.get("entity_norm", "")),
        time_offset_sec=int(item.get("time_offset_sec", 0)),
        _ts_epoch=float(item.get("_ts_epoch", 0.0)),
    )


def _parse_entity(item: Dict[str, Any]) -> CanonicalEntity:
    return CanonicalEntity(
        entity_id=str(item["entity_id"]),
        primary_alias=str(item.get("primary_alias", item.get("aliases", ["unknown"])[0])),
        aliases=list(item.get("aliases", [])),
        confidence_score=float(item.get("confidence_score", 0.5)),
        sources=list(item.get("sources", [])),
        modalities=list(item.get("modalities", [])),
        locations=list(item.get("locations", [])),
        roles=list(item.get("roles", [])),
        earliest_timestamp=str(item.get("earliest_timestamp", "")),
        latest_timestamp=str(item.get("latest_timestamp", "")),
        time_span_seconds=int(item.get("time_span_seconds", 0)),
        candidate_mentions=list(item.get("candidate_mentions", [])),
    )


class JsonObservationRepository(ObservationRepository):
    """
    Reads observations from the in-memory payload passed at construction time.
    When migrating to PostgreSQL, replace this class; the interface stays the same.
    """

    def __init__(self, raw_observations: List[Dict[str, Any]]) -> None:
        self._obs: Dict[str, RawObservation] = {}
        for item in raw_observations:
            try:
                obs = _parse_obs(item)
                self._obs[obs.obs_id] = obs
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Skipping malformed observation %s: %s", item.get("obs_id"), exc)

    def get_all(self, case_id: str) -> List[RawObservation]:
        return list(self._obs.values())

    def get_by_id(self, case_id: str, obs_id: str) -> Optional[RawObservation]:
        return self._obs.get(obs_id)


class JsonEntityRepository(EntityRepository):
    """
    Reads canonical entities from the in-memory entity_resolved payload.
    """

    def __init__(self, canonical_entities: List[Dict[str, Any]]) -> None:
        self._entities: Dict[str, CanonicalEntity] = {}
        # obs_id → entity_id reverse index
        self._obs_index: Dict[str, str] = {}
        for item in canonical_entities:
            try:
                ent = _parse_entity(item)
                self._entities[ent.entity_id] = ent
                for obs_id in ent.sources:
                    self._obs_index[obs_id] = ent.entity_id
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Skipping malformed entity %s: %s", item.get("entity_id"), exc)

    def get_all(self, case_id: str) -> List[CanonicalEntity]:
        return list(self._entities.values())

    def get_by_id(self, case_id: str, entity_id: str) -> Optional[CanonicalEntity]:
        return self._entities.get(entity_id)

    def get_by_obs_id(self, case_id: str, obs_id: str) -> Optional[CanonicalEntity]:
        entity_id = self._obs_index.get(obs_id)
        if entity_id:
            return self._entities.get(entity_id)
        return None


class JsonTimelineRepository(TimelineRepository):
    """Persists TimelineVersion objects to JSON files."""

    def __init__(self, output_dir: str = ".") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str, version: str) -> Path:
        safe = case_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}_timeline_{version}.json"

    def save(self, timeline: TimelineVersion) -> None:
        path = self._path(timeline.case_id, timeline.version)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(timeline.to_dict(), fh, indent=2, ensure_ascii=False)
        log.info("Timeline saved → %s", path)

    def load(self, case_id: str, version: str) -> Optional[TimelineVersion]:
        path = self._path(case_id, version)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        log.info("Timeline loaded ← %s", path)
        # Lightweight reconstruction — used for reading, not full deserialisation
        return data  # type: ignore[return-value]  # callers handle dict form
