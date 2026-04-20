"""Intake validation service for entity resolution."""

import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import ValidationError

from ...schemas.observation import Observation, CaseInput


@dataclass
class IntakeError:
    """Single intake error."""

    obs_id: str
    error_type: str
    message: str


@dataclass
class IntakeReport:
    """Intake validation report."""

    total_observations: int = 0
    valid_observations: int = 0
    invalid_observations: int = 0
    duplicate_observations: int = 0
    errors: List[IntakeError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dict."""
        return {
            "total_observations": self.total_observations,
            "valid_observations": self.valid_observations,
            "invalid_observations": self.invalid_observations,
            "duplicate_observations": self.duplicate_observations,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [
                {
                    "obs_id": e.obs_id,
                    "error_type": e.error_type,
                    "message": e.message,
                }
                for e in self.errors
            ],
            "warnings": self.warnings,
        }


class IntakeValidator:
    """Validates observations during intake."""

    def __init__(self):
        """Initialize validator."""
        self._seen_hashes: Dict[str, str] = {}  # hash → obs_id
        self._case_base_time: Optional[datetime] = None

    def validate_case(self, case_input: Dict[str, Any]) -> Tuple[List[Observation], IntakeReport]:
        """
        Validate case input and return validated observations.

        Args:
            case_input: Raw dict input from client

        Returns:
            Tuple of (validated_observations, report)
        """
        report = IntakeReport()

        # Validate top-level schema
        try:
            case_data = CaseInput(**case_input)
        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(x) for x in error["loc"])
                report.errors.append(
                    IntakeError(
                        obs_id="[CASE]",
                        error_type="SCHEMA_ERROR",
                        message=f"{field_path}: {error['msg']}",
                    )
                )
            report.total_observations = 0
            return [], report

        report.total_observations = len(case_data.observations)
        valid_observations: List[Observation] = []

        # Process each observation
        for obs in case_data.observations:
            try:
                # Observation already validated by Pydantic during CaseInput parsing
                # Check for duplicates
                obs_hash = self._compute_observation_hash(obs)
                if obs_hash in self._seen_hashes:
                    report.duplicate_observations += 1
                    report.warnings.append(
                        f"Duplicate observation {obs.obs_id} "
                        f"(identical to {self._seen_hashes[obs_hash]}); keeping higher confidence"
                    )
                    # Keep observation with higher confidence
                    existing_idx = [
                        i for i, o in enumerate(valid_observations)
                        if o.obs_id == self._seen_hashes[obs_hash]
                    ]
                    if existing_idx and obs.confidence > valid_observations[existing_idx[0]].confidence:
                        valid_observations[existing_idx[0]] = obs
                        self._seen_hashes[obs_hash] = obs.obs_id
                    continue

                self._seen_hashes[obs_hash] = obs.obs_id

                # Additional validation checks
                if obs.confidence < 0.0 or obs.confidence > 1.0:
                    report.errors.append(
                        IntakeError(
                            obs_id=obs.obs_id,
                            error_type="CONFIDENCE_ERROR",
                            message=f"Confidence {obs.confidence} out of [0.0, 1.0]",
                        )
                    )
                    report.invalid_observations += 1
                    continue

                # Parse timestamp
                try:
                    ts = datetime.fromisoformat(obs.timestamp.replace("Z", "+00:00"))
                except (ValueError, TypeError) as e:
                    report.errors.append(
                        IntakeError(
                            obs_id=obs.obs_id,
                            error_type="TIMESTAMP_ERROR",
                            message=f"Invalid timestamp {obs.timestamp}: {str(e)}",
                        )
                    )
                    report.invalid_observations += 1
                    continue

                # Store base time for later alignment
                if self._case_base_time is None:
                    self._case_base_time = ts

                # Observation is valid
                valid_observations.append(obs)
                report.valid_observations += 1

            except Exception as e:
                report.errors.append(
                    IntakeError(
                        obs_id=getattr(obs, "obs_id", "[UNKNOWN]"),
                        error_type="UNEXPECTED_ERROR",
                        message=str(e),
                    )
                )
                report.invalid_observations += 1

        return valid_observations, report

    @staticmethod
    def _compute_observation_hash(obs: Observation) -> str:
        """
        Compute deterministic hash of observation content for dedup.

        Args:
            obs: Observation to hash

        Returns:
            Hash string (ignores obs_id and confidence to find semantic duplicates)
        """
        # Create hashable representation (exclude obs_id and confidence)
        content = (
            f"{obs.entity}|{obs.role}|{obs.modality}|{obs.source}|"
            f"{obs.location}|{obs.content}|{obs.timestamp}|{obs.time_offset}"
        )
        return hashlib.sha256(content.encode()).hexdigest()
