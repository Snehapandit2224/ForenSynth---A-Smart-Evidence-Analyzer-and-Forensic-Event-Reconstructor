"""Normalization service for entity resolution."""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from ...schemas.observation import Observation, Modality, NormalizedObservation
from ...schemas.entity import NormalizedAlias


@dataclass
class NormalizationReport:
    """Report from normalization process."""

    total_observations: int = 0
    successfully_normalized: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, any]:
        """Convert to dict."""
        return {
            "total_observations": self.total_observations,
            "successfully_normalized": self.successfully_normalized,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
        }


class AliasNormalizer:
    """Parses and normalizes alias patterns."""

    # Regex patterns for common alias formats
    _PERSON_PATTERN = re.compile(r"^[Pp]erson[_-]?(\d+)$")
    _GENERIC_PERSON_PATTERN = re.compile(r"^[Pp]erson[_-]?(.+)$")  # Handles Person_X, Person_A, etc.
    _SPEAKER_PATTERN = re.compile(r"^[Ss]peaker[_-]?([A-Za-z])$")
    _SMS_PATTERN = re.compile(r"^[Ss]ms[_-]?(\d+)$")
    _EMAIL_PATTERN = re.compile(r"^[Ee]mail[_-]?(\d+)$")
    _REPORT_PATTERN = re.compile(r"^[Rr]eport[_-]?(\d+)$")
    _LOG_PATTERN = re.compile(r"^[Ll]og[_-]?(\d+)$")

    # Modality hints by alias type
    _MODALITY_HINTS: Dict[str, Modality] = {
        "Person": Modality.VIDEO,  # Person typically from video
        "Speaker": Modality.AUDIO,  # Speaker from audio
        "sms": Modality.TEXT,  # SMS from text
        "email": Modality.TEXT,  # Email from text
        "report": Modality.TEXT,  # Report from text
        "log": Modality.TEXT,  # Log from text
    }

    @classmethod
    def normalize_alias(cls, alias: str) -> NormalizedAlias:
        """
        Parse and normalize an alias string.

        Args:
            alias: Raw alias string (e.g., Person_05, Speaker_A)

        Returns:
            NormalizedAlias with parsed components

        Raises:
            ValueError if alias cannot be parsed
        """
        if not alias or not isinstance(alias, str):
            raise ValueError(f"Invalid alias: {alias}")

        alias = alias.strip()

        # Try each pattern
        patterns = [
            (cls._PERSON_PATTERN, "Person"),
            (cls._SPEAKER_PATTERN, "Speaker"),
            (cls._SMS_PATTERN, "sms"),
            (cls._EMAIL_PATTERN, "email"),
            (cls._REPORT_PATTERN, "report"),
            (cls._LOG_PATTERN, "log"),
        ]

        for pattern, alias_type in patterns:
            match = pattern.match(alias)
            if match:
                alias_id = match.group(1)
                modality_hint = cls._MODALITY_HINTS.get(alias_type, Modality.TEXT)
                return NormalizedAlias(
                    original=alias,
                    alias_type=alias_type,
                    alias_id=alias_id,
                    modality_hint=modality_hint.value,
                    confidence=1.0,
                )

        # Try generic Person pattern (e.g., Person_X, Person_A) before falling back
        generic_match = cls._GENERIC_PERSON_PATTERN.match(alias)
        if generic_match:
            alias_id = generic_match.group(1).lower()  # Normalize suffix to lowercase
            modality_hint = cls._MODALITY_HINTS.get("Person", Modality.VIDEO)
            return NormalizedAlias(
                original=alias,
                alias_type="Person",
                alias_id=alias_id,
                modality_hint=modality_hint.value,
                confidence=0.9,  # High confidence for recognized pattern, non-standard suffix
            )

        # If no pattern matches, return best-effort parse with graceful fallback
        return NormalizedAlias(
            original=alias,
            alias_type="unknown",
            alias_id=alias.lower(),
            modality_hint=Modality.TEXT.value,
            confidence=0.5,  # Lower confidence for unknown patterns
            canonical=alias.strip(),
        )


class Normalizer:
    """Normalizes observations."""

    def __init__(self, case_base_time: Optional[datetime] = None):
        """
        Initialize normalizer.

        Args:
            case_base_time: Base time for case window (for alignment)
        """
        self.case_base_time = case_base_time

    def normalize_observations(
        self, observations: List[Observation]
    ) -> Tuple[List[NormalizedObservation], NormalizationReport]:
        """
        Normalize a list of observations.

        Args:
            observations: List of validated observations

        Returns:
            Tuple of (normalized_observations, report)
        """
        report = NormalizationReport(total_observations=len(observations))
        normalized: List[NormalizedObservation] = []

        for obs in observations:
            try:
                norm_obs = self._normalize_single(obs)
                normalized.append(norm_obs)
                report.successfully_normalized += 1
            except Exception as e:
                report.errors.append(f"Observation {obs.obs_id}: {str(e)}")

        return normalized, report

    def _normalize_single(self, obs: Observation) -> NormalizedObservation:
        """
        Normalize a single observation.

        Args:
            obs: Observation to normalize

        Returns:
            NormalizedObservation
        """
        # Parse timestamp
        timestamp_str = obs.timestamp
        timestamp_dt = datetime.fromisoformat(
            timestamp_str.replace("Z", "+00:00")
        )

        # Compute offset from case base time (if provided)
        if self.case_base_time:
            timestamp_offset_sec = int((timestamp_dt - self.case_base_time).total_seconds())
        else:
            # Use provided time_offset as fallback
            timestamp_offset_sec = obs.time_offset

        # Normalize content
        content_normalized = self._normalize_content(obs.content)

        return NormalizedObservation(
            # Original fields (preserved)
            obs_id=obs.obs_id,
            entity=obs.entity,
            role=obs.role,
            modality=obs.modality,
            source=obs.source,
            location=obs.location,
            content=obs.content,
            timestamp=obs.timestamp,
            time_offset=obs.time_offset,
            confidence=obs.confidence,
            noise_tags=obs.noise_tags,
            # Computed/normalized fields
            timestamp_dt=timestamp_dt,
            timestamp_offset_sec=timestamp_offset_sec,
            content_normalized=content_normalized,
        )

    @staticmethod
    def _normalize_content(content: str) -> str:
        """
        Normalize content: lowercase and clean whitespace.

        Args:
            content: Raw content string

        Returns:
            Normalized content string
        """
        # Lowercase
        content = content.lower()

        # Remove extra whitespace (multiple spaces → single space)
        content = re.sub(r"\s+", " ", content)

        # Strip leading/trailing whitespace
        content = content.strip()

        return content

    @staticmethod
    def extract_modality_hint(alias: str) -> Modality:
        """
        Extract modality hint from alias pattern.

        Args:
            alias: Alias string

        Returns:
            Inferred modality
        """
        try:
            normalized = AliasNormalizer.normalize_alias(alias)
            return Modality(normalized.modality_hint)
        except (ValueError, KeyError):
            return Modality.TEXT  # Default to TEXT

    @staticmethod
    def infer_modality_from_observation(obs: Observation) -> Optional[Modality]:
        """
        Infer or cross-check modality from observation fields.

        Args:
            obs: Observation

        Returns:
            Inferred modality or None if ambiguous
        """
        # Explicit modality field is most reliable
        return obs.modality

    @staticmethod
    def normalize_location(location: str) -> str:
        """
        Normalize location string (preserve but standardize).

        Args:
            location: Raw location string

        Returns:
            Normalized location
        """
        # Lowercase and strip
        location = location.lower().strip()

        # Remove excessive whitespace
        location = re.sub(r"\s+", " ", location)

        return location

    @staticmethod
    def normalize_source(source: str) -> str:
        """
        Normalize source string.

        Args:
            source: Raw source string

        Returns:
            Normalized source
        """
        return source.lower().strip()
