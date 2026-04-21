"""Pipeline configuration for entity resolution."""

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from .scorer import ScoringWeights


@dataclass
class PipelineConfiguration:
    """Configuration for the resolution pipeline."""

    # Intake
    check_duplicates: bool = True

    # Normalization
    case_base_time: Optional[datetime] = None

    # Blocking
    temporal_window_sec: int = 300
    max_temporal_gap_sec: int = 3600
    max_pairs: int = 500          # Cap on candidate pairs (anti-blowup, see Bug 4)

    # Scoring
    scoring_weights: Optional[ScoringWeights] = None

    # Edge classification
    confirmed_threshold: float = 0.70
    candidate_threshold_low: float = 0.50
    candidate_threshold_high: float = 0.70

    def to_dict(self) -> dict:
        """Convert to dict."""
        return {
            "check_duplicates": self.check_duplicates,
            "case_base_time": self.case_base_time.isoformat() if self.case_base_time else None,
            "temporal_window_sec": self.temporal_window_sec,
            "max_temporal_gap_sec": self.max_temporal_gap_sec,
            "max_pairs": self.max_pairs,
            "confirmed_threshold": self.confirmed_threshold,
            "candidate_threshold_low": self.candidate_threshold_low,
            "candidate_threshold_high": self.candidate_threshold_high,
        }