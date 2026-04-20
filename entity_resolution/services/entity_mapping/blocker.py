"""Blocking service for entity resolution candidate pair generation.

Generates candidate mention pairs using multiple blocking signals:
- Temporal proximity
- Location similarity
- Role compatibility
- Modality compatibility
- Lexical similarity (alias/content)
- Source similarity

Strategy: High recall (generate most candidates) and let similarity scoring filter.
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import difflib

from ...schemas.observation import NormalizedObservation, Modality


# ============================================================================
# Data Structures
# ============================================================================


class BlockingSignal(str, Enum):
    """Blocking signals used in pair evaluation."""

    TEMPORAL = "temporal"
    LOCATION = "location"
    ROLE = "role"
    MODALITY = "modality"
    ALIAS_PATTERN = "alias_pattern"
    SOURCE = "source"
    CONTENT = "content"


@dataclass
class CandidatePair:
    """Candidate pair of observations for entity resolution."""

    obs_id_1: str
    obs_id_2: str
    priority: float  # [0.0, 1.0] - higher = more important
    signals: Dict[BlockingSignal, float]  # Signal contributions to priority
    rationale: List[str]  # Textual explanation
    hard_reject: bool = False  # If true, skip comparison

    def __hash__(self):
        """Hash for deduplication."""
        pair = tuple(sorted([self.obs_id_1, self.obs_id_2]))
        return hash(pair)

    def __eq__(self, other):
        """Equality for deduplication."""
        if not isinstance(other, CandidatePair):
            return False
        pair_self = tuple(sorted([self.obs_id_1, self.obs_id_2]))
        pair_other = tuple(sorted([other.obs_id_1, other.obs_id_2]))
        return pair_self == pair_other

    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        return {
            "obs_id_1": self.obs_id_1,
            "obs_id_2": self.obs_id_2,
            "priority": round(self.priority, 3),
            "signals": {sig.value: round(val, 3) for sig, val in self.signals.items()},
            "rationale": self.rationale,
            "hard_reject": self.hard_reject,
        }


@dataclass
class BlockingReport:
    """Report from blocking process."""

    total_candidate_pairs: int = 0
    pairs_after_hard_reject: int = 0
    pairs_with_high_priority: int = 0  # priority >= 0.7
    pairs_with_medium_priority: int = 0  # 0.4 <= priority < 0.7
    pairs_with_low_priority: int = 0  # priority < 0.4
    avg_priority: float = 0.0
    cardinality_reduction: float = 0.0  # (n choose 2 - candidates) / (n choose 2)

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_candidate_pairs": self.total_candidate_pairs,
            "pairs_after_hard_reject": self.pairs_after_hard_reject,
            "high_priority": self.pairs_with_high_priority,
            "medium_priority": self.pairs_with_medium_priority,
            "low_priority": self.pairs_with_low_priority,
            "avg_priority": round(self.avg_priority, 3),
            "cardinality_reduction": round(self.cardinality_reduction, 3),
        }


# ============================================================================
# Blocking Signal Implementations
# ============================================================================


class TemporalBlocker:
    """Temporal proximity blocking."""

    @staticmethod
    def compute_temporal_proximity(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation, window_sec: int = 300
    ) -> float:
        """
        Compute temporal proximity score [0.0, 1.0].

        Args:
            obs_1: First observation
            obs_2: Second observation
            window_sec: Temporal window (default 5 minutes)

        Returns:
            Score [0.0, 1.0] where 1.0 = perfect overlap, 0.0 = far apart
        """
        dt1 = obs_1.timestamp_dt
        dt2 = obs_2.timestamp_dt

        gap_sec = abs((dt2 - dt1).total_seconds())

        # Perfect overlap
        if gap_sec == 0:
            return 1.0

        # Within window
        if gap_sec <= window_sec:
            # Linear decay: window_sec gap → 0.3
            return max(0.3, 1.0 - (gap_sec / window_sec) * 0.7)

        # Outside window
        return 0.0

    @staticmethod
    def should_reject_temporal(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation, max_gap_sec: int = 3600
    ) -> bool:
        """
        Hard rejection: observations too far apart in time.

        Args:
            obs_1: First observation
            obs_2: Second observation
            max_gap_sec: Maximum allowed time gap (default 1 hour)

        Returns:
            True if should hard-reject
        """
        gap_sec = abs((obs_2.timestamp_dt - obs_1.timestamp_dt).total_seconds())
        return gap_sec > max_gap_sec


class LocationBlocker:
    """Location similarity blocking."""

    @staticmethod
    def compute_location_similarity(obs_1: NormalizedObservation, obs_2: NormalizedObservation) -> float:
        """
        Compute location similarity score [0.0, 1.0].

        Uses fuzzy string matching on location strings.

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            Score [0.0, 1.0]
        """
        loc_1 = obs_1.location.lower()
        loc_2 = obs_2.location.lower()

        # Exact match
        if loc_1 == loc_2:
            return 1.0

        # Substring match (e.g., "ATM booth" contains "ATM")
        if loc_1 in loc_2 or loc_2 in loc_1:
            return 0.8

        # Fuzzy match
        ratio = difflib.SequenceMatcher(None, loc_1, loc_2).ratio()
        return ratio if ratio >= 0.3 else 0.0

    @staticmethod
    def should_reject_location(obs_1: NormalizedObservation, obs_2: NormalizedObservation) -> bool:
        """
        Hard rejection: observations in incompatible locations.

        For now, never hard-reject (locations might be aliased).

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            True if should hard-reject
        """
        # No hard rejection on location (too risky with aliasing)
        return False


class RoleBlocker:
    """Role compatibility blocking."""

    @staticmethod
    def compute_role_compatibility(obs_1: NormalizedObservation, obs_2: NormalizedObservation) -> float:
        """
        Compute role compatibility score [0.0, 1.0].

        Same role = high score. Different compatible roles = moderate score.
        Incompatible roles = low score.

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            Score [0.0, 1.0]
        """
        role_1 = obs_1.role.lower()
        role_2 = obs_2.role.lower()

        # Same role
        if role_1 == role_2:
            return 1.0

        # Both people (suspect/witness could be same entity)
        if role_1 in ["suspect", "witness"] and role_2 in ["suspect", "witness"]:
            return 0.7

        # System role unlikely to merge with people
        if "system" in [role_1, role_2]:
            return 0.2

        return 0.5

    @staticmethod
    def should_reject_role(obs_1: NormalizedObservation, obs_2: NormalizedObservation) -> bool:
        """
        Hard rejection: observations with incompatible roles.

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            True if should hard-reject
        """
        # No hard rejection (roles could be misclassified)
        return False


class ModalityBlocker:
    """Modality compatibility blocking."""

    @staticmethod
    def compute_modality_compatibility(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> float:
        """
        Compute modality compatibility score [0.0, 1.0].

        Different modalities (video+audio) often same event = high score.
        Same modality could be same event or different people = moderate.

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            Score [0.0, 1.0]
        """
        mod_1 = obs_1.modality
        mod_2 = obs_2.modality

        # Same modality: could be same or different
        if mod_1 == mod_2:
            # Video + video from different cameras: 0.6
            if mod_1 == Modality.VIDEO:
                return 0.6 if obs_1.source != obs_2.source else 0.4
            # Audio + audio from different sources: 0.5
            if mod_1 == Modality.AUDIO:
                return 0.5
            # Text + text: 0.4
            return 0.4

        # Different modalities: very likely same event (high compatibility)
        # video+audio, audio+text, video+text all possible
        return 0.85

    @staticmethod
    def should_reject_modality(obs_1: NormalizedObservation, obs_2: NormalizedObservation) -> bool:
        """
        Hard rejection: observations with incompatible modalities.

        For now, never hard-reject (modalities often co-occur).

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            True if should hard-reject
        """
        return False


class AliasPatternBlocker:
    """Alias pattern similarity blocking."""

    @staticmethod
    def compute_alias_pattern_similarity(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> float:
        """
        Compute alias pattern similarity [0.0, 1.0].

        Higher score if aliases follow similar patterns (Person_*, Speaker_*, etc).

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            Score [0.0, 1.0]
        """
        alias_1 = obs_1.entity.lower()
        alias_2 = obs_2.entity.lower()

        # Extract prefix (before underscore or number)
        prefix_1 = alias_1.split("_")[0]
        prefix_2 = alias_2.split("_")[0]

        # Same prefix (e.g., both Person_*)
        if prefix_1 == prefix_2:
            return 0.9

        # Different prefixes (e.g., Person_* vs Speaker_*)
        # Can still refer to same entity
        return 0.3

    @staticmethod
    def should_reject_alias(obs_1: NormalizedObservation, obs_2: NormalizedObservation) -> bool:
        """
        Hard rejection: observations with identical aliases.

        Same alias from same source = likely duplicate or independent mention.
        Cannot resolve same alias to different entities.

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            True if should hard-reject
        """
        # Same alias AND same source: already same mention
        if (obs_1.entity == obs_2.entity) and (obs_1.source == obs_2.source):
            # Should already be deduped by intake, but check anyway
            return True

        return False


class SourceBlocker:
    """Source similarity blocking."""

    @staticmethod
    def compute_source_similarity(obs_1: NormalizedObservation, obs_2: NormalizedObservation) -> float:
        """
        Compute source similarity score [0.0, 1.0].

        Same source = can't be same mention (already duplicate checked).
        Different sources of same type (cameras) = moderate.
        Different source types = low.

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            Score [0.0, 1.0]
        """
        src_1 = obs_1.source.lower()
        src_2 = obs_2.source.lower()

        # Same source
        if src_1 == src_2:
            return 0.3  # Could be independent mentions from same device

        # Same source type (e.g., camera_1 vs camera_2)
        src_type_1 = src_1.split("_")[0]
        src_type_2 = src_2.split("_")[0]

        if src_type_1 == src_type_2:
            return 0.7  # Likely same event, different angles

        return 0.4  # Different source types, still possible

    @staticmethod
    def should_reject_source(obs_1: NormalizedObservation, obs_2: NormalizedObservation) -> bool:
        """
        Hard rejection based on source incompatibility.

        For now, never hard-reject (sources can be aliased).

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            True if should hard-reject
        """
        return False


class ContentBlocker:
    """Content lexical similarity blocking."""

    @staticmethod
    def compute_content_similarity(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation, min_length: int = 5
    ) -> float:
        """
        Compute content lexical similarity [0.0, 1.0].

        Uses normalized (lowercased, whitespace-cleaned) content.

        Args:
            obs_1: First observation
            obs_2: Second observation
            min_length: Minimum content length to consider

        Returns:
            Score [0.0, 1.0]
        """
        content_1 = obs_1.content_normalized
        content_2 = obs_2.content_normalized

        # Both too short to be meaningful
        if len(content_1) < min_length or len(content_2) < min_length:
            return 0.2  # Low signal, but not rejected

        # Fuzzy match
        ratio = difflib.SequenceMatcher(None, content_1, content_2).ratio()

        if ratio >= 0.8:
            return 1.0  # Very similar content

        if ratio >= 0.5:
            return 0.6  # Moderately similar

        if ratio >= 0.3:
            return 0.3  # Weakly similar

        return 0.0

    @staticmethod
    def should_reject_content(obs_1: NormalizedObservation, obs_2: NormalizedObservation) -> bool:
        """
        Hard rejection based on content dissimilarity.

        For now, never hard-reject (content can be noisy/paraphrased).

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            True if should hard-reject
        """
        return False


# ============================================================================
# Blocking Orchestrator
# ============================================================================


class Blocker:
    """Main blocking service for candidate pair generation."""

    def __init__(
        self,
        temporal_window_sec: int = 300,
        max_temporal_gap_sec: int = 3600,
        max_pairs: int = 500,
        signal_weights: Optional[Dict[BlockingSignal, float]] = None,
    ):
        """
        Initialize blocker.

        Args:
            temporal_window_sec: Temporal window for proximity scoring
            max_temporal_gap_sec: Hard reject if gap exceeds this
            max_pairs: Maximum number of candidate pairs to generate
            signal_weights: Weights for each signal (default: uniform)
        """
        self.temporal_window_sec = temporal_window_sec
        self.max_temporal_gap_sec = max_temporal_gap_sec
        self.max_pairs = max_pairs

        # Default weights (uniform)
        self.signal_weights = signal_weights or {
            BlockingSignal.TEMPORAL: 1.0,
            BlockingSignal.LOCATION: 1.0,
            BlockingSignal.ROLE: 1.0,
            BlockingSignal.MODALITY: 1.0,
            BlockingSignal.ALIAS_PATTERN: 0.5,
            BlockingSignal.SOURCE: 0.5,
            BlockingSignal.CONTENT: 1.0,
        }

    def generate_candidates(
        self, observations: List[NormalizedObservation]
    ) -> Tuple[List[CandidatePair], BlockingReport]:
        """
        Generate candidate mention pairs.

        Args:
            observations: Normalized observations

        Returns:
            Tuple of (candidate_pairs, report)
        """
        report = BlockingReport()

        if len(observations) < 2:
            return [], report

        # Build observation index for fast lookup
        obs_index = {obs.obs_id: obs for obs in observations}

        # Generate all pairs (O(n²))
        all_pairs = self._generate_all_pairs(observations)
        report.total_candidate_pairs = len(all_pairs)

        # Apply hard rejection rules
        candidates = []
        for obs_1, obs_2 in all_pairs:
            # Check hard rejections
            if self._should_hard_reject(obs_1, obs_2):
                continue

            candidates.append((obs_1, obs_2))

        report.pairs_after_hard_reject = len(candidates)

        # Score and prioritize
        candidate_pairs = []
        for obs_1, obs_2 in candidates:
            pair = self._score_pair(obs_1, obs_2)
            candidate_pairs.append(pair)

        # Sort by priority (descending)
        candidate_pairs.sort(key=lambda p: p.priority, reverse=True)

        # Cap pairs to prevent O(n²) blowup on large cases
        if len(candidate_pairs) > self.max_pairs:
            candidate_pairs = candidate_pairs[:self.max_pairs]

        # Compute statistics
        if candidate_pairs:
            priorities = [p.priority for p in candidate_pairs]
            report.avg_priority = sum(priorities) / len(priorities)

            report.pairs_with_high_priority = sum(1 for p in priorities if p >= 0.7)
            report.pairs_with_medium_priority = sum(1 for p in priorities if 0.4 <= p < 0.7)
            report.pairs_with_low_priority = sum(1 for p in priorities if p < 0.4)

        # Cardinality reduction
        if report.total_candidate_pairs > 0:
            report.cardinality_reduction = 1.0 - (
                len(candidate_pairs) / report.total_candidate_pairs
            )

        return candidate_pairs, report

    @staticmethod
    def _generate_all_pairs(observations: List[NormalizedObservation]) -> List[Tuple]:
        """Generate all observation pairs (O(n²))."""
        pairs = []
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                pairs.append((observations[i], observations[j]))
        return pairs

    def _should_hard_reject(
        self, obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> bool:
        """Check hard rejection rules."""
        # Check each signal's hard rejection
        if TemporalBlocker.should_reject_temporal(obs_1, obs_2, self.max_temporal_gap_sec):
            return True

        if LocationBlocker.should_reject_location(obs_1, obs_2):
            return True

        if RoleBlocker.should_reject_role(obs_1, obs_2):
            return True

        if ModalityBlocker.should_reject_modality(obs_1, obs_2):
            return True

        if AliasPatternBlocker.should_reject_alias(obs_1, obs_2):
            return True

        if SourceBlocker.should_reject_source(obs_1, obs_2):
            return True

        if ContentBlocker.should_reject_content(obs_1, obs_2):
            return True

        return False

    def _score_pair(
        self, obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> CandidatePair:
        """Compute priority score for a pair."""
        signals = {}
        rationale = []

        # Temporal
        temporal_score = TemporalBlocker.compute_temporal_proximity(
            obs_1, obs_2, self.temporal_window_sec
        )
        signals[BlockingSignal.TEMPORAL] = temporal_score
        if temporal_score >= 0.7:
            rationale.append("Strong temporal proximity")
        elif temporal_score >= 0.3:
            rationale.append("Moderate temporal proximity")

        # Location
        location_score = LocationBlocker.compute_location_similarity(obs_1, obs_2)
        signals[BlockingSignal.LOCATION] = location_score
        if location_score >= 0.8:
            rationale.append("Same/similar location")
        elif location_score >= 0.5:
            rationale.append("Related locations")

        # Role
        role_score = RoleBlocker.compute_role_compatibility(obs_1, obs_2)
        signals[BlockingSignal.ROLE] = role_score
        if role_score >= 0.9:
            rationale.append("Role match")
        elif role_score >= 0.5:
            rationale.append("Compatible roles")

        # Modality
        modality_score = ModalityBlocker.compute_modality_compatibility(obs_1, obs_2)
        signals[BlockingSignal.MODALITY] = modality_score
        if modality_score >= 0.8:
            rationale.append("Different modalities (high compatibility)")
        elif modality_score >= 0.5:
            rationale.append("Compatible modalities")

        # Alias pattern
        alias_score = AliasPatternBlocker.compute_alias_pattern_similarity(obs_1, obs_2)
        signals[BlockingSignal.ALIAS_PATTERN] = alias_score
        if alias_score >= 0.8:
            rationale.append("Similar alias patterns")

        # Source
        source_score = SourceBlocker.compute_source_similarity(obs_1, obs_2)
        signals[BlockingSignal.SOURCE] = source_score
        if source_score >= 0.7:
            rationale.append("Same source type")

        # Content
        content_score = ContentBlocker.compute_content_similarity(obs_1, obs_2)
        signals[BlockingSignal.CONTENT] = content_score
        if content_score >= 0.6:
            rationale.append("Similar content")
        elif content_score >= 0.2:
            rationale.append("Somewhat similar content")

        # Weighted priority
        weighted_sum = sum(
            signals[sig] * self.signal_weights.get(sig, 1.0) for sig in signals.keys()
        )
        total_weight = sum(self.signal_weights.values())
        priority = weighted_sum / total_weight if total_weight > 0 else 0.5

        return CandidatePair(
            obs_id_1=obs_1.obs_id,
            obs_id_2=obs_2.obs_id,
            priority=priority,
            signals=signals,
            rationale=rationale,
        )
