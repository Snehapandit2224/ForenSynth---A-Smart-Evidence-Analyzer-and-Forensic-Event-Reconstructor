"""Feature computation service for entity resolution candidate pairs.

Computes 6 feature types for each candidate pair:
1. temporal_score - time proximity between observations
2. location_score - geographic/location similarity
3. context_score - contextual/semantic similarity
4. interaction_score - role and modality interaction compatibility
5. lexical_score - text content similarity (using rapidfuzz)
6. modality_compatibility_score - modality combination strength

All scores normalized to [0.0, 1.0]. High score = strong likelihood of same entity.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from rapidfuzz import fuzz
from math import exp, log

from ...schemas.observation import NormalizedObservation, Modality
from .blocker import CandidatePair


# ============================================================================
# Data Structures
# ============================================================================


class FeatureType(str, Enum):
    """Feature types computed for candidate pairs."""

    TEMPORAL = "temporal"
    LOCATION = "location"
    CONTEXT = "context"
    INTERACTION = "interaction"
    LEXICAL = "lexical"
    MODALITY = "modality"


@dataclass
class FeatureVector:
    """Feature vector for a candidate pair."""

    obs_id_1: str
    obs_id_2: str
    pair_priority: float  # From blocker (context)
    
    # Core 6 features
    temporal_score: float  # [0.0, 1.0]
    location_score: float  # [0.0, 1.0]
    context_score: float  # [0.0, 1.0]
    interaction_score: float  # [0.0, 1.0]
    lexical_score: float  # [0.0, 1.0]
    modality_compatibility_score: float  # [0.0, 1.0]
    
    # Composite scores
    combined_score: float = field(init=False)  # Weighted average
    
    # Metadata
    temporal_gap_sec: int = 0  # Seconds between observations
    location_distance: str = ""  # Location relationship
    content_similarity: float = 0.0  # Direct text similarity
    modality_pair: str = ""  # e.g., "video-audio"
    rationale: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute combined score after initialization."""
        # Default weighting (can be tuned)
        weights = {
            FeatureType.TEMPORAL: 0.25,
            FeatureType.LOCATION: 0.20,
            FeatureType.CONTEXT: 0.20,
            FeatureType.INTERACTION: 0.10,
            FeatureType.LEXICAL: 0.20,
            FeatureType.MODALITY: 0.05,
        }
        
        scores = [
            self.temporal_score * weights[FeatureType.TEMPORAL],
            self.location_score * weights[FeatureType.LOCATION],
            self.context_score * weights[FeatureType.CONTEXT],
            self.interaction_score * weights[FeatureType.INTERACTION],
            self.lexical_score * weights[FeatureType.LEXICAL],
            self.modality_compatibility_score * weights[FeatureType.MODALITY],
        ]
        self.combined_score = sum(scores)

    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        return {
            "obs_id_1": self.obs_id_1,
            "obs_id_2": self.obs_id_2,
            "pair_priority": round(self.pair_priority, 3),
            "temporal_score": round(self.temporal_score, 3),
            "location_score": round(self.location_score, 3),
            "context_score": round(self.context_score, 3),
            "interaction_score": round(self.interaction_score, 3),
            "lexical_score": round(self.lexical_score, 3),
            "modality_compatibility_score": round(self.modality_compatibility_score, 3),
            "combined_score": round(self.combined_score, 3),
            "temporal_gap_sec": self.temporal_gap_sec,
            "location_distance": self.location_distance,
            "content_similarity": round(self.content_similarity, 3),
            "modality_pair": self.modality_pair,
            "rationale": self.rationale,
        }


@dataclass
class FeatureReport:
    """Report from feature computation."""

    total_pairs: int = 0
    avg_combined_score: float = 0.0
    high_confidence_pairs: int = 0  # combined_score >= 0.7
    medium_confidence_pairs: int = 0  # 0.4 <= combined_score < 0.7
    low_confidence_pairs: int = 0  # combined_score < 0.4
    feature_averages: Dict[str, float] = field(default_factory=dict)
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_pairs": self.total_pairs,
            "avg_combined_score": round(self.avg_combined_score, 3),
            "high_confidence": self.high_confidence_pairs,
            "medium_confidence": self.medium_confidence_pairs,
            "low_confidence": self.low_confidence_pairs,
            "feature_averages": {k: round(v, 3) for k, v in self.feature_averages.items()},
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ============================================================================
# Feature Computation Implementations
# ============================================================================


class TemporalFeatures:
    """Compute temporal features for observation pairs."""

    @staticmethod
    def compute_temporal_score(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> Tuple[float, int]:
        """
        Compute temporal proximity score [0.0, 1.0].

        Strategy:
        - Same observation (within 1 second) → 1.0
        - < 60 seconds apart → 0.9-1.0 (very close)
        - < 300 seconds (5 min) apart → 0.6-0.9 (close)
        - < 3600 seconds (1 hour) apart → 0.2-0.6 (moderate)
        - > 3600 seconds apart → 0.0-0.2 (distant)

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            Tuple of (temporal_score, gap_seconds)
        """
        dt1 = obs_1.timestamp_dt
        dt2 = obs_2.timestamp_dt

        gap_sec = abs((dt2 - dt1).total_seconds())

        # Exponential decay: closer in time = higher score
        if gap_sec <= 1:
            score = 1.0
        elif gap_sec <= 60:
            # Linear interpolation: 60s → 0.9
            score = 0.9 + 0.1 * (1.0 - gap_sec / 60.0)
        elif gap_sec <= 300:
            # 300s → 0.6
            score = 0.9 * exp(-gap_sec / 150.0)
        elif gap_sec <= 3600:
            # 3600s → 0.05
            score = 0.6 * exp(-gap_sec / 1800.0)
        else:
            score = 0.05 * exp(-gap_sec / 7200.0)

        # Ensure bounds
        score = max(0.0, min(1.0, score))
        return score, int(gap_sec)


class LocationFeatures:
    """Compute location-based features for observation pairs."""

    @staticmethod
    def compute_location_score(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> Tuple[float, str]:
        """
        Compute location similarity score [0.0, 1.0].

        Strategy:
        - Exact match → 1.0
        - Fuzzy match (>85%) → 0.8-0.95
        - Substring match → 0.6-0.7
        - No match → 0.0

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            Tuple of (location_score, distance_description)
        """
        loc1 = (obs_1.location or "").lower().strip()
        loc2 = (obs_2.location or "").lower().strip()

        if not loc1 or not loc2:
            return 0.1, "missing_location"  # Weak signal if location missing

        # Exact match
        if loc1 == loc2:
            return 1.0, "exact_match"

        # Fuzzy match using token_set_ratio (handles word reordering)
        fuzzy_score = fuzz.token_set_ratio(loc1, loc2) / 100.0

        if fuzzy_score >= 0.85:
            return fuzzy_score, "fuzzy_match"
        elif fuzzy_score >= 0.6:
            return fuzzy_score * 0.8, "partial_match"
        
        # Substring match (one contains the other)
        if loc1 in loc2 or loc2 in loc1:
            return 0.6, "substring_match"

        return 0.0, "no_match"


class ContextFeatures:
    """Compute contextual/semantic features for observation pairs."""

    @staticmethod
    def compute_context_score(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> float:
        """
        Compute contextual similarity score [0.0, 1.0].

        Strategy:
        - Combines location consistency with role context
        - Same location + same role → higher score
        - Location context clues (e.g., "ATM" appears in content) → boost

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            context_score
        """
        score = 0.0

        # Location consistency
        loc1 = (obs_1.location or "").lower()
        loc2 = (obs_2.location or "").lower()
        if loc1 and loc2:
            loc_overlap = len(set(loc1.split()) & set(loc2.split())) / max(
                len(set(loc1.split())), len(set(loc2.split())), 1
            )
            score += 0.4 * loc_overlap

        # Role consistency
        if obs_1.role and obs_2.role:
            role_match = obs_1.role == obs_2.role
            score += 0.3 if role_match else 0.0

        # Content context clues
        content1 = (obs_1.content_normalized or "").lower()
        content2 = (obs_2.content_normalized or "").lower()
        
        # Check if both contents reference similar objects/activities
        context_keywords = ["atm", "phone", "door", "window", "person", "suspect", "witness"]
        matching_keywords = sum(
            1 for kw in context_keywords 
            if (kw in content1 and kw in content2)
        )
        score += 0.3 * (matching_keywords / max(len(context_keywords), 1))

        return min(1.0, score)


class InteractionFeatures:
    """Compute interaction-based features (role and modality combinations)."""

    @staticmethod
    def compute_interaction_score(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> float:
        """
        Compute interaction compatibility score [0.0, 1.0].

        Strategy:
        - Same role + complementary modalities (video+audio) → high
        - Different roles (suspect+witness) → moderate
        - System observations interact with any role → low

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            interaction_score
        """
        score = 0.0

        # Role compatibility
        if obs_1.role == obs_2.role:
            score += 0.5  # Same role = strong interaction
        elif obs_1.role in ["suspect", "witness"] and obs_2.role in ["suspect", "witness"]:
            score += 0.3  # Different but both human roles
        else:
            score += 0.1  # One or both system

        # Modality complementarity
        mod1 = obs_1.modality
        mod2 = obs_2.modality

        if mod1 == mod2:
            # Same modality - can they represent same entity?
            # Same video source should be very high
            if mod1 == Modality.VIDEO and obs_1.source == obs_2.source:
                score += 0.4
            else:
                score += 0.2
        else:
            # Different modalities - higher interaction potential
            # Video + audio is excellent (can capture same event)
            if {mod1, mod2} == {Modality.VIDEO, Modality.AUDIO}:
                score += 0.4
            elif mod1 == Modality.TEXT or mod2 == Modality.TEXT:
                score += 0.2  # Text is more generic
            else:
                score += 0.3

        return min(1.0, score)


class LexicalFeatures:
    """Compute lexical/text-based features for observation pairs."""

    @staticmethod
    def compute_lexical_score(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> Tuple[float, float]:
        """
        Compute lexical similarity score [0.0, 1.0] using rapidfuzz.

        Strategy:
        - Uses token_set_ratio for robustness to word reordering
        - Normalized content (stripped, lowercase) for comparison
        - Short content (<5 words) gets lower weight to avoid spurious matches
        - Longer content (>10 words) gets higher weight for descriptive content

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            Tuple of (lexical_score, raw_similarity_percentage)
        """
        content1 = (obs_1.content_normalized or "").strip()
        content2 = (obs_2.content_normalized or "").strip()

        if not content1 or not content2:
            return 0.0, 0.0  # No content to compare

        # Use token_set_ratio for flexibility with word reordering
        raw_similarity = fuzz.token_set_ratio(content1, content2) / 100.0

        # Adjust for content length
        # Prevent spurious matches with very short content
        words1 = len(content1.split())
        words2 = len(content2.split())
        min_words = min(words1, words2)

        if min_words < 3:
            # Very short content - reduce confidence
            adjusted_score = raw_similarity * 0.6
        elif min_words < 5:
            # Short content - slight reduction
            adjusted_score = raw_similarity * 0.8
        else:
            # Medium/long content - full weight
            adjusted_score = raw_similarity

        return adjusted_score, raw_similarity


class ModalityFeatures:
    """Compute modality compatibility features."""

    @staticmethod
    def compute_modality_compatibility_score(
        obs_1: NormalizedObservation, obs_2: NormalizedObservation
    ) -> Tuple[float, str]:
        """
        Compute modality compatibility score [0.0, 1.0].

        Strategy:
        - Video + Audio (same event, different sensors) → 0.9 (excellent)
        - Video + Video (same source) → 0.85 (same entity, sequential)
        - Video + Video (different cameras) → 0.5 (may or may not be same)
        - Audio + Audio → 0.6 (could be same source/speaker)
        - Any + Text → 0.3 (low specificity of text)
        - Any + Any different → base 0.4

        Args:
            obs_1: First observation
            obs_2: Second observation

        Returns:
            Tuple of (modality_compatibility_score, modality_pair_description)
        """
        mod1 = obs_1.modality
        mod2 = obs_2.modality
        src1 = obs_1.source
        src2 = obs_2.source

        # Handle both string and enum values
        mod1_val = mod1.value if hasattr(mod1, 'value') else mod1
        mod2_val = mod2.value if hasattr(mod2, 'value') else mod2
        pair_str = f"{mod1_val}-{mod2_val}"

        # Convert to enum if needed for comparison
        if isinstance(mod1, str):
            mod1 = Modality(mod1)
        if isinstance(mod2, str):
            mod2 = Modality(mod2)

        if mod1 == mod2:
            if mod1 == Modality.VIDEO:
                if src1 == src2:
                    return 0.85, "video-video-same-source"  # Sequential frames
                else:
                    return 0.5, "video-video-diff-source"  # Different cameras
            elif mod1 == Modality.AUDIO:
                if src1 == src2:
                    return 0.7, "audio-audio-same-source"  # Same recording
                else:
                    return 0.4, "audio-audio-diff-source"  # Different mics
            else:  # TEXT
                if src1 == src2:
                    return 0.5, "text-text-same-source"
                else:
                    return 0.3, "text-text-diff-source"
        else:
            # Different modalities
            modal_pair = {mod1, mod2}
            if modal_pair == {Modality.VIDEO, Modality.AUDIO}:
                return 0.9, "video-audio"  # Complementary capture same event
            elif modal_pair == {Modality.VIDEO, Modality.TEXT}:
                return 0.4, "video-text"
            elif modal_pair == {Modality.AUDIO, Modality.TEXT}:
                return 0.3, "audio-text"
            else:
                return 0.2, pair_str


# ============================================================================
# Feature Orchestrator
# ============================================================================


class Features:
    """Orchestrates feature computation for candidate pairs."""

    def __init__(self):
        """Initialize features orchestrator."""
        self.temporal_computer = TemporalFeatures()
        self.location_computer = LocationFeatures()
        self.context_computer = ContextFeatures()
        self.interaction_computer = InteractionFeatures()
        self.lexical_computer = LexicalFeatures()
        self.modality_computer = ModalityFeatures()

    def compute_features(
        self, candidates: List[CandidatePair], observations: List[NormalizedObservation]
    ) -> Tuple[List[FeatureVector], FeatureReport]:
        """
        Compute feature vectors for all candidate pairs.

        Args:
            candidates: List of candidate pairs from blocker
            observations: List of normalized observations (indexed by obs_id)

        Returns:
            Tuple of (feature_vectors, report)
        """
        import time
        start_time = time.time()

        # Build observation index
        obs_dict = {obs.obs_id: obs for obs in observations}

        # Compute features for each pair
        feature_vectors: List[FeatureVector] = []
        scores_by_type: Dict[str, List[float]] = {
            ft.value: [] for ft in FeatureType
        }

        for candidate in candidates:
            obs_1 = obs_dict.get(candidate.obs_id_1)
            obs_2 = obs_dict.get(candidate.obs_id_2)

            if not obs_1 or not obs_2:
                continue  # Skip if observations not found

            # Compute individual features
            temporal_score, temporal_gap = self.temporal_computer.compute_temporal_score(
                obs_1, obs_2
            )
            location_score, location_dist = self.location_computer.compute_location_score(
                obs_1, obs_2
            )
            context_score = self.context_computer.compute_context_score(obs_1, obs_2)
            interaction_score = self.interaction_computer.compute_interaction_score(
                obs_1, obs_2
            )
            lexical_score, raw_content_sim = self.lexical_computer.compute_lexical_score(
                obs_1, obs_2
            )
            modality_score, modality_pair = self.modality_computer.compute_modality_compatibility_score(
                obs_1, obs_2
            )

            # Create rationale
            rationale = []
            if temporal_score >= 0.7:
                rationale.append(f"Strong temporal proximity (gap {temporal_gap}s)")
            if location_score >= 0.8:
                rationale.append(f"Exact/near exact location match ({location_dist})")
            if modality_score >= 0.85:
                rationale.append(f"Highly compatible modalities ({modality_pair})")
            if lexical_score >= 0.7:
                rationale.append(f"Strong content similarity ({raw_content_sim:.1%})")
            if interaction_score >= 0.7:
                rationale.append("Strong role/modality interaction")

            # Create feature vector
            fv = FeatureVector(
                obs_id_1=candidate.obs_id_1,
                obs_id_2=candidate.obs_id_2,
                pair_priority=candidate.priority,
                temporal_score=temporal_score,
                location_score=location_score,
                context_score=context_score,
                interaction_score=interaction_score,
                lexical_score=lexical_score,
                modality_compatibility_score=modality_score,
                temporal_gap_sec=temporal_gap,
                location_distance=location_dist,
                content_similarity=raw_content_sim,
                modality_pair=modality_pair,
                rationale=rationale,
            )

            feature_vectors.append(fv)

            # Track feature scores for averaging
            scores_by_type[FeatureType.TEMPORAL.value].append(temporal_score)
            scores_by_type[FeatureType.LOCATION.value].append(location_score)
            scores_by_type[FeatureType.CONTEXT.value].append(context_score)
            scores_by_type[FeatureType.INTERACTION.value].append(interaction_score)
            scores_by_type[FeatureType.LEXICAL.value].append(lexical_score)
            scores_by_type[FeatureType.MODALITY.value].append(modality_score)

        # Sort by combined score (descending)
        feature_vectors.sort(key=lambda fv: fv.combined_score, reverse=True)

        # Compute statistics
        elapsed = time.time() - start_time

        high_conf = sum(1 for fv in feature_vectors if fv.combined_score >= 0.7)
        med_conf = sum(1 for fv in feature_vectors if 0.4 <= fv.combined_score < 0.7)
        low_conf = sum(1 for fv in feature_vectors if fv.combined_score < 0.4)

        avg_combined = (
            sum(fv.combined_score for fv in feature_vectors) / len(feature_vectors)
            if feature_vectors
            else 0.0
        )

        feature_averages = {
            ft: (sum(scores) / len(scores) if scores else 0.0)
            for ft, scores in scores_by_type.items()
        }

        report = FeatureReport(
            total_pairs=len(feature_vectors),
            avg_combined_score=avg_combined,
            high_confidence_pairs=high_conf,
            medium_confidence_pairs=med_conf,
            low_confidence_pairs=low_conf,
            feature_averages=feature_averages,
            processing_time_sec=elapsed,
        )

        return feature_vectors, report
