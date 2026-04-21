"""Similarity scoring service for entity resolution.

Stage 5: Computes weighted similarity score from feature vectors.

Scoring Weights (aligned with Features.py):
- alias_identity: 0.30 (CRITICAL - same alias is strongest signal for forensic entity matching)
- temporal: 0.20 (time proximity)
- location: 0.10 (geographic match)
- context: 0.15 (semantic context)
- interaction: 0.08 (role/modality compatibility)
- lexical: 0.12 (text similarity)
- modality: 0.05 (modality type compatibility)

Total: 1.00

Output: Single similarity_score [0.0, 1.0] for downstream classification.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

from .features import FeatureVector
from ...schemas.observation import NormalizedObservation


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class ScoredPair:
    """Result of similarity scoring for a candidate pair."""

    obs_id_1: str
    obs_id_2: str
    feature_vector: FeatureVector  # Reference to original features
    similarity_score: float  # [0.0, 1.0] weighted sum
    score_components: Dict[str, float] = field(default_factory=dict)  # Per-feature contributions
    rationale: List[str] = field(default_factory=list)  # Explanation of score

    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        return {
            "obs_id_1": self.obs_id_1,
            "obs_id_2": self.obs_id_2,
            "similarity_score": round(self.similarity_score, 3),
            "score_components": {k: round(v, 3) for k, v in self.score_components.items()},
            "rationale": self.rationale,
        }


@dataclass
class ScoringReport:
    """Report from similarity scoring stage."""

    total_pairs: int = 0
    avg_similarity_score: float = 0.0
    high_score_pairs: int = 0  # >= 0.7
    medium_score_pairs: int = 0  # 0.4-0.7
    low_score_pairs: int = 0  # < 0.4
    score_distribution: Dict[str, int] = field(default_factory=dict)  # Histogram
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_pairs": self.total_pairs,
            "avg_similarity_score": round(self.avg_similarity_score, 3),
            "high_score": self.high_score_pairs,
            "medium_score": self.medium_score_pairs,
            "low_score": self.low_score_pairs,
            "score_distribution": self.score_distribution,
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ============================================================================
# Scoring Weights Configuration
# ============================================================================


class ScoringWeights:
    """Configurable scoring weights for similarity computation."""

    # Default weights (must sum to 1.0)
    # Aligned with Features.py weights for consistency
    DEFAULT_WEIGHTS = {
        "alias_identity": 0.30,  # CRITICAL: identity match (same alias) is strongest signal
        "temporal": 0.175,  # Time proximity
        "location": 0.14,  # Geographic match
        "context": 0.14,  # Semantic context
        "interaction": 0.07,  # Role/modality compatibility
        "lexical": 0.14,  # Text content similarity
        "modality": 0.035,  # Modality type compatibility
    }

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize scoring weights.

        Args:
            weights: Optional custom weights dict. If not provided, uses defaults.
                    Must have keys: alias_identity, temporal, location, context, interaction, lexical, modality
                    Must sum to approximately 1.0

        Raises:
            ValueError: If weights don't sum to 1.0 or missing required keys
        """
        if weights is None:
            self.weights = self.DEFAULT_WEIGHTS.copy()
        else:
            # Validate required keys (now includes alias_identity)
            required_keys = {"alias_identity", "temporal", "location", "context", "interaction", "lexical", "modality"}
            provided_keys = set(weights.keys())
            if provided_keys != required_keys:
                missing = required_keys - provided_keys
                extra = provided_keys - required_keys
                msg = "Invalid weights keys."
                if missing:
                    msg += f" Missing: {missing}."
                if extra:
                    msg += f" Extra: {extra}."
                raise ValueError(msg)

            # Validate sum approximately equals 1.0
            weight_sum = sum(weights.values())
            if not (0.99 <= weight_sum <= 1.01):
                raise ValueError(f"Weights must sum to 1.0, got {weight_sum:.3f}")

            self.weights = weights.copy()

    def get(self, feature_name: str) -> float:
        """Get weight for a feature."""
        return self.weights.get(feature_name, 0.0)

    def to_dict(self) -> Dict[str, float]:
        """Return weights as dict."""
        return self.weights.copy()


# ============================================================================
# Similarity Scoring Implementation
# ============================================================================


class Scorer:
    """Computes weighted similarity scores for candidate pairs."""

    def __init__(self, weights: ScoringWeights = None):
        """
        Initialize scorer with optional custom weights.

        Args:
            weights: Optional ScoringWeights. If not provided, uses defaults.
        """
        if weights is None:
            self.weights = ScoringWeights()
        else:
            self.weights = weights

    def compute_similarity_score(self, feature_vector: FeatureVector) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Compute weighted similarity score from feature vector.

        Strategy:
        - Apply configured weights to each feature (INCLUDING alias_identity)
        - Produce single composite score [0.0, 1.0]
        - Track individual contributions for explainability

        Args:
            feature_vector: FeatureVector from Stage 4

        Returns:
            Tuple of (similarity_score, score_components, rationale)
            - similarity_score: [0.0, 1.0] weighted average
            - score_components: Dict[feature_name, weighted_value]
            - rationale: List of explanation strings
        """
        # Extract features from feature vector
        alias_identity = feature_vector.alias_identity_score
        temporal = feature_vector.temporal_score
        location = feature_vector.location_score
        context = feature_vector.context_score
        interaction = feature_vector.interaction_score
        lexical = feature_vector.lexical_score
        modality = feature_vector.modality_compatibility_score

        # Get weights (now includes alias_identity)
        w_alias_identity = self.weights.get("alias_identity")
        w_temporal = self.weights.get("temporal")
        w_location = self.weights.get("location")
        w_context = self.weights.get("context")
        w_interaction = self.weights.get("interaction")
        w_lexical = self.weights.get("lexical")
        w_modality = self.weights.get("modality")

        # Compute weighted contributions (now includes alias_identity)
        components = {
            "alias_identity": alias_identity * w_alias_identity,
            "temporal": temporal * w_temporal,
            "location": location * w_location,
            "context": context * w_context,
            "interaction": interaction * w_interaction,
            "lexical": lexical * w_lexical,
            "modality": modality * w_modality,
        }

        # Compute total similarity score
        similarity_score = sum(components.values())

        if alias_identity == 1.0:
            similarity_score = max(similarity_score, 0.90)
            rationale_boost = "Exact alias match: forced confirm score"
        elif alias_identity >= 0.85:
            similarity_score = max(similarity_score, 0.82)
            rationale_boost = "Alias boost: co-event signal triggered"
        else:
            rationale_boost = ""

        # Build rationale
        rationale = []

        if rationale_boost:
            rationale.append(rationale_boost)
        
        # Alias identity is the strongest signal
        if alias_identity >= 0.99:
            rationale.append(f"Perfect alias match ({alias_identity:.2f})")
        elif alias_identity >= 0.9:
            rationale.append(f"Very strong alias match ({alias_identity:.2f})")
        elif alias_identity >= 0.5:
            rationale.append(f"Partial alias match ({alias_identity:.2f})")
        
        if temporal >= 0.8:
            rationale.append(f"Excellent temporal alignment ({temporal:.2f})")
        elif temporal >= 0.5:
            rationale.append(f"Good temporal alignment ({temporal:.2f})")

        if location >= 0.9:
            rationale.append(f"Perfect location match ({location:.2f})")
        elif location >= 0.7:
            rationale.append(f"Strong location similarity ({location:.2f})")

        if context >= 0.7:
            rationale.append(f"Strong contextual alignment ({context:.2f})")

        if interaction >= 0.9:
            rationale.append(f"Excellent role/modality alignment ({interaction:.2f})")
        elif interaction >= 0.7:
            rationale.append(f"Good role/modality alignment ({interaction:.2f})")

        if lexical >= 0.7:
            rationale.append(f"Strong lexical/content similarity ({lexical:.2f})")
        elif lexical >= 0.4:
            rationale.append(f"Moderate content similarity ({lexical:.2f})")

        if modality >= 0.85:
            rationale.append(f"Ideal modality combination ({modality:.2f})")

        # Add score threshold context
        if similarity_score >= 0.75:
            rationale.append("High confidence pairing")
        elif similarity_score >= 0.55:
            rationale.append("Moderate confidence pairing")

        return similarity_score, components, rationale

    def score_pairs(
        self,
        feature_vectors: List[FeatureVector],
        observations: Optional[List[NormalizedObservation]] = None,
    ) -> Tuple[List[ScoredPair], ScoringReport]:
        """
        Score all candidate pairs.

        Args:
            feature_vectors: List of FeatureVector from Stage 4

        Returns:
            Tuple of (scored_pairs, report)
            - scored_pairs: List of ScoredPair sorted by similarity_score descending
            - report: ScoringReport with statistics
        """
        import time
        start_time = time.time()

        scored_pairs: List[ScoredPair] = []
        scores: List[float] = []

        obs_lookup: Dict[str, NormalizedObservation] = {}
        earliest_obs_by_alias: Dict[str, str] = {}
        if observations:
            obs_lookup = {obs.obs_id: obs for obs in observations}

            def observation_sort_key(obs: NormalizedObservation) -> Tuple[datetime, int, str]:
                timestamp = obs.timestamp_dt if getattr(obs, "timestamp_dt", None) else datetime.max
                time_offset = getattr(obs, "time_offset", 0) or 0
                return timestamp, time_offset, obs.obs_id

            alias_groups: Dict[str, List[NormalizedObservation]] = {}
            for obs in observations:
                alias = obs.entity.strip().lower()
                alias_groups.setdefault(alias, []).append(obs)

            for alias, alias_observations in alias_groups.items():
                alias_observations.sort(key=observation_sort_key)
                earliest_obs_by_alias[alias] = alias_observations[0].obs_id

        preferred_boost_pairs: Dict[str, Tuple[int, int, str]] = {}
        preferred_pair_index: Dict[str, int] = {}
        candidate_boost_flags: List[bool] = []

        for index, fv in enumerate(feature_vectors):
            similarity_score, components, rationale = self.compute_similarity_score(fv)

            is_cross_alias_boost = fv.alias_identity_score >= 0.85 and fv.alias_identity_score < 0.99
            candidate_boost_flags.append(is_cross_alias_boost)

            if observations and is_cross_alias_boost:
                obs_1 = obs_lookup.get(fv.obs_id_1)
                obs_2 = obs_lookup.get(fv.obs_id_2)
                if obs_1 and obs_2:
                    alias_1 = obs_1.entity.strip().lower()
                    alias_2 = obs_2.entity.strip().lower()
                    earliest_1 = earliest_obs_by_alias.get(alias_1)
                    earliest_2 = earliest_obs_by_alias.get(alias_2)

                    temporal_gap = fv.temporal_gap_sec if fv.temporal_gap_sec is not None else 10**9
                    rank = (
                        int(temporal_gap),
                        -int(round(similarity_score * 1000)),
                        min(obs_1.time_offset if hasattr(obs_1, "time_offset") else 0, obs_2.time_offset if hasattr(obs_2, "time_offset") else 0),
                        fv.obs_id_1 + "|" + fv.obs_id_2,
                    )

                    if fv.obs_id_1 == earliest_1:
                        current_rank = preferred_boost_pairs.get(fv.obs_id_1)
                        if current_rank is None or rank < current_rank:
                            preferred_boost_pairs[fv.obs_id_1] = rank
                            preferred_pair_index[fv.obs_id_1] = index

                    if fv.obs_id_2 == earliest_2:
                        current_rank = preferred_boost_pairs.get(fv.obs_id_2)
                        if current_rank is None or rank < current_rank:
                            preferred_boost_pairs[fv.obs_id_2] = rank
                            preferred_pair_index[fv.obs_id_2] = index

            scored_pair = ScoredPair(
                obs_id_1=fv.obs_id_1,
                obs_id_2=fv.obs_id_2,
                feature_vector=fv,
                similarity_score=similarity_score,
                score_components=components,
                rationale=rationale,
            )

            scored_pairs.append(scored_pair)
            scores.append(similarity_score)

        if observations:
            for index, scored_pair in enumerate(scored_pairs):
                if not candidate_boost_flags[index]:
                    continue

                fv = scored_pair.feature_vector
                keep_index_1 = preferred_pair_index.get(fv.obs_id_1)
                keep_index_2 = preferred_pair_index.get(fv.obs_id_2)
                if index != keep_index_1 and index != keep_index_2:
                    scored_pair.similarity_score = min(scored_pair.similarity_score, 0.69)
                    scored_pair.rationale.append("Co-event throttle: later alias occurrence suppressed")

            scores = [sp.similarity_score for sp in scored_pairs]

        # Sort by similarity score (descending)
        scored_pairs.sort(key=lambda sp: sp.similarity_score, reverse=True)

        # Compute statistics
        elapsed = time.time() - start_time

        high_score = sum(1 for s in scores if s >= 0.7)
        med_score = sum(1 for s in scores if 0.4 <= s < 0.7)
        low_score = sum(1 for s in scores if s < 0.4)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Create score distribution histogram
        distribution = {}
        for score in scores:
            bucket = f"{int(score * 10)}/10"
            distribution[bucket] = distribution.get(bucket, 0) + 1

        report = ScoringReport(
            total_pairs=len(scored_pairs),
            avg_similarity_score=avg_score,
            high_score_pairs=high_score,
            medium_score_pairs=med_score,
            low_score_pairs=low_score,
            score_distribution=distribution,
            processing_time_sec=elapsed,
        )

        return scored_pairs, report
