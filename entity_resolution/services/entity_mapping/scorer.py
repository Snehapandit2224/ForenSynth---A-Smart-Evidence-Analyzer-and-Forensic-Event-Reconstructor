"""Similarity scoring service for entity resolution.

Stage 5: Computes weighted similarity score from feature vectors.

Scoring Weights:
- temporal: 0.25 (time proximity)
- location: 0.20 (geographic match)
- context: 0.20 (semantic context)
- interaction: 0.10 (role/modality compatibility)
- lexical: 0.20 (text similarity)
- modality: 0.05 (modality type compatibility)

Total: 1.00

Output: Single similarity_score [0.0, 1.0] for downstream classification.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from .features import FeatureVector


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
    DEFAULT_WEIGHTS = {
        "temporal": 0.25,
        "location": 0.20,
        "context": 0.20,
        "interaction": 0.10,
        "lexical": 0.20,
        "modality": 0.05,
    }

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize scoring weights.

        Args:
            weights: Optional custom weights dict. If not provided, uses defaults.
                    Must have keys: temporal, location, context, interaction, lexical, modality
                    Must sum to approximately 1.0

        Raises:
            ValueError: If weights don't sum to 1.0 or missing required keys
        """
        if weights is None:
            self.weights = self.DEFAULT_WEIGHTS.copy()
        else:
            # Validate required keys
            required_keys = {"temporal", "location", "context", "interaction", "lexical", "modality"}
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
        - Apply configured weights to each feature
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
        # Extract features
        temporal = feature_vector.temporal_score
        location = feature_vector.location_score
        context = feature_vector.context_score
        interaction = feature_vector.interaction_score
        lexical = feature_vector.lexical_score
        modality = feature_vector.modality_compatibility_score

        # Get weights
        w_temporal = self.weights.get("temporal")
        w_location = self.weights.get("location")
        w_context = self.weights.get("context")
        w_interaction = self.weights.get("interaction")
        w_lexical = self.weights.get("lexical")
        w_modality = self.weights.get("modality")

        # Compute weighted contributions
        components = {
            "temporal": temporal * w_temporal,
            "location": location * w_location,
            "context": context * w_context,
            "interaction": interaction * w_interaction,
            "lexical": lexical * w_lexical,
            "modality": modality * w_modality,
        }

        # Compute total similarity score
        similarity_score = sum(components.values())

        # Build rationale
        rationale = []
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
        self, feature_vectors: List[FeatureVector]
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

        for fv in feature_vectors:
            similarity_score, components, rationale = self.compute_similarity_score(fv)

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
