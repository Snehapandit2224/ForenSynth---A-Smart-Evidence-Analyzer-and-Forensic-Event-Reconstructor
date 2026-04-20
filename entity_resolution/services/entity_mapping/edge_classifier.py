"""Edge classification service for entity resolution.

Stage 6: Classifies edges (observation pairs) as confirmed/candidate/rejected
based on similarity scores.

Classification Thresholds:
- >= 0.80 → confirmed (high confidence - likely same entity)
- 0.60-0.80 → candidate (moderate confidence - needs further review)
- < 0.60 → rejected (low confidence - likely different entities)

Output: Classified edge with decision and confidence level.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .scorer import ScoredPair


# ============================================================================
# Data Structures
# ============================================================================


class EdgeClassification(str, Enum):
    """Edge classification outcomes."""

    CONFIRMED = "confirmed"  # >= 0.80
    CANDIDATE = "candidate"  # 0.60-0.80
    REJECTED = "rejected"  # < 0.60


class ConfidenceLevel(str, Enum):
    """Confidence level for classification."""

    HIGH = "high"  # >= 0.80
    MEDIUM = "medium"  # 0.60-0.80
    LOW = "low"  # < 0.60


@dataclass
class ClassifiedEdge:
    """Classified observation pair (edge in entity graph)."""

    obs_id_1: str
    obs_id_2: str
    similarity_score: float  # [0.0, 1.0]
    classification: EdgeClassification  # confirmed/candidate/rejected
    confidence_level: ConfidenceLevel  # high/medium/low
    
    # Additional context
    distance_to_threshold: float = 0.0  # How far from nearest threshold
    threshold_description: str = ""  # Human-readable threshold context
    rationale: List[str] = field(default_factory=list)  # Decision justification

    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        return {
            "obs_id_1": self.obs_id_1,
            "obs_id_2": self.obs_id_2,
            "similarity_score": round(self.similarity_score, 3),
            "classification": self.classification.value,
            "confidence_level": self.confidence_level.value,
            "distance_to_threshold": round(self.distance_to_threshold, 3),
            "threshold_description": self.threshold_description,
            "rationale": self.rationale,
        }


@dataclass
class EdgeClassificationReport:
    """Report from edge classification stage."""

    total_edges: int = 0
    confirmed_edges: int = 0  # >= 0.80
    candidate_edges: int = 0  # 0.60-0.80
    rejected_edges: int = 0  # < 0.60
    
    # Statistics
    avg_confirmed_score: float = 0.0  # Average score for confirmed edges
    avg_candidate_score: float = 0.0  # Average score for candidate edges
    avg_rejected_score: float = 0.0  # Average score for rejected edges
    
    # Thresholds used
    confirmed_threshold: float = 0.80
    candidate_threshold_low: float = 0.60
    candidate_threshold_high: float = 0.80
    
    # Edge counts
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dict."""
        return {
            "total_edges": self.total_edges,
            "confirmed": self.confirmed_edges,
            "candidate": self.candidate_edges,
            "rejected": self.rejected_edges,
            "avg_confirmed_score": round(self.avg_confirmed_score, 3),
            "avg_candidate_score": round(self.avg_candidate_score, 3),
            "avg_rejected_score": round(self.avg_rejected_score, 3),
            "confirmed_threshold": self.confirmed_threshold,
            "candidate_threshold": f"{self.candidate_threshold_low}-{self.candidate_threshold_high}",
            "high_confidence": self.high_confidence_count,
            "medium_confidence": self.medium_confidence_count,
            "low_confidence": self.low_confidence_count,
            "processing_time_sec": round(self.processing_time_sec, 3),
        }


# ============================================================================
# Edge Classification Configuration
# ============================================================================


class ClassificationThresholds:
    """Configurable classification thresholds."""

    # Default thresholds
    _CONFIRMED_THRESHOLD = 0.80  # >= this = confirmed
    _CANDIDATE_LOW_THRESHOLD = 0.60  # >= this and < confirmed = candidate
    _CANDIDATE_HIGH_THRESHOLD = 0.80  # < this and >= low = candidate
    _REJECTED_THRESHOLD = 0.60  # < this = rejected

    def __init__(
        self,
        confirmed: float = 0.80,
        candidate_low: float = 0.60,
    ):
        """
        Initialize classification thresholds.

        Args:
            confirmed: Score threshold for confirmed edges (default 0.80)
            candidate_low: Score threshold for candidate edges (default 0.60)

        Raises:
            ValueError: If thresholds are invalid
        """
        if not (0.0 <= candidate_low <= confirmed <= 1.0):
            raise ValueError(
                f"Invalid thresholds: candidate_low ({candidate_low}) must be "
                f"<= confirmed ({confirmed}), and both in [0.0, 1.0]"
            )

        self.confirmed = confirmed
        self.candidate_low = candidate_low
        self.candidate_high = confirmed

    def classify(self, similarity_score: float) -> Tuple[EdgeClassification, ConfidenceLevel]:
        """
        Classify a similarity score.

        Args:
            similarity_score: [0.0, 1.0]

        Returns:
            Tuple of (classification, confidence_level)
        """
        if similarity_score >= self.confirmed:
            return EdgeClassification.CONFIRMED, ConfidenceLevel.HIGH
        elif similarity_score >= self.candidate_low:
            return EdgeClassification.CANDIDATE, ConfidenceLevel.MEDIUM
        else:
            return EdgeClassification.REJECTED, ConfidenceLevel.LOW

    def distance_to_nearest_threshold(self, similarity_score: float) -> float:
        """
        Compute distance to nearest classification threshold.

        Args:
            similarity_score: [0.0, 1.0]

        Returns:
            Minimum distance to a threshold
        """
        thresholds = [self.candidate_low, self.confirmed]
        return min(abs(similarity_score - t) for t in thresholds)

    def to_dict(self) -> Dict:
        """Return thresholds as dict."""
        return {
            "confirmed": self.confirmed,
            "candidate_low": self.candidate_low,
            "candidate_high": self.candidate_high,
        }


# ============================================================================
# Edge Classification Implementation
# ============================================================================


class EdgeClassifier:
    """Classifies observation pairs (edges) based on similarity scores."""

    def __init__(self, thresholds: ClassificationThresholds = None):
        """
        Initialize edge classifier with optional custom thresholds.

        Args:
            thresholds: Optional ClassificationThresholds. If not provided, uses defaults.
        """
        if thresholds is None:
            self.thresholds = ClassificationThresholds()
        else:
            self.thresholds = thresholds

    def classify_edge(self, scored_pair: ScoredPair) -> ClassifiedEdge:
        """
        Classify a single scored pair as an edge.

        Args:
            scored_pair: ScoredPair from Stage 5 (scorer)

        Returns:
            ClassifiedEdge with classification and confidence
        """
        similarity_score = scored_pair.similarity_score

        # Get classification and confidence
        classification, confidence = self.thresholds.classify(similarity_score)

        # Compute distance to nearest threshold
        distance = self.thresholds.distance_to_nearest_threshold(similarity_score)

        # Build threshold description
        if classification == EdgeClassification.CONFIRMED:
            threshold_desc = f"Well above confirmed threshold ({self.thresholds.confirmed})"
        elif classification == EdgeClassification.CANDIDATE:
            threshold_desc = (
                f"In candidate range "
                f"({self.thresholds.candidate_low}-{self.thresholds.confirmed})"
            )
        else:  # REJECTED
            threshold_desc = f"Below candidate threshold ({self.thresholds.candidate_low})"

        # Build rationale
        rationale = scored_pair.rationale.copy() if scored_pair.rationale else []
        rationale.append(
            f"Classified as {classification.value} "
            f"({distance:.3f} from nearest threshold)"
        )

        classified_edge = ClassifiedEdge(
            obs_id_1=scored_pair.obs_id_1,
            obs_id_2=scored_pair.obs_id_2,
            similarity_score=similarity_score,
            classification=classification,
            confidence_level=confidence,
            distance_to_threshold=distance,
            threshold_description=threshold_desc,
            rationale=rationale,
        )

        return classified_edge

    def classify_edges(
        self, scored_pairs: List[ScoredPair]
    ) -> Tuple[List[ClassifiedEdge], EdgeClassificationReport]:
        """
        Classify all scored pairs into edges.

        Args:
            scored_pairs: List of ScoredPair from Stage 5

        Returns:
            Tuple of (classified_edges, report)
            - classified_edges: List of ClassifiedEdge sorted by classification then score
            - report: EdgeClassificationReport with statistics
        """
        import time
        start_time = time.time()

        classified_edges: List[ClassifiedEdge] = []
        confirmed_scores: List[float] = []
        candidate_scores: List[float] = []
        rejected_scores: List[float] = []

        for scored_pair in scored_pairs:
            edge = self.classify_edge(scored_pair)
            classified_edges.append(edge)

            # Track scores by classification
            if edge.classification == EdgeClassification.CONFIRMED:
                confirmed_scores.append(edge.similarity_score)
            elif edge.classification == EdgeClassification.CANDIDATE:
                candidate_scores.append(edge.similarity_score)
            else:
                rejected_scores.append(edge.similarity_score)

        # Sort by classification (confirmed first, then candidate, then rejected)
        # Within each group, sort by score descending
        classification_order = {
            EdgeClassification.CONFIRMED: 0,
            EdgeClassification.CANDIDATE: 1,
            EdgeClassification.REJECTED: 2,
        }
        classified_edges.sort(
            key=lambda e: (classification_order[e.classification], -e.similarity_score)
        )

        # Compute averages
        avg_confirmed = sum(confirmed_scores) / len(confirmed_scores) if confirmed_scores else 0.0
        avg_candidate = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0
        avg_rejected = sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0.0

        # Count confidence levels
        high_conf = sum(1 for e in classified_edges if e.confidence_level == ConfidenceLevel.HIGH)
        med_conf = sum(1 for e in classified_edges if e.confidence_level == ConfidenceLevel.MEDIUM)
        low_conf = sum(1 for e in classified_edges if e.confidence_level == ConfidenceLevel.LOW)

        elapsed = time.time() - start_time

        report = EdgeClassificationReport(
            total_edges=len(classified_edges),
            confirmed_edges=len(confirmed_scores),
            candidate_edges=len(candidate_scores),
            rejected_edges=len(rejected_scores),
            avg_confirmed_score=avg_confirmed,
            avg_candidate_score=avg_candidate,
            avg_rejected_score=avg_rejected,
            confirmed_threshold=self.thresholds.confirmed,
            candidate_threshold_low=self.thresholds.candidate_low,
            candidate_threshold_high=self.thresholds.candidate_high,
            high_confidence_count=high_conf,
            medium_confidence_count=med_conf,
            low_confidence_count=low_conf,
            processing_time_sec=elapsed,
        )

        return classified_edges, report
