"""Entity Mapping Services."""

from .intake import IntakeValidator, IntakeReport
from .normalizer import AliasNormalizer, Normalizer
from .blocker import Blocker, CandidatePair, BlockingReport, BlockingSignal
from .features import Features, FeatureVector, FeatureReport, FeatureType
from .scorer import Scorer, ScoredPair, ScoringReport, ScoringWeights
from .edge_classifier import EdgeClassifier, ClassifiedEdge, EdgeClassificationReport, EdgeClassification, ConfidenceLevel
from .graph_builder import GraphBuilder, GraphBuildingReport
from .clusterer import Clusterer, EntityCluster, ClusteringReport
from .candidate_attacher import CandidateAttacher, CandidateAttachment, AttachmentReport as AttacherReport, ConflictFlag
from .conflict_handler import ConflictHandler, DetectedConflict, ConflictReport, ConflictType, ConflictSeverity
from .labeler import Labeler, CanonicalEntity, LabelingReport
from .resolver import Resolver, ResolutionResult
from .config import PipelineConfiguration

__all__ = [
    "IntakeValidator",
    "IntakeReport",
    "AliasNormalizer",
    "Normalizer",
    "Blocker",
    "CandidatePair",
    "BlockingReport",
    "BlockingSignal",
    "Features",
    "FeatureVector",
    "FeatureReport",
    "FeatureType",
    "Scorer",
    "ScoredPair",
    "ScoringReport",
    "ScoringWeights",
    "EdgeClassifier",
    "ClassifiedEdge",
    "EdgeClassificationReport",
    "EdgeClassification",
    "ConfidenceLevel",
    "GraphBuilder",
    "GraphBuildingReport",
    "Clusterer",
    "EntityCluster",
    "ClusteringReport",
    "CandidateAttacher",
    "CandidateAttachment",
    "AttacherReport",
    "ConflictFlag",
    "ConflictHandler",
    "DetectedConflict",
    "ConflictReport",
    "ConflictType",
    "ConflictSeverity",
    "Labeler",
    "CanonicalEntity",
    "LabelingReport",
    "Resolver",
    "ResolutionResult",
    "PipelineConfiguration",
]
