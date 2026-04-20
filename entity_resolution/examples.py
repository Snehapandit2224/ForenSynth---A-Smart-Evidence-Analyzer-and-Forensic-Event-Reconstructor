"""Example usage and validation of entity resolution schemas and services."""

import json
from datetime import datetime

from .schemas.observation import Observation, CaseInput, Modality, NormalizedObservation
from .services.entity_mapping.intake import IntakeValidator
from .services.entity_mapping.normalizer import Normalizer, AliasNormalizer
from .services.entity_mapping.blocker import Blocker
from .services.entity_mapping.features import Features
from .services.entity_mapping.scorer import Scorer
from .services.entity_mapping.edge_classifier import EdgeClassifier
from .services.entity_mapping.graph_builder import GraphBuilder
from .services.entity_mapping.clusterer import Clusterer
from .services.entity_mapping.candidate_attacher import CandidateAttacher
from .services.entity_mapping.conflict_handler import ConflictHandler
from .services.entity_mapping.labeler import Labeler
from .services.entity_mapping.resolver import Resolver, PipelineConfiguration


def example_intake_validation():
    """Example: Intake validation."""
    print("=" * 70)
    print("EXAMPLE 1: INTAKE VALIDATION")
    print("=" * 70)

    # Sample case input
    case_input = {
        "case_id": "CASE_20250315_001",
        "observations": [
            {
                "obs_id": "O1",
                "entity": "Person_05",
                "role": "suspect",
                "modality": "video",
                "source": "camera_1",
                "location": "ATM booth interior",
                "content": "Suspect approaching ATM booth",
                "timestamp": "2024-01-15T10:15:30Z",
                "time_offset": 180,
                "confidence": 0.95,
                "noise_tags": ["blur"],
            },
            {
                "obs_id": "O2",
                "entity": "Speaker_A",
                "role": "suspect",
                "modality": "audio",
                "source": "mic_booth",
                "location": "ATM booth interior",
                "content": "Voice near machine",
                "timestamp": "2024-01-15T10:14:30Z",
                "time_offset": 120,
                "confidence": 0.89,
                "noise_tags": [],
            },
            {
                "obs_id": "O3",
                "entity": "Person_05",
                "role": "suspect",
                "modality": "video",
                "source": "camera_1",
                "location": "ATM booth interior",
                "content": "Suspect operating keypad",
                "timestamp": "2024-01-15T10:16:00Z",
                "time_offset": 210,
                "confidence": 0.92,
                "noise_tags": ["partial_occlusion"],
            },
        ],
    }

    # Validate
    validator = IntakeValidator()
    observations, report = validator.validate_case(case_input)

    print(f"[OK] Valid observations: {report.valid_observations}")
    print(f"[FAIL] Invalid observations: {report.invalid_observations}")
    print(f"[WARN] Duplicate observations: {report.duplicate_observations}")
    print(f"Errors: {len(report.errors)}")
    print(f"Warnings: {len(report.warnings)}")

    if observations:
        print(f"\nFirst observation:")
        print(f"  obs_id: {observations[0].obs_id}")
        print(f"  entity: {observations[0].entity}")
        print(f"  modality: {observations[0].modality}")
        print(f"  confidence: {observations[0].confidence}")


def example_alias_normalization():
    """Example: Alias normalization."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: ALIAS NORMALIZATION")
    print("=" * 70)

    test_aliases = [
        "Person_05",
        "person_12",
        "Speaker_A",
        "speaker_z",
        "sms_15",
        "email_42",
        "report_001",
        "log_999",
    ]

    for alias in test_aliases:
        try:
            normalized = AliasNormalizer.normalize_alias(alias)
            print(
                f"{alias:15} -> type={normalized.alias_type:10} "
                f"id={normalized.alias_id:3} modality={normalized.modality_hint:5} "
                f"conf={normalized.confidence}"
            )
        except ValueError as e:
            print(f"{alias:15} -> ERROR: {e}")


def example_observation_normalization():
    """Example: Observation normalization."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: OBSERVATION NORMALIZATION")
    print("=" * 70)

    # Create test observations
    obs1 = Observation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="  SUSPECT  approaching    ATM booth   ",  # Extra whitespace
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
    )

    obs2 = Observation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM BOOTH INTERIOR",  # Mixed case
        content="Voice near MACHINE",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
    )

    # Set base time
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")
    normalizer = Normalizer(case_base_time=base_time)

    normalized, report = normalizer.normalize_observations([obs1, obs2])

    print(f"Successfully normalized: {report.successfully_normalized}/{report.total_observations}")

    for i, norm_obs in enumerate(normalized):
        print(f"\nObservation {i + 1}:")
        print(f"  obs_id: {norm_obs.obs_id}")
        print(f"  entity: {norm_obs.entity}")
        print(f"  Original content: '{norm_obs.content}'")
        print(f"  Normalized content: '{norm_obs.content_normalized}'")
        print(f"  Original timestamp: {norm_obs.timestamp}")
        print(f"  Parsed datetime: {norm_obs.timestamp_dt}")
        print(f"  Offset from base: {norm_obs.timestamp_offset_sec}s")


def example_invalid_input():
    """Example: Handling invalid input."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: ERROR HANDLING (Invalid Input)")
    print("=" * 70)

    # Invalid case: missing required field
    case_input = {
        "case_id": "CASE_001",
        "observations": [
            {
                "obs_id": "O1",
                # Missing 'entity' field
                "role": "suspect",
                "modality": "video",
                "source": "camera_1",
                "location": "ATM booth",
                "content": "...",
                "timestamp": "2024-01-15T10:15:30Z",
                "time_offset": 180,
                "confidence": 0.95,
            }
        ],
    }

    validator = IntakeValidator()
    observations, report = validator.validate_case(case_input)

    print(f"Valid observations: {report.valid_observations}")
    print(f"Invalid observations: {report.invalid_observations}")
    print(f"Errors ({len(report.errors)}):")
    for error in report.errors:
        print(f"  - [{error.error_type}] {error.obs_id}: {error.message}")


def example_duplicate_detection():
    """Example: Duplicate observation detection."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: DUPLICATE DETECTION")
    print("=" * 70)

    # Two nearly identical observations (different confidence)
    case_input = {
        "case_id": "CASE_002",
        "observations": [
            {
                "obs_id": "O1",
                "entity": "Person_05",
                "role": "suspect",
                "modality": "video",
                "source": "camera_1",
                "location": "ATM booth interior",
                "content": "Suspect approaching ATM",
                "timestamp": "2024-01-15T10:15:30Z",
                "time_offset": 180,
                "confidence": 0.85,
                "noise_tags": [],
            },
            {
                "obs_id": "O1_dup",  # Different obs_id
                "entity": "Person_05",
                "role": "suspect",
                "modality": "video",
                "source": "camera_1",
                "location": "ATM booth interior",
                "content": "Suspect approaching ATM",  # Identical content
                "timestamp": "2024-01-15T10:15:30Z",
                "time_offset": 180,
                "confidence": 0.95,  # Different confidence
                "noise_tags": [],
            },
        ],
    }

    validator = IntakeValidator()
    observations, report = validator.validate_case(case_input)

    print(f"Valid observations: {report.valid_observations}")
    print(f"Duplicate observations detected: {report.duplicate_observations}")
    if report.warnings:
        print(f"Warnings ({len(report.warnings)}):")
        for warning in report.warnings:
            print(f"  - {warning}")


def example_blocking():
    """Example: Candidate pair generation with blocking."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: BLOCKING (CANDIDATE PAIR GENERATION)")
    print("=" * 70)

    # Create normalized observations (simulating stages 1-2)
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")

    obs_1 = NormalizedObservation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect approaching ATM",
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:15:30+00:00"),
        timestamp_offset_sec=930,
        content_normalized="suspect approaching atm",
    )

    obs_2 = NormalizedObservation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM booth interior",
        content="Voice near machine",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:14:30+00:00"),
        timestamp_offset_sec=870,
        content_normalized="voice near machine",
    )

    obs_3 = NormalizedObservation(
        obs_id="O3",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect operating keypad",
        timestamp="2024-01-15T10:16:00Z",
        time_offset=210,
        confidence=0.92,
        noise_tags=["partial_occlusion"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:16:00+00:00"),
        timestamp_offset_sec=960,
        content_normalized="suspect operating keypad",
    )

    obs_4 = NormalizedObservation(
        obs_id="O4",
        entity="Person_08",
        role="witness",
        modality=Modality.VIDEO,
        source="camera_2",
        location="ATM street frontage",
        content="Witness observing ATM",
        timestamp="2024-01-15T14:30:00Z",  # Different time (4+ hours later)
        time_offset=16200,
        confidence=0.88,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T14:30:00+00:00"),
        timestamp_offset_sec=16200,
        content_normalized="witness observing atm",
    )

    observations = [obs_1, obs_2, obs_3, obs_4]

    # Generate candidates
    blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
    candidates, report = blocker.generate_candidates(observations)

    print(f"Blocking Report:")
    print(f"  Total candidate pairs (n choose 2): {report.total_candidate_pairs}")
    print(f"  After hard rejections: {report.pairs_after_hard_reject}")
    print(f"  High priority (>=0.7): {report.pairs_with_high_priority}")
    print(f"  Medium priority (0.4-0.7): {report.pairs_with_medium_priority}")
    print(f"  Low priority (<0.4): {report.pairs_with_low_priority}")
    print(f"  Average priority: {report.avg_priority:.3f}")
    print(f"  Cardinality reduction: {report.cardinality_reduction:.1%}")

    print(f"\nTop candidate pairs (sorted by priority):")
    for i, pair in enumerate(candidates[:5]):  # Top 5
        print(
            f"\n  {i+1}. ({pair.obs_id_1}, {pair.obs_id_2}) - Priority: {pair.priority:.3f}"
        )
        print(f"     Rationale: {', '.join(pair.rationale)}")
        print(f"     Signals:")
        for signal, score in pair.signals.items():
            print(f"       - {signal.value}: {score:.3f}")


def example_features():
    """Example: Feature computation for candidate pairs."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: FEATURE COMPUTATION")
    print("=" * 70)

    # Create normalized observations (simulating stages 1-2)
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")

    obs_1 = NormalizedObservation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect approaching ATM",
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:15:30+00:00"),
        timestamp_offset_sec=930,
        content_normalized="suspect approaching atm",
    )

    obs_2 = NormalizedObservation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM booth interior",
        content="Voice near machine",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:14:30+00:00"),
        timestamp_offset_sec=870,
        content_normalized="voice near machine",
    )

    obs_3 = NormalizedObservation(
        obs_id="O3",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect operating keypad",
        timestamp="2024-01-15T10:16:00Z",
        time_offset=210,
        confidence=0.92,
        noise_tags=["partial_occlusion"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:16:00+00:00"),
        timestamp_offset_sec=960,
        content_normalized="suspect operating keypad",
    )

    obs_4 = NormalizedObservation(
        obs_id="O4",
        entity="Person_08",
        role="witness",
        modality=Modality.VIDEO,
        source="camera_2",
        location="ATM street frontage",
        content="Witness observing ATM",
        timestamp="2024-01-15T14:30:00Z",
        time_offset=16200,
        confidence=0.88,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T14:30:00+00:00"),
        timestamp_offset_sec=16200,
        content_normalized="witness observing atm",
    )

    observations = [obs_1, obs_2, obs_3, obs_4]

    # Stage 3: Generate candidates
    blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
    candidates, blocking_report = blocker.generate_candidates(observations)

    # Stage 4: Compute features
    features_computer = Features()
    feature_vectors, feature_report = features_computer.compute_features(
        candidates, observations
    )

    print(f"Feature Computation Report:")
    print(f"  Total pairs evaluated: {feature_report.total_pairs}")
    print(f"  High confidence (>=0.7): {feature_report.high_confidence_pairs}")
    print(f"  Medium confidence (0.4-0.7): {feature_report.medium_confidence_pairs}")
    print(f"  Low confidence (<0.4): {feature_report.low_confidence_pairs}")
    print(f"  Average combined score: {feature_report.avg_combined_score:.3f}")
    print(f"  Processing time: {feature_report.processing_time_sec:.3f}s")

    print(f"\nFeature Averages:")
    for feature_type, avg_score in feature_report.feature_averages.items():
        print(f"  {feature_type}: {avg_score:.3f}")

    print(f"\nTop Feature Vectors (sorted by combined score):")
    for i, fv in enumerate(feature_vectors[:5]):
        print(f"\n  {i+1}. ({fv.obs_id_1}, {fv.obs_id_2}) - Combined Score: {fv.combined_score:.3f}")
        print(f"     Blocker Priority: {fv.pair_priority:.3f}")
        print(f"     Individual Scores:")
        print(f"       - Temporal: {fv.temporal_score:.3f} (gap {fv.temporal_gap_sec}s)")
        print(f"       - Location: {fv.location_score:.3f} ({fv.location_distance})")
        print(f"       - Context: {fv.context_score:.3f}")
        print(f"       - Interaction: {fv.interaction_score:.3f}")
        print(f"       - Lexical: {fv.lexical_score:.3f} ({fv.content_similarity:.1%} similarity)")
        print(f"       - Modality: {fv.modality_compatibility_score:.3f} ({fv.modality_pair})")
        if fv.rationale:
            print(f"     Rationale: {', '.join(fv.rationale)}")


def example_scorer():
    """Example: Similarity scoring."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: SIMILARITY SCORING")
    print("=" * 70)

    # Create normalized observations
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")

    obs_1 = NormalizedObservation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect approaching ATM",
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:15:30+00:00"),
        timestamp_offset_sec=930,
        content_normalized="suspect approaching atm",
    )

    obs_2 = NormalizedObservation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM booth interior",
        content="Voice near machine",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:14:30+00:00"),
        timestamp_offset_sec=870,
        content_normalized="voice near machine",
    )

    obs_3 = NormalizedObservation(
        obs_id="O3",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect operating keypad",
        timestamp="2024-01-15T10:16:00Z",
        time_offset=210,
        confidence=0.92,
        noise_tags=["partial_occlusion"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:16:00+00:00"),
        timestamp_offset_sec=960,
        content_normalized="suspect operating keypad",
    )

    obs_4 = NormalizedObservation(
        obs_id="O4",
        entity="Person_08",
        role="witness",
        modality=Modality.VIDEO,
        source="camera_2",
        location="ATM street frontage",
        content="Witness observing ATM",
        timestamp="2024-01-15T14:30:00Z",
        time_offset=16200,
        confidence=0.88,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T14:30:00+00:00"),
        timestamp_offset_sec=16200,
        content_normalized="witness observing atm",
    )

    observations = [obs_1, obs_2, obs_3, obs_4]

    # Stage 3: Generate candidates
    blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
    candidates, blocking_report = blocker.generate_candidates(observations)

    # Stage 4: Compute features
    features_computer = Features()
    feature_vectors, feature_report = features_computer.compute_features(
        candidates, observations
    )

    # Stage 5: Score pairs
    scorer = Scorer()
    scored_pairs, scoring_report = scorer.score_pairs(feature_vectors)

    print(f"Scoring Report:")
    print(f"  Total pairs scored: {scoring_report.total_pairs}")
    print(f"  High score (>=0.7): {scoring_report.high_score_pairs}")
    print(f"  Medium score (0.4-0.7): {scoring_report.medium_score_pairs}")
    print(f"  Low score (<0.4): {scoring_report.low_score_pairs}")
    print(f"  Average similarity score: {scoring_report.avg_similarity_score:.3f}")
    print(f"  Processing time: {scoring_report.processing_time_sec:.3f}s")

    print(f"\n  Score Distribution:")
    for bucket, count in sorted(scoring_report.score_distribution.items()):
        print(f"    {bucket}: {count} pairs")

    print(f"\nScoring Weights (Default):")
    print(f"  Temporal: 0.25, Location: 0.20, Context: 0.20")
    print(f"  Interaction: 0.10, Lexical: 0.20, Modality: 0.05")

    print(f"\nTop Scored Pairs:")
    for i, sp in enumerate(scored_pairs[:3]):
        print(f"\n  {i+1}. ({sp.obs_id_1}, {sp.obs_id_2}) - Similarity: {sp.similarity_score:.3f}")
        print(f"     Score Components:")
        for feature, contribution in sp.score_components.items():
            print(f"       - {feature}: {contribution:.3f}")
        if sp.rationale:
            print(f"     Rationale: {', '.join(sp.rationale[:2])}")


def example_edge_classifier():
    """Example: Edge classification."""
    print("\n" + "=" * 70)
    print("EXAMPLE 9: EDGE CLASSIFICATION")
    print("=" * 70)

    # Create normalized observations
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")

    obs_1 = NormalizedObservation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect approaching ATM",
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:15:30+00:00"),
        timestamp_offset_sec=930,
        content_normalized="suspect approaching atm",
    )

    obs_2 = NormalizedObservation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM booth interior",
        content="Voice near machine",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:14:30+00:00"),
        timestamp_offset_sec=870,
        content_normalized="voice near machine",
    )

    obs_3 = NormalizedObservation(
        obs_id="O3",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect operating keypad",
        timestamp="2024-01-15T10:16:00Z",
        time_offset=210,
        confidence=0.92,
        noise_tags=["partial_occlusion"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:16:00+00:00"),
        timestamp_offset_sec=960,
        content_normalized="suspect operating keypad",
    )

    obs_4 = NormalizedObservation(
        obs_id="O4",
        entity="Person_08",
        role="witness",
        modality=Modality.VIDEO,
        source="camera_2",
        location="ATM street frontage",
        content="Witness observing ATM",
        timestamp="2024-01-15T14:30:00Z",
        time_offset=16200,
        confidence=0.88,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T14:30:00+00:00"),
        timestamp_offset_sec=16200,
        content_normalized="witness observing atm",
    )

    observations = [obs_1, obs_2, obs_3, obs_4]

    # Stages 3-5: Generate candidates, compute features, score
    blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
    candidates, _ = blocker.generate_candidates(observations)

    features_computer = Features()
    feature_vectors, _ = features_computer.compute_features(candidates, observations)

    scorer = Scorer()
    scored_pairs, _ = scorer.score_pairs(feature_vectors)

    # Stage 6: Classify edges
    classifier = EdgeClassifier()
    classified_edges, classification_report = classifier.classify_edges(scored_pairs)

    print(f"Edge Classification Report:")
    print(f"  Total edges: {classification_report.total_edges}")
    print(f"  Confirmed (>=0.80): {classification_report.confirmed_edges}")
    print(f"  Candidate (0.60-0.80): {classification_report.candidate_edges}")
    print(f"  Rejected (<0.60): {classification_report.rejected_edges}")
    print(f"  Processing time: {classification_report.processing_time_sec:.3f}s")

    print(f"\n  Classification Thresholds:")
    print(f"    Confirmed threshold: {classification_report.confirmed_threshold}")
    print(f"    Candidate range: {classification_report.candidate_threshold_low}-{classification_report.candidate_threshold_high}")

    print(f"\n  Average Scores by Classification:")
    if classification_report.confirmed_edges > 0:
        print(f"    Confirmed edges: {classification_report.avg_confirmed_score:.3f}")
    if classification_report.candidate_edges > 0:
        print(f"    Candidate edges: {classification_report.avg_candidate_score:.3f}")
    if classification_report.rejected_edges > 0:
        print(f"    Rejected edges: {classification_report.avg_rejected_score:.3f}")

    print(f"\n  Confidence Levels:")
    print(f"    High: {classification_report.high_confidence_count}")
    print(f"    Medium: {classification_report.medium_confidence_count}")
    print(f"    Low: {classification_report.low_confidence_count}")

    print(f"\nClassified Edges (sorted by classification then score):")
    for i, edge in enumerate(classified_edges[:5]):
        print(
            f"\n  {i+1}. ({edge.obs_id_1}, {edge.obs_id_2}) - "
            f"Classification: {edge.classification.value.upper()}"
        )
        print(f"     Similarity Score: {edge.similarity_score:.3f}")
        print(f"     Confidence: {edge.confidence_level.value}")
        print(f"     Distance to threshold: {edge.distance_to_threshold:.3f}")
        print(f"     {edge.threshold_description}")


def example_graph_builder():
    """Example: Graph construction from classified edges."""
    print("\n" + "=" * 70)
    print("EXAMPLE 10: GRAPH BUILDING")
    print("=" * 70)

    # Create normalized observations
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")

    obs_1 = NormalizedObservation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect approaching ATM",
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:15:30+00:00"),
        timestamp_offset_sec=930,
        content_normalized="suspect approaching atm",
    )

    obs_2 = NormalizedObservation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM booth interior",
        content="Voice near machine",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:14:30+00:00"),
        timestamp_offset_sec=870,
        content_normalized="voice near machine",
    )

    obs_3 = NormalizedObservation(
        obs_id="O3",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect operating keypad",
        timestamp="2024-01-15T10:16:00Z",
        time_offset=210,
        confidence=0.92,
        noise_tags=["partial_occlusion"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:16:00+00:00"),
        timestamp_offset_sec=960,
        content_normalized="suspect operating keypad",
    )

    obs_4 = NormalizedObservation(
        obs_id="O4",
        entity="Person_08",
        role="witness",
        modality=Modality.VIDEO,
        source="camera_2",
        location="ATM street frontage",
        content="Witness observing ATM",
        timestamp="2024-01-15T14:30:00Z",
        time_offset=16200,
        confidence=0.88,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T14:30:00+00:00"),
        timestamp_offset_sec=16200,
        content_normalized="witness observing atm",
    )

    observations = [obs_1, obs_2, obs_3, obs_4]
    all_obs_ids = [obs.obs_id for obs in observations]

    # Stages 3-6: Generate candidates, features, score, classify
    blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
    candidates, _ = blocker.generate_candidates(observations)

    features_computer = Features()
    feature_vectors, _ = features_computer.compute_features(candidates, observations)

    scorer = Scorer()
    scored_pairs, _ = scorer.score_pairs(feature_vectors)

    classifier = EdgeClassifier()
    classified_edges, _ = classifier.classify_edges(scored_pairs)

    # Stage 7: Build graph (only CONFIRMED edges)
    graph_builder = GraphBuilder()
    graph, graph_report = graph_builder.build_graph(classified_edges, all_obs_ids)

    print(f"Graph Building Report:")
    print(f"  Total observations (nodes): {graph_report.total_observations}")
    print(f"  Confirmed edges added: {graph_report.total_confirmed_edges}")
    print(f"  Candidate edges (not added): {graph_report.total_candidate_edges}")
    print(f"  Rejected edges (not added): {graph_report.total_rejected_edges}")
    print(f"  Processing time: {graph_report.processing_time_sec:.3f}s")

    print(f"\n  Graph Statistics:")
    print(f"    Nodes: {graph_report.nodes_count}")
    print(f"    Edges: {graph_report.edges_count}")
    print(f"    Avg node degree: {graph_report.avg_node_degree:.2f}")
    print(f"    Max node degree: {graph_report.max_node_degree}")
    print(f"    Min node degree: {graph_report.min_node_degree}")
    print(f"    Isolated nodes (degree 0): {graph_report.isolated_nodes}")

    print(f"\n  Edge Similarity Statistics:")
    print(f"    Avg similarity: {graph_report.avg_edge_similarity:.3f}")
    print(f"    Min similarity: {graph_report.min_edge_similarity:.3f}")
    print(f"    Max similarity: {graph_report.max_edge_similarity:.3f}")

    print(f"\nGraph Edges (CONFIRMED only):")
    for i, (u, v, data) in enumerate(graph.edges(data=True)):
        print(f"  {i+1}. ({u}, {v}) - Similarity: {data.get('weight', 0):.3f}")


def example_clusterer():
    """Example: Clustering via connected components."""
    print("\n" + "=" * 70)
    print("EXAMPLE 11: CLUSTERING (CONNECTED COMPONENTS)")
    print("=" * 70)

    # Create normalized observations
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")

    obs_1 = NormalizedObservation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect approaching ATM",
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:15:30+00:00"),
        timestamp_offset_sec=930,
        content_normalized="suspect approaching atm",
    )

    obs_2 = NormalizedObservation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM booth interior",
        content="Voice near machine",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:14:30+00:00"),
        timestamp_offset_sec=870,
        content_normalized="voice near machine",
    )

    obs_3 = NormalizedObservation(
        obs_id="O3",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect operating keypad",
        timestamp="2024-01-15T10:16:00Z",
        time_offset=210,
        confidence=0.92,
        noise_tags=["partial_occlusion"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:16:00+00:00"),
        timestamp_offset_sec=960,
        content_normalized="suspect operating keypad",
    )

    obs_4 = NormalizedObservation(
        obs_id="O4",
        entity="Person_08",
        role="witness",
        modality=Modality.VIDEO,
        source="camera_2",
        location="ATM street frontage",
        content="Witness observing ATM",
        timestamp="2024-01-15T14:30:00Z",
        time_offset=16200,
        confidence=0.88,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T14:30:00+00:00"),
        timestamp_offset_sec=16200,
        content_normalized="witness observing atm",
    )

    observations = [obs_1, obs_2, obs_3, obs_4]
    all_obs_ids = [obs.obs_id for obs in observations]

    # Stages 3-7: Full pipeline through graph building
    blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
    candidates, _ = blocker.generate_candidates(observations)

    features_computer = Features()
    feature_vectors, _ = features_computer.compute_features(candidates, observations)

    scorer = Scorer()
    scored_pairs, _ = scorer.score_pairs(feature_vectors)

    classifier = EdgeClassifier()
    classified_edges, _ = classifier.classify_edges(scored_pairs)

    graph_builder = GraphBuilder()
    graph, _ = graph_builder.build_graph(classified_edges, all_obs_ids)

    # Stage 8: Cluster observations
    clusterer = Clusterer()
    clusters, clustering_report = clusterer.cluster_observations(graph, all_obs_ids)

    print(f"Clustering Report:")
    print(f"  Total observations: {clustering_report.total_observations}")
    print(f"  Total clusters identified: {clustering_report.total_clusters}")
    print(f"  Processing time: {clustering_report.processing_time_sec:.3f}s")

    print(f"\n  Cluster Size Distribution:")
    print(f"    Avg cluster size: {clustering_report.avg_cluster_size:.2f}")
    print(f"    Max cluster size: {clustering_report.max_cluster_size}")
    print(f"    Min cluster size: {clustering_report.min_cluster_size}")
    print(f"    Singleton clusters (size 1): {clustering_report.singleton_clusters}")
    print(f"    Pair clusters (size 2): {clustering_report.pair_clusters}")
    print(f"    Triplet clusters (size 3): {clustering_report.triplet_clusters}")
    print(f"    Large clusters (size >= 4): {clustering_report.large_clusters}")

    print(f"\n  Coverage: {clustering_report.clustered_coverage_ratio:.1%} of observations in clusters")

    print(f"\nCanonical Entity Clusters (sorted by size):")
    for i, cluster in enumerate(clusters):
        print(f"\n  {i+1}. {cluster.cluster_id} (size {cluster.size})")
        print(f"     Observation IDs: {', '.join(cluster.obs_ids)}")
        
        # Decode what observations these are
        cluster_entities = set()
        for obs_id in cluster.obs_ids:
            for obs in observations:
                if obs.obs_id == obs_id:
                    modality_str = obs.modality if isinstance(obs.modality, str) else obs.modality.value
                    cluster_entities.add(f"{obs.entity}({modality_str})")
        print(f"     Entities: {', '.join(sorted(cluster_entities))}")


def example_candidate_attacher():
    """Example: Attaching unresolved mentions to clusters via candidate edges."""
    print("\n" + "=" * 70)
    print("EXAMPLE 12: CANDIDATE ATTACHMENT")
    print("=" * 70)

    # Create normalized observations
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")

    obs_1 = NormalizedObservation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect approaching ATM",
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:15:30+00:00"),
        timestamp_offset_sec=930,
        content_normalized="suspect approaching atm",
    )

    obs_2 = NormalizedObservation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM booth interior",
        content="Voice near machine",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:14:30+00:00"),
        timestamp_offset_sec=870,
        content_normalized="voice near machine",
    )

    obs_3 = NormalizedObservation(
        obs_id="O3",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect operating keypad",
        timestamp="2024-01-15T10:16:00Z",
        time_offset=210,
        confidence=0.92,
        noise_tags=["partial_occlusion"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:16:00+00:00"),
        timestamp_offset_sec=960,
        content_normalized="suspect operating keypad",
    )

    obs_4 = NormalizedObservation(
        obs_id="O4",
        entity="Person_08",
        role="witness",
        modality=Modality.VIDEO,
        source="camera_2",
        location="ATM street frontage",
        content="Witness observing ATM",
        timestamp="2024-01-15T14:30:00Z",
        time_offset=16200,
        confidence=0.88,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T14:30:00+00:00"),
        timestamp_offset_sec=16200,
        content_normalized="witness observing atm",
    )

    observations = [obs_1, obs_2, obs_3, obs_4]
    all_obs_ids = [obs.obs_id for obs in observations]

    # Stages 3-6: Generate candidates, features, score, classify
    blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
    candidates, _ = blocker.generate_candidates(observations)

    features_computer = Features()
    feature_vectors, _ = features_computer.compute_features(candidates, observations)

    scorer = Scorer()
    scored_pairs, _ = scorer.score_pairs(feature_vectors)

    classifier = EdgeClassifier()
    classified_edges, _ = classifier.classify_edges(scored_pairs)

    # Stages 7-8: Build graph and cluster
    graph_builder = GraphBuilder()
    graph, _ = graph_builder.build_graph(classified_edges, all_obs_ids)

    clusterer = Clusterer()
    clusters, _ = clusterer.cluster_observations(graph, all_obs_ids)

    # Stage 9: Attach candidate mentions to clusters
    attacher = CandidateAttacher()
    attachments, attachment_report = attacher.attach_candidates(
        classified_edges, clusters, observations
    )

    print(f"Candidate Attachment Report:")
    print(f"  Total candidate edges: {attachment_report.total_candidate_edges}")
    print(f"  Attachments created: {attachment_report.total_attachments_created}")
    print(f"  Processing time: {attachment_report.processing_time_sec:.3f}s")

    print(f"\n  Confidence Distribution:")
    print(f"    High-confidence (0.75-0.80): {attachment_report.high_confidence_attachments}")
    print(f"    Medium-confidence (0.65-0.75): {attachment_report.medium_confidence_attachments}")
    print(f"    Low-confidence (0.60-0.65): {attachment_report.low_confidence_attachments}")

    print(f"\n  Conflict Analysis:")
    print(f"    Attachments with conflicts: {attachment_report.attachments_with_conflicts}")
    print(f"    Total conflicts detected: {attachment_report.total_conflicts_detected}")
    print(f"    High severity conflicts: {attachment_report.high_severity_conflicts}")
    print(f"    Medium severity conflicts: {attachment_report.medium_severity_conflicts}")
    print(f"    Low severity conflicts: {attachment_report.low_severity_conflicts}")

    if attachments:
        print(f"\nCandidate Attachments (sorted by similarity):")
        for i, attachment in enumerate(attachments[:5]):  # Top 5
            print(f"\n  {i+1}. Observation {attachment.obs_id} -> Cluster {attachment.cluster_id}")
            print(
                f"     Similarity: {attachment.similarity_score:.3f}, "
                f"Confidence: {attachment.confidence}"
            )
            print(f"     Reasons: {', '.join(attachment.reasons)}")
            
            if attachment.conflict_flags:
                print(f"     Conflicts:")
                for flag in attachment.conflict_flags:
                    print(
                        f"       - [{flag.severity.upper()}] {flag.conflict_type}: "
                        f"{flag.description}"
                    )
                    print(f"         Recommendation: {flag.recommendation}")


def example_conflict_handler():
    """Example: Detecting conflicts in entity resolution results."""
    print("\n" + "=" * 70)
    print("EXAMPLE 13: CONFLICT DETECTION")
    print("=" * 70)

    # Create normalized observations
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")

    obs_1 = NormalizedObservation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect approaching ATM",
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:15:30+00:00"),
        timestamp_offset_sec=930,
        content_normalized="suspect approaching atm",
    )

    obs_2 = NormalizedObservation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM booth interior",
        content="Voice near machine",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:14:30+00:00"),
        timestamp_offset_sec=870,
        content_normalized="voice near machine",
    )

    obs_3 = NormalizedObservation(
        obs_id="O3",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect operating keypad",
        timestamp="2024-01-15T10:16:00Z",
        time_offset=210,
        confidence=0.92,
        noise_tags=["partial_occlusion"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:16:00+00:00"),
        timestamp_offset_sec=960,
        content_normalized="suspect operating keypad",
    )

    obs_4 = NormalizedObservation(
        obs_id="O4",
        entity="Person_08",
        role="witness",
        modality=Modality.VIDEO,
        source="camera_2",
        location="ATM street frontage",
        content="Witness observing ATM",
        timestamp="2024-01-15T14:30:00Z",
        time_offset=16200,
        confidence=0.88,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T14:30:00+00:00"),
        timestamp_offset_sec=16200,
        content_normalized="witness observing atm",
    )

    observations = [obs_1, obs_2, obs_3, obs_4]
    all_obs_ids = [obs.obs_id for obs in observations]

    # Stages 3-8: Generate candidates through clustering
    blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
    candidates, _ = blocker.generate_candidates(observations)

    features_computer = Features()
    feature_vectors, _ = features_computer.compute_features(candidates, observations)

    scorer = Scorer()
    scored_pairs, _ = scorer.score_pairs(feature_vectors)

    classifier = EdgeClassifier()
    classified_edges, _ = classifier.classify_edges(scored_pairs)

    graph_builder = GraphBuilder()
    graph, _ = graph_builder.build_graph(classified_edges, all_obs_ids)

    clusterer = Clusterer()
    clusters, _ = clusterer.cluster_observations(graph, all_obs_ids)

    # Stage 10: Detect conflicts
    conflict_handler = ConflictHandler()
    conflicts, conflict_report = conflict_handler.detect_conflicts(clusters, observations)

    print(f"Conflict Detection Report:")
    print(f"  Total clusters analyzed: {conflict_report.total_clusters}")
    print(f"  Total observations: {conflict_report.total_observations}")
    print(f"  Total conflicts detected: {conflict_report.total_conflicts_detected}")
    print(f"  Clusters with conflicts: {conflict_report.clusters_with_conflicts}")
    print(f"  Observations affected: {conflict_report.observations_affected}")
    print(f"  Processing time: {conflict_report.processing_time_sec:.3f}s")

    print(f"\n  Conflict Severity Distribution:")
    print(f"    High: {conflict_report.high_severity_conflicts}")
    print(f"    Medium: {conflict_report.medium_severity_conflicts}")
    print(f"    Low: {conflict_report.low_severity_conflicts}")

    print(f"\n  Conflict Types:")
    print(f"    Role contradictions: {conflict_report.role_contradiction_count}")
    print(f"    Temporal conflicts: {conflict_report.temporal_conflict_count}")
    print(f"    Low-confidence clusters: {conflict_report.low_confidence_count}")
    print(f"    Suspicious sizes: {conflict_report.suspicious_size_count}")

    if conflicts:
        print(f"\nDetected Conflicts (sorted by severity):")
        for i, conflict in enumerate(conflicts[:5]):  # Top 5
            print(f"\n  {i+1}. [{conflict.severity.value.upper()}] {conflict.conflict_id}")
            print(f"      Type: {conflict.conflict_type.value}")
            print(f"      Description: {conflict.description}")
            print(f"      Affected clusters: {', '.join(conflict.affected_clusters)}")
            print(f"      Affected observations: {', '.join(conflict.affected_observations)}")
            print(f"      Conservative action: {conflict.conservative_action}")
    else:
        print(f"\n[OK] No conflicts detected - clustering appears clean!")


def example_labeler():
    """Example: Entity labeling and ID assignment."""
    print("\n" + "=" * 70)
    print("EXAMPLE 14: ENTITY LABELING")
    print("=" * 70)

    # Create normalized observations
    base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")

    obs_1 = NormalizedObservation(
        obs_id="O1",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect approaching ATM",
        timestamp="2024-01-15T10:15:30Z",
        time_offset=180,
        confidence=0.95,
        noise_tags=["blur"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:15:30+00:00"),
        timestamp_offset_sec=930,
        content_normalized="suspect approaching atm",
    )

    obs_2 = NormalizedObservation(
        obs_id="O2",
        entity="Speaker_A",
        role="suspect",
        modality=Modality.AUDIO,
        source="mic_booth",
        location="ATM booth interior",
        content="Voice near machine",
        timestamp="2024-01-15T10:14:30Z",
        time_offset=120,
        confidence=0.89,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:14:30+00:00"),
        timestamp_offset_sec=870,
        content_normalized="voice near machine",
    )

    obs_3 = NormalizedObservation(
        obs_id="O3",
        entity="Person_05",
        role="suspect",
        modality=Modality.VIDEO,
        source="camera_1",
        location="ATM booth interior",
        content="Suspect operating keypad",
        timestamp="2024-01-15T10:16:00Z",
        time_offset=210,
        confidence=0.92,
        noise_tags=["partial_occlusion"],
        timestamp_dt=datetime.fromisoformat("2024-01-15T10:16:00+00:00"),
        timestamp_offset_sec=960,
        content_normalized="suspect operating keypad",
    )

    obs_4 = NormalizedObservation(
        obs_id="O4",
        entity="Person_08",
        role="witness",
        modality=Modality.VIDEO,
        source="camera_2",
        location="ATM street frontage",
        content="Witness observing ATM",
        timestamp="2024-01-15T14:30:00Z",
        time_offset=16200,
        confidence=0.88,
        noise_tags=[],
        timestamp_dt=datetime.fromisoformat("2024-01-15T14:30:00+00:00"),
        timestamp_offset_sec=16200,
        content_normalized="witness observing atm",
    )

    observations = [obs_1, obs_2, obs_3, obs_4]
    all_obs_ids = [obs.obs_id for obs in observations]

    # Stages 3-8: Full pipeline through clustering
    blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
    candidates, _ = blocker.generate_candidates(observations)

    features_computer = Features()
    feature_vectors, _ = features_computer.compute_features(candidates, observations)

    scorer = Scorer()
    scored_pairs, _ = scorer.score_pairs(feature_vectors)

    classifier = EdgeClassifier()
    classified_edges, classification_report = classifier.classify_edges(scored_pairs)

    graph_builder = GraphBuilder()
    graph, _ = graph_builder.build_graph(classified_edges, all_obs_ids)

    clusterer = Clusterer()
    clusters, _ = clusterer.cluster_observations(graph, all_obs_ids)

    # Stage 11: Label entities
    labeler = Labeler()
    canonical_entities, labeling_report = labeler.label_entities(
        clusters,
        observations,
        confirmed_edge_count=classification_report.confirmed_edges,
        candidate_edge_count=classification_report.candidate_edges,
    )

    print(f"Entity Labeling Report:")
    print(f"  Total clusters: {labeling_report.total_clusters}")
    print(f"  Canonical entities created: {labeling_report.total_entities_created}")
    print(f"  Processing time: {labeling_report.processing_time_sec:.3f}s")

    print(f"\n  Entity Size Distribution:")
    print(f"    Singletons (size 1): {labeling_report.singleton_entities}")
    print(f"    Pairs (size 2): {labeling_report.pair_entities}")
    print(f"    Triplets (size 3): {labeling_report.triplet_entities}")
    print(f"    Large clusters (size >= 4): {labeling_report.large_entities}")

    print(f"\n  Confidence Distribution:")
    print(f"    High (>=0.80): {labeling_report.high_confidence_entities}")
    print(f"    Medium (0.60-0.80): {labeling_report.medium_confidence_entities}")
    print(f"    Low (<0.60): {labeling_report.low_confidence_entities}")

    print(f"\n  Modality Coverage:")
    print(f"    Video entities: {labeling_report.video_entities}")
    print(f"    Audio entities: {labeling_report.audio_entities}")
    print(f"    Text entities: {labeling_report.text_entities}")
    print(f"    Multimodal entities: {labeling_report.multimodal_entities}")

    print(f"\n  Canonical Entities:")
    for i, entity in enumerate(canonical_entities[:5]):  # Top 5
        print(f"\n  {i+1}. {entity.entity_id}")
        print(f"     Aliases: {', '.join(sorted(entity.aliases))}")
        print(f"     Mentions: {entity.total_mention_count}")
        print(f"     Confidence: {entity.confidence_score:.3f}")
        print(f"     Modalities: {', '.join(sorted(entity.modalities))}")
        print(f"     Roles: {', '.join(sorted(entity.roles))}")
        print(f"     Locations: {', '.join(sorted(entity.locations))}")


def example_resolver():
    """Example: Complete end-to-end resolution pipeline."""
    print("\n" + "=" * 70)
    print("EXAMPLE 15: COMPLETE PIPELINE (RESOLVER)")
    print("=" * 70)

    # Create a complete case input
    case_input = {
        "case_id": "CASE_20250315_COMPLETE",
        "observations": [
            {
                "obs_id": "O1",
                "entity": "Person_05",
                "role": "suspect",
                "modality": "video",
                "source": "camera_1",
                "location": "ATM booth interior",
                "content": "Suspect approaching ATM booth",
                "timestamp": "2024-01-15T10:15:30Z",
                "time_offset": 180,
                "confidence": 0.95,
                "noise_tags": ["blur"],
            },
            {
                "obs_id": "O2",
                "entity": "Speaker_A",
                "role": "suspect",
                "modality": "audio",
                "source": "mic_booth",
                "location": "ATM booth interior",
                "content": "Voice near machine",
                "timestamp": "2024-01-15T10:14:30Z",
                "time_offset": 120,
                "confidence": 0.89,
                "noise_tags": [],
            },
            {
                "obs_id": "O3",
                "entity": "Person_05",
                "role": "suspect",
                "modality": "video",
                "source": "camera_1",
                "location": "ATM booth interior",
                "content": "Suspect operating keypad",
                "timestamp": "2024-01-15T10:16:00Z",
                "time_offset": 210,
                "confidence": 0.92,
                "noise_tags": ["partial_occlusion"],
            },
            {
                "obs_id": "O4",
                "entity": "Person_08",
                "role": "witness",
                "modality": "video",
                "source": "camera_2",
                "location": "ATM street frontage",
                "content": "Witness observing ATM",
                "timestamp": "2024-01-15T14:30:00Z",
                "time_offset": 16200,
                "confidence": 0.88,
                "noise_tags": [],
            },
        ],
    }

    # Initialize resolver with custom configuration
    config = PipelineConfiguration(
        check_duplicates=True,
        temporal_window_sec=300,
        max_temporal_gap_sec=3600,
        confirmed_threshold=0.80,
        candidate_threshold_low=0.60,
        candidate_threshold_high=0.80,
    )

    resolver = Resolver(config=config)

    # Run complete resolution
    result = resolver.resolve_case_dict(case_input)

    print(f"Resolution Status: {result.status.upper()}")
    if result.error_message:
        print(f"Error: {result.error_message}")
        return

    print(f"\n  Case ID: {result.case_id}")
    print(f"  Total Processing Time: {result.total_processing_time_sec:.3f}s")

    print(f"\n  PIPELINE RESULTS:")
    print(f"  ├─ Intake: {result.intake_report.valid_observations} valid observations")
    print(f"  ├─ Blocking: {result.blocking_report.pairs_after_hard_reject} candidate pairs")
    print(f"  ├─ Features: {result.features_report.total_pairs} pairs evaluated")
    print(f"  ├─ Scoring: avg similarity {result.scoring_report.avg_similarity_score:.3f}")
    print(f"  ├─ Classification: {result.classification_report.confirmed_edges} confirmed, {result.classification_report.candidate_edges} candidate")
    print(f"  ├─ Graph: {result.graph_report.nodes_count} nodes, {result.graph_report.edges_count} edges")
    print(f"  ├─ Clustering: {result.clustering_report.total_clusters} clusters")
    print(f"  ├─ Attachment: {result.attachment_report.total_attachments_created} attachments")
    print(f"  ├─ Conflicts: {result.conflict_report.total_conflicts_detected} conflicts detected")
    print(f"  └─ Labeling: {result.labeling_report.total_entities_created} canonical entities")

    print(f"\n  STAGE TIMINGS:")
    for stage, timing in sorted(result.stage_timings.items()):
        print(f"    {stage}: {timing:.4f}s")

    print(f"\n  CANONICAL ENTITIES ({result.entity_count}):")
    for i, entity in enumerate(result.canonical_entities[:5]):  # Show top 5
        print(f"\n    {i+1}. {entity.entity_id}")
        print(f"       Primary Alias: {entity.primary_alias}")
        print(f"       All Aliases: {', '.join(sorted(entity.aliases))}")
        print(f"       Mentions: {entity.total_mention_count}")
        print(f"       Confidence: {entity.confidence_score:.3f}")
        print(f"       Modalities: {', '.join(sorted(entity.modalities))}")
        print(f"       Roles: {', '.join(sorted(entity.roles))}")

    if result.conflicts:
        print(f"\n  DETECTED CONFLICTS: {len(result.conflicts)}")
        for i, conflict in enumerate(result.conflicts[:3]):  # Top 3
            print(f"    {i+1}. [{conflict.severity.value.upper()}] {conflict.conflict_type.value}")
            print(f"       {conflict.description}")


if __name__ == "__main__":
    example_intake_validation()
    example_alias_normalization()
    example_observation_normalization()
    example_invalid_input()
    example_duplicate_detection()
    example_blocking()
    example_features()
    example_scorer()
    example_edge_classifier()
    example_graph_builder()
    example_clusterer()
    example_candidate_attacher()
    example_conflict_handler()
    example_labeler()
    example_resolver()

    print("\n" + "=" * 70)
    print("[OK] All examples completed successfully!")
    print("=" * 70)
