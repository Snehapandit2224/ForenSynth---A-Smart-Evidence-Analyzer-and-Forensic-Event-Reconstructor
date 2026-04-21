#!/usr/bin/env python
"""Debug script to test the features computation directly."""

import json
from datetime import datetime
from entity_resolution.services.entity_mapping.normalizer import Normalizer
from entity_resolution.services.entity_mapping.blocker import Blocker
from entity_resolution.services.entity_mapping.features import Features
from entity_resolution.services.entity_mapping.scorer import Scorer, ScoringWeights
from entity_resolution.services.entity_mapping.edge_classifier import EdgeClassifier, ClassificationThresholds
from entity_resolution.services.entity_mapping.intake import IntakeValidator

# Load test case
with open('GENERATOR_FIXED/cases/merged_case_file.json', 'r') as f:
    test_case = json.load(f)

print("=" * 80)
print("STEP 1: INTAKE VALIDATION")
print("=" * 80)
intake_validator = IntakeValidator()
observations, intake_report = intake_validator.validate_case(test_case)
print(f"Valid observations: {intake_report.valid_observations}")
for obs in observations:
    print(f"  {obs.obs_id}: {obs.entity} | {obs.location} | {obs.content}")

print("\n" + "=" * 80)
print("STEP 2: NORMALIZATION")
print("=" * 80)
normalizer = Normalizer()
normalized_obs, norm_report = normalizer.normalize_observations(observations)
print(f"Successfully normalized: {norm_report.successfully_normalized}")
for obs in normalized_obs:
    print(f"  {obs.obs_id}: {obs.entity} | {obs.location} | timestamp_dt={obs.timestamp_dt}")

print("\n" + "=" * 80)
print("STEP 3: BLOCKING")
print("=" * 80)
blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
candidate_pairs, blocking_report = blocker.generate_candidates(normalized_obs)
print(f"Candidate pairs: {blocking_report.total_candidate_pairs}")
for pair in candidate_pairs:
    print(f"  {pair.obs_id_1} <-> {pair.obs_id_2} (priority={pair.priority:.3f})")

print("\n" + "=" * 80)
print("STEP 4: FEATURES")
print("=" * 80)
features_computer = Features()
feature_vectors, features_report = features_computer.compute_features(candidate_pairs, normalized_obs)
print(f"Feature vectors: {features_report.total_pairs}")
print(f"Avg combined score: {features_report.avg_combined_score:.3f}")
for fv in feature_vectors:
    print(f"\n  Pair: {fv.obs_id_1} <-> {fv.obs_id_2}")
    print(f"    Temporal: {fv.temporal_score:.3f}")
    print(f"    Location: {fv.location_score:.3f}")
    print(f"    Context: {fv.context_score:.3f}")
    print(f"    Interaction: {fv.interaction_score:.3f}")
    print(f"    Lexical: {fv.lexical_score:.3f}")
    print(f"    Modality: {fv.modality_compatibility_score:.3f}")
    print(f"    Alias Identity: {fv.alias_identity_score:.3f}")
    print(f"    Combined Score: {fv.combined_score:.3f}")
    print(f"    Rationale: {fv.rationale}")

print("\n" + "=" * 80)
print("STEP 5: SCORING")
print("=" * 80)
scorer = Scorer(weights=ScoringWeights())
scored_pairs, scoring_report = scorer.score_pairs(feature_vectors)
print(f"Scored pairs: {scoring_report.total_pairs}")
for sp in scored_pairs:
    print(f"  {sp.obs_id_1} <-> {sp.obs_id_2}: {sp.similarity_score:.3f}")

print("\n" + "=" * 80)
print("STEP 6: EDGE CLASSIFICATION")
print("=" * 80)
thresholds = ClassificationThresholds(confirmed=0.70, candidate_low=0.50)
print(f"Thresholds: confirmed={thresholds.confirmed}, candidate_low={thresholds.candidate_low}")
classifier = EdgeClassifier(thresholds=thresholds)
classified_edges, classification_report = classifier.classify_edges(scored_pairs)
print(f"Confirmed: {classification_report.confirmed_edges}")
print(f"Candidate: {classification_report.candidate_edges}")
print(f"Rejected: {classification_report.rejected_edges}")
for edge in classified_edges:
    print(f"  {edge.obs_id_1} <-> {edge.obs_id_2}: {edge.similarity_score:.3f} [{edge.classification.value}]")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Test case expects: entity_count=1, status=success")
print(f"Scores computed: {len(feature_vectors)} pairs with avg score {features_report.avg_combined_score:.3f}")
print(f"Edges classified: confirmed={classification_report.confirmed_edges}, candidate={classification_report.candidate_edges}")
