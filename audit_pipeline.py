#!/usr/bin/env python
"""Comprehensive audit of the entity resolution pipeline to identify where confirmed edges are lost."""

import json
from datetime import datetime
from entity_resolution.services.entity_mapping.normalizer import Normalizer
from entity_resolution.services.entity_mapping.blocker import Blocker
from entity_resolution.services.entity_mapping.features import Features
from entity_resolution.services.entity_mapping.scorer import Scorer, ScoringWeights
from entity_resolution.services.entity_mapping.edge_classifier import EdgeClassifier, ClassificationThresholds
from entity_resolution.services.entity_mapping.graph_builder import GraphBuilder
from entity_resolution.services.entity_mapping.clusterer import Clusterer
from entity_resolution.services.entity_mapping.intake import IntakeValidator

# Create a simple test case with TWO observations that should DEFINITELY merge
test_case = {
    "case_id": "AUDIT_TEST_001",
    "observations": [
        {
            "obs_id": "A1",
            "entity": "Person_14",
            "role": "suspect",
            "modality": "video",
            "source": "camera_1",
            "location": "ATM entrance",
            "content": "Person at ATM",
            "timestamp": "2024-01-15T10:00:00",
            "time_offset": 0,
            "confidence": 0.95
        },
        {
            "obs_id": "A2",
            "entity": "Person_14",  # IDENTICAL alias
            "role": "suspect",
            "modality": "video",
            "source": "camera_1",
            "location": "ATM entrance",
            "content": "Person at ATM",  # IDENTICAL content
            "timestamp": "2024-01-15T10:00:02",  # 2 seconds later
            "time_offset": 2,
            "confidence": 0.95
        }
    ]
}

print("=" * 90)
print("ENTITY RESOLUTION AUDIT: Person_14 should merge")
print("=" * 90)

# STAGE 1: INTAKE
print("\n[STAGE 1] INTAKE VALIDATION")
print("-" * 90)
intake_validator = IntakeValidator()
observations, intake_report = intake_validator.validate_case(test_case)
print(f"✓ Valid observations: {intake_report.valid_observations}")
for obs in observations:
    print(f"  - {obs.obs_id}: {obs.entity}")

# STAGE 2: NORMALIZATION
print("\n[STAGE 2] NORMALIZATION")
print("-" * 90)
normalizer = Normalizer()
normalized_obs, norm_report = normalizer.normalize_observations(observations)
print(f"✓ Normalized: {norm_report.successfully_normalized}")

# STAGE 3: BLOCKING
print("\n[STAGE 3] BLOCKING")
print("-" * 90)
blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
candidate_pairs, blocking_report = blocker.generate_candidates(normalized_obs)
print(f"✓ Candidate pairs: {blocking_report.total_candidate_pairs}")
print(f"✓ After hard reject: {blocking_report.pairs_after_hard_reject}")
if candidate_pairs:
    for pair in candidate_pairs:
        print(f"  - {pair.obs_id_1} <-> {pair.obs_id_2} (priority={pair.priority:.3f})")

# STAGE 4: FEATURES
print("\n[STAGE 4] FEATURES")
print("-" * 90)
features_computer = Features()
feature_vectors, features_report = features_computer.compute_features(candidate_pairs, normalized_obs)
print(f"✓ Feature vectors: {features_report.total_pairs}")
print(f"✓ Avg combined score: {features_report.avg_combined_score:.3f}")
for fv in feature_vectors:
    print(f"\n  Pair: {fv.obs_id_1} <-> {fv.obs_id_2}")
    print(f"    Alias Identity: {fv.alias_identity_score:.3f}")
    print(f"    Temporal: {fv.temporal_score:.3f}")
    print(f"    Location: {fv.location_score:.3f}")
    print(f"    Context: {fv.context_score:.3f}")
    print(f"    Interaction: {fv.interaction_score:.3f}")
    print(f"    Lexical: {fv.lexical_score:.3f}")
    print(f"    Modality: {fv.modality_compatibility_score:.3f}")
    print(f"    >>> COMBINED SCORE (in features): {fv.combined_score:.3f}")

# STAGE 5: SCORING
print("\n[STAGE 5] SCORING")
print("-" * 90)
scorer = Scorer(weights=ScoringWeights())
print(f"  Scorer weights: {scorer.weights.to_dict()}")
scored_pairs, scoring_report = scorer.score_pairs(feature_vectors)
print(f"✓ Scored pairs: {scoring_report.total_pairs}")
print(f"✓ Avg similarity score: {scoring_report.avg_similarity_score:.3f}")
for sp in scored_pairs:
    print(f"\n  {sp.obs_id_1} <-> {sp.obs_id_2}")
    print(f"    Similarity Score: {sp.similarity_score:.3f}")
    print(f"    Components: {sp.score_components}")
    print(f"    >>> BUG CHECK: Does Scorer use alias_identity? {('alias_identity' in sp.score_components)}")

# STAGE 6: EDGE CLASSIFICATION
print("\n[STAGE 6] EDGE CLASSIFICATION")
print("-" * 90)
thresholds = ClassificationThresholds(confirmed=0.65, candidate_low=0.50)
print(f"  Confirmed threshold: {thresholds.confirmed}")
print(f"  Candidate low threshold: {thresholds.candidate_low}")
classifier = EdgeClassifier(thresholds=thresholds)
classified_edges, classification_report = classifier.classify_edges(scored_pairs)
print(f"✓ Confirmed edges: {classification_report.confirmed_edges}")
print(f"✓ Candidate edges: {classification_report.candidate_edges}")
print(f"✓ Rejected edges: {classification_report.rejected_edges}")
for edge in classified_edges:
    print(f"  {edge.obs_id_1} <-> {edge.obs_id_2}: {edge.similarity_score:.3f} [{edge.classification.value}]")

# STAGE 7: GRAPH BUILDING
print("\n[STAGE 7] GRAPH BUILDING")
print("-" * 90)
all_obs_ids = [obs.obs_id for obs in normalized_obs]
graph_builder = GraphBuilder()
graph, graph_report = graph_builder.build_graph(classified_edges, all_obs_ids)
print(f"✓ Confirmed edges added to graph: {graph_report.total_confirmed_edges}")
print(f"✓ Candidate edges (not added): {graph_report.total_candidate_edges}")
print(f"✓ Graph edges: {graph_report.edges_count}")
print(f"✓ Isolated nodes: {graph_report.isolated_nodes}")

# STAGE 8: CLUSTERING
print("\n[STAGE 8] CLUSTERING")
print("-" * 90)
clusterer = Clusterer()
clusters, clustering_report = clusterer.cluster_observations(graph, all_obs_ids)
print(f"✓ Total clusters: {clustering_report.total_clusters}")
print(f"✓ Singleton clusters: {clustering_report.singleton_clusters}")
for cluster in clusters:
    print(f"  Cluster {cluster.cluster_id}: {cluster.obs_ids} (size={cluster.size})")

# FINAL VERDICT
print("\n" + "=" * 90)
print("AUDIT RESULT")
print("=" * 90)
if scoring_report.total_pairs > 0 and 'alias_identity' not in scored_pairs[0].score_components:
    print("\n🔴 CRITICAL BUG FOUND:")
    print("   Scorer.compute_similarity_score() does NOT include alias_identity!")
    print("   Features.py computes it, but Scorer.py ignores it.")
    print("   This causes alias matches to score too low for confirmation.")
    print("\n   FIX: Add 'alias_identity' to ScoringWeights and Scorer.compute_similarity_score()")
    
print(f"\n📊 RESULTS:")
print(f"   Expected: 1 cluster with both observations")
print(f"   Actual: {clustering_report.total_clusters} clusters")
print(f"   Confirmed edges: {classification_report.confirmed_edges} (need > 0)")
if clustering_report.total_clusters == 1 and clustering_report.singleton_clusters == 0:
    print("\n✓ SUCCESS: Observations merged!")
else:
    print("\n✗ FAILURE: Observations NOT merged")
