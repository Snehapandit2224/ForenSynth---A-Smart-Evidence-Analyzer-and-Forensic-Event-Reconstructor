#!/usr/bin/env python
"""Debug script to test the blocker output."""

import json
from datetime import datetime
from entity_resolution.services.entity_mapping.normalizer import Normalizer
from entity_resolution.services.entity_mapping.blocker import Blocker
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
    print(f"  ID: {obs.obs_id}, Entity: {obs.entity}")

print("\n" + "=" * 80)
print("STEP 2: NORMALIZATION")
print("=" * 80)
normalizer = Normalizer()
normalized_obs, norm_report = normalizer.normalize_observations(observations)
print(f"Successfully normalized: {norm_report.successfully_normalized}")
for obs in normalized_obs:
    print(f"  ID: {obs.obs_id}, Entity: {obs.entity}")

print("\n" + "=" * 80)
print("STEP 3: BLOCKING - DETAILED")
print("=" * 80)
blocker = Blocker(temporal_window_sec=300, max_temporal_gap_sec=3600)
candidate_pairs, blocking_report = blocker.generate_candidates(normalized_obs)
print(f"Total candidate pairs: {blocking_report.total_candidate_pairs}")
print(f"Pairs after hard reject: {blocking_report.pairs_after_hard_reject}")
print(f"Candidate pairs list length: {len(candidate_pairs)}")
print(f"Candidate pairs:")
for i, pair in enumerate(candidate_pairs):
    print(f"  [{i}] obs_id_1={pair.obs_id_1}, obs_id_2={pair.obs_id_2}, priority={pair.priority:.3f}")
    
# Let's also check normalized_obs list
print("\nNormalized observations:")
for i, obs in enumerate(normalized_obs):
    print(f"  [{i}] obs_id={obs.obs_id}, type={type(obs)}")
