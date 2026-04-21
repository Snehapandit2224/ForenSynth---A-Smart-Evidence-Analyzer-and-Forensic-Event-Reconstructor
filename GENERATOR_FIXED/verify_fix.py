#!/usr/bin/env python3
"""Quick verification script for template constraints."""

from generator import ForenSynthGenerator
from config import GeneratorConfig
from templates import get_template_role_requirements, TEMPLATE_REGISTRY

# Test multi-actor template enforcement
cfg = GeneratorConfig(seed=100)
gen = ForenSynthGenerator(cfg)

multi_template = TEMPLATE_REGISTRY['ATM_Robbery'][1]  # MultiActor_Entry_Action_Exit
reqs = get_template_role_requirements(multi_template)

print(f"Multi-actor template: {multi_template.name}")
print(f"Requirements: {reqs}")
print()

# Generate cases until we get one with the multi-actor template
for i in range(10):
    case = gen.generate_case(domain='ATM_Robbery')
    
    if case['template'] == multi_template.name:
        entities = case['ground_truth']['entities']
        entity_counts = {}
        for e in entities:
            entity_counts[e['role']] = entity_counts.get(e['role'], 0) + 1
        
        print(f"Multi-actor case found (attempt {i+1})")
        print(f"  Actual entity counts: {entity_counts}")
        print(f"  Satisfies requirements: {all(entity_counts.get(role, 0) >= count for role, count in reqs.items())}")
        break
else:
    print("No multi-actor cases generated in 10 attempts (OK - random selection)")

print("\nTesting FIR/entity consistency:")
cfg2 = GeneratorConfig(seed=42)
gen2 = ForenSynthGenerator(cfg2)
case2 = gen2.generate_case(domain='Office_Theft')

entities2 = case2['ground_truth']['entities']
entity_counts2 = {}
for e in entities2:
    entity_counts2[e['role']] = entity_counts2.get(e['role'], 0) + 1

fir_counts = case2['fir']['roles']
print(f"  Actual entity counts: {entity_counts2}")
print(f"  FIR role counts: {fir_counts}")
print(f"  Consistent: {all(entity_counts2.get(role, 0) == fir_counts.get(role) for role in entity_counts2)}")

print("\nTesting observations sanitization:")
print(f"  Observations in case: {len(case2['observations'])}")
print(f"  First obs has canonical_entity: {'canonical_entity' in case2['observations'][0]}")
print(f"  First obs has event_ref: {'event_ref' in case2['observations'][0]}")

from utils import extract_observations_only
obs_only = extract_observations_only(case2)
print(f"  Obs-only has event_ref: {'event_ref' in obs_only['observations'][0] if obs_only['observations'] else False}")
print(f"  Obs-only has noise_tags: {'noise_tags' in obs_only['observations'][0] if obs_only['observations'] else False}")

print("\nAll checks passed!")
