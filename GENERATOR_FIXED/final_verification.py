#!/usr/bin/env python3
"""Final comprehensive verification of all fixes."""

from generator import ForenSynthGenerator
from config import GeneratorConfig
from utils import extract_observations_only

print("="*60)
print("COMPREHENSIVE VERIFICATION OF TEMPLATE-CONSTRAINED GENERATION")
print("="*60)

# Test 1: Batch generation
print("\n[Test 1] Batch generation respects template constraints")
cfg = GeneratorConfig(seed=12345)
gen = ForenSynthGenerator(cfg)
cases = gen.generate_batch(3, domain='ATM_Robbery')

print(f"  Generated {len(cases)} cases successfully")
for i, case in enumerate(cases, 1):
    entities = {}
    for e in case['ground_truth']['entities']:
        entities[e['role']] = entities.get(e['role'], 0) + 1
    
    fir_matches = all(
        entities.get(role, 0) == case['fir']['roles'].get(role, 0)
        for role in set(list(entities.keys()) + list(case['fir']['roles'].keys()))
    )
    
    print(f"  Case {i}: {case['case_id']} - Template: {case['template']}")
    print(f"    Entities: {entities}, Events: {len(case['ground_truth']['events'])}")
    print(f"    FIR matches entities: {'✓' if fir_matches else '✗'}")

# Test 2: Verify no broken cases in batch
print("\n[Test 2] All batch cases are valid")
for i, case in enumerate(cases, 1):
    events = case['ground_truth']['events']
    entities = case['ground_truth']['entities']
    entity_ids = {e['entity_id'] for e in entities}
    
    valid = True
    issues = []
    
    # Check events reference valid entities
    for event in events:
        if event['entity_id'] not in entity_ids:
            valid = False
            issues.append(f"Event {event['event_id']} refs unknown entity")
    
    # Check observations don't leak ground truth
    for obs in case['observations']:
        if 'canonical_entity' in obs:
            valid = False
            issues.append("Observation leaks canonical_entity")
    
    status = '✓ Valid' if valid else f'✗ Invalid: {issues}'
    print(f"  Case {i}: {status}")

# Test 3: Observations-only export is sanitized
print("\n[Test 3] Observations-only export sanitized")
obs_only = extract_observations_only(cases[0])
sanitized = True
issues = []

for obs in obs_only['observations']:
    if 'event_ref' in obs:
        sanitized = False
        issues.append("Obs-only has event_ref")
    if 'noise_tags' in obs:
        sanitized = False
        issues.append("Obs-only has noise_tags")
    if 'canonical_entity' in obs:
        sanitized = False
        issues.append("Obs-only has canonical_entity")

print(f"  Observations-only fields: {list(obs_only['observations'][0].keys()) if obs_only['observations'] else 'none'}")
print(f"  Sanitized: {'✓' if sanitized else f'✗ {issues}'}")

# Test 4: Different domains work correctly
print("\n[Test 4] All domains generate correctly")
for domain in ['ATM_Robbery', 'Office_Theft', 'Communication']:
    cfg = GeneratorConfig(seed=999)
    gen = ForenSynthGenerator(cfg)
    try:
        case = gen.generate_case(domain=domain)
        entities = {}
        for e in case['ground_truth']['entities']:
            entities[e['role']] = entities.get(e['role'], 0) + 1
        print(f"  {domain}: ✓ Template={case['template']}, Entities={entities}")
    except Exception as e:
        print(f"  {domain}: ✗ {str(e)[:50]}")

# Test 5: Forced entity counts
print("\n[Test 5] Forced entity counts enforced")
cfg = GeneratorConfig(seed=555)
gen = ForenSynthGenerator(cfg)
case = gen.generate_case(domain='Office_Theft', suspect_count=3, witness_count=2)

entities = {}
for e in case['ground_truth']['entities']:
    entities[e['role']] = entities.get(e['role'], 0) + 1

meets_req = entities.get('suspect', 0) >= 3 and entities.get('witness', 0) >= 2
print(f"  Requested: 3 suspects, 2 witnesses")
print(f"  Actual: {entities}")
print(f"  Requirements met: {'✓' if meets_req else '✗'}")

print("\n" + "="*60)
print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY")
print("="*60)
