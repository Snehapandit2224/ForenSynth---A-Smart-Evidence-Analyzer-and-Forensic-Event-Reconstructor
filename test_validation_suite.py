#!/usr/bin/env python
"""
Comprehensive validation test for entity resolution pipeline fixes.

This test verifies that all bugs have been fixed and entities merge correctly
for various real-world scenarios.
"""

import json
from entity_resolution.services.entity_mapping.resolver import Resolver

# Test cases covering various merging scenarios
TEST_CASES = [
    {
        "name": "Perfect Match - Identical Alias",
        "case": {
            "case_id": "TEST_001",
            "observations": [
                {
                    "obs_id": "O1",
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
                    "obs_id": "O2",
                    "entity": "Person_14",
                    "role": "suspect",
                    "modality": "video",
                    "source": "camera_1",
                    "location": "ATM entrance",
                    "content": "Person at ATM",
                    "timestamp": "2024-01-15T10:00:02",
                    "time_offset": 2,
                    "confidence": 0.95
                }
            ]
        },
        "expected_entity_count": 1,
        "expected_mentions": 2
    },
    {
        "name": "Same Alias Different Time",
        "case": {
            "case_id": "TEST_002",
            "observations": [
                {
                    "obs_id": "O1",
                    "entity": "Speaker_A",
                    "role": "witness",
                    "modality": "audio",
                    "source": "microphone_1",
                    "location": "Interview room",
                    "content": "Speaking",
                    "timestamp": "2024-01-15T11:00:00",
                    "time_offset": 0,
                    "confidence": 0.90
                },
                {
                    "obs_id": "O2",
                    "entity": "Speaker_A",
                    "role": "witness",
                    "modality": "audio",
                    "source": "microphone_2",
                    "location": "Interview room",
                    "content": "Speaking",
                    "timestamp": "2024-01-15T11:05:00",
                    "time_offset": 300,
                    "confidence": 0.90
                }
            ]
        },
        "expected_entity_count": 1,
        "expected_mentions": 2
    },
    {
        "name": "Single Observation",
        "case": {
            "case_id": "TEST_003",
            "observations": [
                {
                    "obs_id": "O1",
                    "entity": "Person_99",
                    "role": "suspect",
                    "modality": "text",
                    "source": "report_1",
                    "location": "Police station",
                    "content": "Arrest report",
                    "timestamp": "2024-01-15T12:00:00",
                    "time_offset": 0,
                    "confidence": 0.85
                }
            ]
        },
        "expected_entity_count": 1,
        "expected_mentions": 1
    },
    {
        "name": "Three Observations Same Entity",
        "case": {
            "case_id": "TEST_004",
            "observations": [
                {
                    "obs_id": "O1",
                    "entity": "Person_05",
                    "role": "suspect",
                    "modality": "video",
                    "source": "camera_1",
                    "location": "Parking garage",
                    "content": "Individual in parking area",
                    "timestamp": "2024-01-15T14:00:00",
                    "time_offset": 0,
                    "confidence": 0.92
                },
                {
                    "obs_id": "O2",
                    "entity": "Person_05",
                    "role": "suspect",
                    "modality": "video",
                    "source": "camera_1",
                    "location": "Parking garage",
                    "content": "Individual in parking area",
                    "timestamp": "2024-01-15T14:00:05",
                    "time_offset": 5,
                    "confidence": 0.92
                },
                {
                    "obs_id": "O3",
                    "entity": "Person_05",
                    "role": "suspect",
                    "modality": "video",
                    "source": "camera_1",
                    "location": "Parking garage",
                    "content": "Individual in parking area",
                    "timestamp": "2024-01-15T14:00:10",
                    "time_offset": 10,
                    "confidence": 0.92
                }
            ]
        },
        "expected_entity_count": 1,
        "expected_mentions": 3
    }
]

def run_test(test_case_dict):
    """Run a single test case."""
    name = test_case_dict["name"]
    case = test_case_dict["case"]
    expected_entity_count = test_case_dict["expected_entity_count"]
    expected_mentions = test_case_dict["expected_mentions"]
    
    print(f"\nTest: {name}")
    print("-" * 80)
    
    try:
        resolver = Resolver()
        result = resolver.resolve_case(case)
        
        # Check results
        entity_count_match = result.entity_count == expected_entity_count
        
        total_mentions = sum(e.total_mention_count for e in result.canonical_entities)
        mentions_match = total_mentions == expected_mentions
        
        status_ok = result.status == "success"
        
        # Print results
        print(f"  Status: {result.status} {'✓' if status_ok else '✗'}")
        print(f"  Entity Count: {result.entity_count} (expected {expected_entity_count}) {'✓' if entity_count_match else '✗'}")
        print(f"  Total Mentions: {total_mentions} (expected {expected_mentions}) {'✓' if mentions_match else '✗'}")
        
        if result.canonical_entities:
            for entity in result.canonical_entities:
                print(f"    - {entity.entity_id}: {entity.aliases} [{entity.total_mention_count} mentions]")
        
        # Overall result
        if entity_count_match and mentions_match and status_ok:
            print(f"  Result: ✓ PASS")
            return True
        else:
            print(f"  Result: ✗ FAIL")
            return False
            
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Result: ✗ FAIL")
        return False

# Run all tests
print("=" * 80)
print("COMPREHENSIVE ENTITY RESOLUTION VALIDATION")
print("=" * 80)

results = []
for test_case in TEST_CASES:
    passed = run_test(test_case)
    results.append((test_case["name"], passed))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
passed_count = sum(1 for _, passed in results if passed)
total_count = len(results)
print(f"Passed: {passed_count}/{total_count}")
for name, passed in results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")

if passed_count == total_count:
    print("\n✓ ALL TESTS PASSED - Entity resolution pipeline is working correctly!")
else:
    print(f"\n✗ {total_count - passed_count} TESTS FAILED - Issues remain")
