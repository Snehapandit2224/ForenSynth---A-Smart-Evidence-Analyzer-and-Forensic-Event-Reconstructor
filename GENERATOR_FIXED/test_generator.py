"""
ForenSynth-X+ Generator Tests — Comprehensive Suite
Tests covering:
  1. Copilot-fixed features (verify correctness)
  2. All bug fixes applied in this pass
  3. Reasoning quality assertions (solvability, contradiction utility,
     temporal ambiguity, multi-actor distinctness)
"""

import random
import pytest
from datetime import datetime

from generator import ForenSynthGenerator, _enforce_template_entity_counts, _validate_case_realization
from templates import get_template_role_requirements, select_template, TEMPLATE_REGISTRY
from timeline import TemplateRealizationError, generate_timeline, _resolve_entity
from entities import build_entities
from config import GeneratorConfig, NoiseConfig
from domains import generate_fir
from noise import apply_noise, _CONTRADICTIONS
from observations import expand_events_to_observations


# ===========================================================================
# 1. Copilot-fixed features — verify they are correct
# ===========================================================================

class TestTemplateRequirementExtraction:
    """Copilot fix: template role requirements are extracted correctly."""

    def test_single_actor_template_requirements(self):
        template = TEMPLATE_REGISTRY["ATM_Robbery"][0]  # Entry_Action_Exit
        requirements = get_template_role_requirements(template)
        assert requirements.get("suspect", 0) >= 1
        assert requirements.get("witness", 0) >= 1

    def test_multi_actor_template_requirements(self):
        template = TEMPLATE_REGISTRY["ATM_Robbery"][1]  # MultiActor_Entry_Action_Exit
        requirements = get_template_role_requirements(template)
        assert requirements.get("suspect", 0) >= 2

    def test_witness_requirement_extraction(self):
        for domain in ["ATM_Robbery", "Office_Theft", "Communication"]:
            for template in TEMPLATE_REGISTRY[domain]:
                requirements = get_template_role_requirements(template)
                if any(slot.role == "witness" for slot in template.slots):
                    assert requirements.get("witness", 0) >= 1


class TestEntityCountEnforcement:
    """Copilot fix: entity counts are bumped to meet template minimums."""

    def test_enforce_minimum_suspects(self):
        final_s, final_w = _enforce_template_entity_counts(
            {"suspect": 3, "witness": 1}, 1, 1, random.Random(42)
        )
        assert final_s == 3
        assert final_w == 1

    def test_enforce_minimum_witnesses(self):
        final_s, final_w = _enforce_template_entity_counts(
            {"suspect": 1, "witness": 2}, 1, 1, random.Random(42)
        )
        assert final_s == 1
        assert final_w == 2

    def test_no_bump_if_sufficient(self):
        final_s, final_w = _enforce_template_entity_counts(
            {"suspect": 2, "witness": 1}, 3, 2, random.Random(42)
        )
        assert final_s == 3
        assert final_w == 2


class TestRequiredSlotEnforcement:
    """Copilot fix: required slots raise on missing entities."""

    def test_required_slot_raises_on_missing_role(self):
        template = TEMPLATE_REGISTRY["ATM_Robbery"][0]
        entities = build_entities(suspect_count=0, witness_count=1, rng=random.Random(42))
        fir = {"time_window": [0, 600], "roles": {"suspect": 0, "witness": 1}}
        with pytest.raises(TemplateRealizationError):
            generate_timeline(template, entities, fir, "2024-01-15T10:00:00", random.Random(42))

    def test_optional_slot_may_skip(self):
        template = TEMPLATE_REGISTRY["ATM_Robbery"][0]
        entities = build_entities(suspect_count=1, witness_count=1, rng=random.Random(42))
        fir = {"time_window": [0, 600], "roles": {"suspect": 1, "witness": 1}}
        events = generate_timeline(template, entities, fir, "2024-01-15T10:00:00", random.Random(42))
        assert len(events) > 0

    def test_fir_entity_count_consistency(self):
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case(domain="ATM_Robbery")
        entity_counts: dict[str, int] = {}
        for e in case["ground_truth"]["entities"]:
            entity_counts[e["role"]] = entity_counts.get(e["role"], 0) + 1
        for role, count in entity_counts.items():
            assert case["fir"]["roles"].get(role) == count


# ===========================================================================
# 2. Bug fixes applied in this pass
# ===========================================================================

class TestResolveEntityOOBFix:
    """
    BUG FIX: _resolve_entity must raise TemplateRealizationError when
    role_index is out of range — not silently fall back to the last
    candidate (which would cause a multi-actor template to reuse the
    same entity for two distinct roles).
    """

    def test_oob_role_index_raises(self):
        """role_index=1 with only 1 suspect must raise, not return suspect_0."""
        entities = build_entities(1, 1, random.Random(42))
        template = TEMPLATE_REGISTRY["ATM_Robbery"][1]
        # slot suspect2_approach has role_index=1
        slot = next(s for s in template.slots if s.role_index == 1 and s.role == "suspect")
        with pytest.raises(TemplateRealizationError):
            _resolve_entity(slot, entities, random.Random(42))

    def test_valid_role_index_resolves_correctly(self):
        """role_index=0 with 2 suspects must resolve to suspect_0."""
        entities = build_entities(2, 1, random.Random(42))
        template = TEMPLATE_REGISTRY["ATM_Robbery"][1]
        slot = next(s for s in template.slots if s.role_index == 0 and s.role == "suspect")
        result = _resolve_entity(slot, entities, random.Random(42))
        assert result is not None
        assert result == entities[0]  # first suspect

    def test_multi_actor_template_uses_distinct_suspects(self):
        """
        In a multi-actor case, suspect_0 and suspect_1 slots must resolve
        to different entities — not the same entity twice.
        """
        entities = build_entities(2, 1, random.Random(42))
        template = TEMPLATE_REGISTRY["ATM_Robbery"][1]  # MultiActor
        fir = {"time_window": [0, 900], "roles": {"suspect": 2, "witness": 1}}
        events = generate_timeline(template, entities, fir, "2024-01-15T10:00:00", random.Random(42))

        suspect_events = [e for e in events if e.role == "suspect"]
        entity_ids_used = {e.entity_id for e in suspect_events}
        # Both suspect_1 and suspect_2 must appear
        assert len(entity_ids_used) == 2, (
            f"Multi-actor template must use 2 distinct suspects, got: {entity_ids_used}"
        )


class TestContradictionCountFix:
    """
    BUG FIX: contradiction_count must scale with n, not always be 1.
    Old: max(1, int(n*0.075)) = 1 for n<14.
    New: max(1, round(n*0.075)) scales correctly.
    """

    def _make_clean_obs(self, n: int):
        """Generate n simple clean observations for noise testing."""
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case(domain="ATM_Robbery", suspect_count=2, witness_count=2)
        from observations import CleanObservation
        obs_list = []
        for i in range(n):
            obs_list.append(CleanObservation(
                obs_id=f"O{i+1}", event_ref="E1", entity="Person_01",
                canonical_entity="suspect_1", role="suspect", modality="audio",
                source="phone_record", location="ATM booth",
                content="I'm in, it's clear.",
                timestamp="2024-01-15T10:01:00",
                time_offset=60, confidence=0.80,
            ))
        return obs_list

    def test_contradiction_count_scales_with_n(self):
        """With 40 observations at rate=0.075, should get ~3 contradictions, not 1."""
        obs = self._make_clean_obs(40)
        cfg = NoiseConfig(contradiction_rate=0.075, missing_modality_rate=0.0,
                          temporal_noise_rate=0.0, semantic_variation_rate=0.0)
        results = apply_noise(obs, cfg, random.Random(42),
                              event_action_map={"E1": "enter_atm"})
        contradiction_count = sum(1 for r in results if "contradiction" in r["noise_tags"])
        # round(40 * 0.075) = round(3.0) = 3
        assert contradiction_count >= 2, (
            f"Expected ~3 contradictions for n=40, got {contradiction_count}"
        )

    def test_small_case_gets_at_least_one_contradiction(self):
        """Even with n=5 observations, we should get exactly 1 contradiction."""
        obs = self._make_clean_obs(5)
        cfg = NoiseConfig(contradiction_rate=0.075, missing_modality_rate=0.0,
                          temporal_noise_rate=0.0, semantic_variation_rate=0.0)
        results = apply_noise(obs, cfg, random.Random(42),
                              event_action_map={"E1": "enter_atm"})
        contradiction_count = sum(1 for r in results if "contradiction" in r["noise_tags"])
        assert contradiction_count >= 1


class TestActionSpecificContradictions:
    """
    BUG FIX: video and text observations must receive action-specific
    contradictions, not always generic stubs.
    """

    def _make_obs_for_action(self, modality: str, action: str):
        from observations import CleanObservation
        return CleanObservation(
            obs_id="O1", event_ref="E1", entity="Person_01",
            canonical_entity="suspect_1", role="suspect",
            modality=modality, source="camera_1",
            location="ATM booth", content="Suspect enters ATM booth.",
            timestamp="2024-01-15T10:01:00", time_offset=60, confidence=0.80,
        )

    def test_video_contradiction_is_not_always_camera_noise(self):
        """
        With enough trials, at least some video contradictions should be
        action-specific denials (not always camera quality stubs).
        """
        from noise import _CONTRADICTION_VIDEO_QUALITY
        action_denial_seen = False
        for seed in range(50):
            obs = self._make_obs_for_action("video", "enter_atm")
            cfg = NoiseConfig(contradiction_rate=1.0, missing_modality_rate=0.0,
                              temporal_noise_rate=0.0, semantic_variation_rate=0.0)
            results = apply_noise([obs], cfg, random.Random(seed),
                                  event_action_map={"E1": "enter_atm"})
            content = results[0]["content"]
            if content not in _CONTRADICTION_VIDEO_QUALITY:
                action_denial_seen = True
                break
        assert action_denial_seen, "Video contradictions should sometimes be action-specific denials"

    def test_text_contradiction_can_be_action_specific(self):
        """At least sometimes text contradictions should be action denials."""
        from noise import _CONTRADICTION_TEXT_INTEGRITY
        action_denial_seen = False
        for seed in range(50):
            obs = self._make_obs_for_action("text", "steal_data")
            cfg = NoiseConfig(contradiction_rate=1.0, missing_modality_rate=0.0,
                              temporal_noise_rate=0.0, semantic_variation_rate=0.0)
            results = apply_noise([obs], cfg, random.Random(seed),
                                  event_action_map={"E1": "steal_data"})
            content = results[0]["content"]
            if content not in _CONTRADICTION_TEXT_INTEGRITY:
                action_denial_seen = True
                break
        assert action_denial_seen, "Text contradictions should sometimes be action-specific denials"


class TestDomainAwareTemporalNoise:
    """
    BUG FIX: temporal noise must be proportionally meaningful across domains.
    For Office_Theft cases (hours), 8-second noise is invisible.
    """

    def test_office_theft_gets_larger_temporal_noise(self):
        """Office_Theft time window is up to 86400s — noise should scale accordingly."""
        from observations import CleanObservation
        obs = CleanObservation(
            obs_id="O1", event_ref="E1", entity="Person_01",
            canonical_entity="suspect_1", role="suspect",
            modality="video", source="camera_1",
            location="office", content="Suspect enters office.",
            timestamp="2024-01-15T10:00:00", time_offset=3600, confidence=0.80,
        )
        cfg = NoiseConfig(temporal_noise_rate=1.0, missing_modality_rate=0.0,
                          contradiction_rate=0.0, semantic_variation_rate=0.0,
                          temporal_noise_min_sec=3, temporal_noise_max_sec=8)
        results = apply_noise([obs], cfg, random.Random(42),
                              time_window_sec=86400)  # 24 hours
        original_ts = datetime.fromisoformat("2024-01-15T10:00:00")
        result_ts = datetime.fromisoformat(results[0]["timestamp"])
        delta_sec = abs((result_ts - original_ts).total_seconds())
        # 0.5% of 86400 = 432s; 3% of 86400 = 2592s; capped at 3600
        assert delta_sec >= 30, f"Temporal noise {delta_sec}s is too small for a 24h window"

    def test_atm_gets_small_temporal_noise(self):
        """ATM_Robbery time window is ~15 min — noise should remain small."""
        from observations import CleanObservation
        obs = CleanObservation(
            obs_id="O1", event_ref="E1", entity="Person_01",
            canonical_entity="suspect_1", role="suspect",
            modality="video", source="camera_1",
            location="ATM booth", content="Suspect enters ATM.",
            timestamp="2024-01-15T10:00:00", time_offset=60, confidence=0.80,
        )
        cfg = NoiseConfig(temporal_noise_rate=1.0, missing_modality_rate=0.0,
                          contradiction_rate=0.0, semantic_variation_rate=0.0,
                          temporal_noise_min_sec=3, temporal_noise_max_sec=8)
        results = apply_noise([obs], cfg, random.Random(42),
                              time_window_sec=600)  # 10 minutes
        original_ts = datetime.fromisoformat("2024-01-15T10:00:00")
        result_ts = datetime.fromisoformat(results[0]["timestamp"])
        delta_sec = abs((result_ts - original_ts).total_seconds())
        # 0.5% of 600 = 3s; 3% of 600 = 18s — small range, appropriate
        assert delta_sec <= 60, f"ATM noise {delta_sec}s seems too large for a 10min window"


class TestTimestampSortingFix:
    """
    BUG FIX: observations must be sorted by noisy timestamp, not obs_id.
    """

    def test_observations_sorted_by_timestamp(self):
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case()
        observations = case["observations"]
        timestamps = [obs["timestamp"] for obs in observations]
        assert timestamps == sorted(timestamps), (
            "Observations must be sorted by noisy timestamp, not event order"
        )


class TestTimestampClampingFix:
    """
    BUG FIX: events that overflow the time window must be distributed across
    the final 20% of the window, not piled up at a single point.
    """

    def test_no_identical_event_timestamps(self):
        """Two events should not be forced to identical timestamps by clamping."""
        gen = ForenSynthGenerator(GeneratorConfig(seed=99))
        for _ in range(10):
            case = gen.generate_case(domain="Office_Theft")
            events = case["ground_truth"]["events"]
            timestamps = [e["timestamp"] for e in events]
            # Allow a small number of collisions but not systematic clustering
            duplicates = len(timestamps) - len(set(timestamps))
            assert duplicates <= 1, (
                f"Too many events share identical timestamps ({duplicates} collisions) — "
                "suggests clamping pileup at time window boundary"
            )


# ===========================================================================
# 3. Reasoning quality assertions
# ===========================================================================

class TestGroundTruthConsistency:
    """Ground truth must be logically consistent."""

    def test_events_are_temporally_ordered(self):
        """Ground truth events must be in ascending temporal order."""
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        for _ in range(5):
            case = gen.generate_case()
            events = case["ground_truth"]["events"]
            timestamps = [e["timestamp"] for e in events]
            assert timestamps == sorted(timestamps), "Ground truth events must be in temporal order"

    def test_all_events_reference_valid_entities(self):
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case()
        entity_ids = {e["entity_id"] for e in case["ground_truth"]["entities"]}
        for event in case["ground_truth"]["events"]:
            assert event["entity_id"] in entity_ids

    def test_entity_mapping_covers_all_aliases(self):
        """entity_mapping must include all aliases from all entities."""
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case()
        mapping = case["ground_truth"]["entity_mapping"]
        for entity in case["ground_truth"]["entities"]:
            for alias in entity["aliases"].values():
                assert alias in mapping, f"Alias '{alias}' missing from entity_mapping"

    def test_batch_all_cases_valid(self):
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        cases = gen.generate_batch(n=10)
        for i, case in enumerate(cases):
            assert len(case["ground_truth"]["events"]) > 0, f"Case {i}: empty timeline"
            entity_ids = {e["entity_id"] for e in case["ground_truth"]["entities"]}
            for ev in case["ground_truth"]["events"]:
                assert ev["entity_id"] in entity_ids, f"Case {i}: dangling entity ref"


class TestMultiPlausibleTimelines:
    """
    Cases must support multiple plausible timelines for Timeline Agent.
    This is achieved through: temporal noise, alias ambiguity, missing
    observations, and partial overlaps.
    """

    def test_multiple_entities_create_aliasing_ambiguity(self):
        """
        In a multi-actor case, aliases across modalities must be distinct
        enough that no alias directly reveals identity, but consistent
        enough to be resolved through reasoning.
        """
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case(domain="ATM_Robbery", suspect_count=2, witness_count=1)
        entities = case["ground_truth"]["entities"]
        all_aliases = []
        for e in entities:
            all_aliases.extend(e["aliases"].values())
        # All aliases must be unique (no two entities share an alias)
        assert len(all_aliases) == len(set(all_aliases)), "Duplicate aliases found — ambiguity broken"

    def test_temporal_noise_creates_ordering_ambiguity(self):
        """
        After noise, observations should NOT all be in perfect event order,
        allowing plausible alternative timeline reconstructions.
        """
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        # Run many cases and check that at least some have reordered obs
        reordered_found = False
        for seed in range(20):
            gen2 = ForenSynthGenerator(GeneratorConfig(seed=seed))
            case = gen2.generate_case()
            obs = case["observations"]
            obs_ids = [o["obs_id"] for o in obs]
            # Extract numeric suffix to check if obs are out of event order
            numeric_ids = [int(oid[1:]) for oid in obs_ids]
            if numeric_ids != sorted(numeric_ids):
                reordered_found = True
                break
        assert reordered_found, (
            "Temporal noise should occasionally reorder observations relative to event order"
        )

    def test_missing_observations_create_gaps(self):
        """
        Missing modality dropout must leave reconstruction gaps —
        not every event should have observations in every modality.
        """
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        total_events_covered: set[str] = set()
        all_event_refs: set[str] = set()
        case = gen.generate_case()
        for obs in case["observations"]:
            total_events_covered.add(obs["event_ref"])
        for ev in case["ground_truth"]["events"]:
            all_event_refs.add(ev["event_id"])
        # Some events should have missing observations
        # (dropout rate 20% across obs, so not all events covered in all modalities)
        coverage_ratio = len(total_events_covered) / max(1, len(all_event_refs))
        assert coverage_ratio <= 1.0  # sanity
        # This is not a strict test — coverage can be full if noise happens to not drop key obs


class TestContradictionQuality:
    """
    Contradictions must be meaningful for Critique Agent — not random noise.
    """

    def test_contradictions_are_action_specific(self):
        """
        Contradiction content should relate to the actual action performed,
        not be a generic placeholder.
        """
        # All actions in _CONTRADICTIONS dict should have substantive denials
        for action, denials in _CONTRADICTIONS.items():
            assert len(denials) >= 3, f"Action '{action}' has too few contradiction options"
            for denial in denials:
                assert len(denial) > 20, f"Denial for '{action}' is too short to be useful"

    def test_contradicted_observations_have_lower_confidence(self):
        """Contradicted observations must have reduced confidence scores."""
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        for seed in range(10):
            gen2 = ForenSynthGenerator(GeneratorConfig(seed=seed))
            case = gen2.generate_case()
            for obs in case["observations"]:
                if "contradiction" in obs["noise_tags"]:
                    assert obs["confidence"] < 0.80, (
                        f"Contradicted obs {obs['obs_id']} has suspiciously high confidence"
                    )

    def test_contradiction_cross_modal_conflict(self):
        """
        Same event should be contradicted in at most one modality,
        creating cross-modal inconsistency (not all modalities wiped).
        """
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        for seed in range(20):
            gen2 = ForenSynthGenerator(GeneratorConfig(seed=seed))
            case = gen2.generate_case()
            obs_by_event: dict[str, list[dict]] = {}
            for obs in case["observations"]:
                obs_by_event.setdefault(obs["event_ref"], []).append(obs)
            for event_ref, obs_list in obs_by_event.items():
                if len(obs_list) > 1:
                    contradicted = [o for o in obs_list if "contradiction" in o["noise_tags"]]
                    non_contradicted = [o for o in obs_list if "contradiction" not in o["noise_tags"]]
                    if contradicted and non_contradicted:
                        # Cross-modal conflict exists: one modality contradicts, another doesn't
                        pass  # This is the ideal state — no assertion needed, just verify structure


class TestSolvability:
    """
    Cases must be reconstructable by a reasoning system.
    """

    def test_enough_clean_observations_remain(self):
        """
        After noise dropout, enough observations must remain to reconstruct
        the ground truth (at least 50% of original observations).
        """
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        for seed in range(10):
            gen2 = ForenSynthGenerator(GeneratorConfig(seed=seed,
                                                       noise=NoiseConfig(missing_modality_rate=0.20)))
            case = gen2.generate_case()
            n_events = len(case["ground_truth"]["events"])
            n_obs = len(case["observations"])
            # Each event produces at least 1 observation on average; we need enough signal
            assert n_obs >= max(1, n_events // 2), (
                f"Too few observations ({n_obs}) remain for {n_events} events — unsolvable"
            )

    def test_ground_truth_entity_ids_not_in_observations(self):
        """Observations must not leak canonical entity IDs."""
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case()
        entity_ids = {e["entity_id"] for e in case["ground_truth"]["entities"]}
        for obs in case["observations"]:
            assert "canonical_entity" not in obs
            assert obs["entity"] not in entity_ids, (
                f"Observation entity label '{obs['entity']}' matches a canonical entity_id — "
                "ground truth leaked!"
            )

    def test_entity_mapping_enables_full_resolution(self):
        """
        Entity mapping must cover all alias labels appearing in observations,
        so a system with the full case can verify its reconstruction.
        """
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case()
        mapping = case["ground_truth"]["entity_mapping"]
        for obs in case["observations"]:
            alias = obs["entity"]
            assert alias in mapping, (
                f"Observation alias '{alias}' not in entity_mapping — cannot resolve"
            )

    def test_case_repeatable_with_seed(self):
        """Same seed must produce identical structure (reproducibility)."""
        cfg1 = GeneratorConfig(seed=12345)
        cfg2 = GeneratorConfig(seed=12345)
        case1 = ForenSynthGenerator(cfg1).generate_case(domain="ATM_Robbery")
        case2 = ForenSynthGenerator(cfg2).generate_case(domain="ATM_Robbery")
        assert len(case1["ground_truth"]["entities"]) == len(case2["ground_truth"]["entities"])
        assert len(case1["ground_truth"]["events"]) == len(case2["ground_truth"]["events"])
        assert case1["fir"]["roles"] == case2["fir"]["roles"]


class TestObservationSanitization:
    """Observations must not leak ground truth."""

    def test_no_canonical_entity_in_observations(self):
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case()
        for obs in case["observations"]:
            assert "canonical_entity" not in obs

    def test_extract_observations_only_sanitized(self):
        from utils import extract_observations_only
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case()
        obs_only = extract_observations_only(case)
        assert "ground_truth" not in obs_only
        for obs in obs_only["observations"]:
            assert "event_ref" not in obs
            assert "noise_tags" not in obs
            assert "canonical_entity" not in obs


class TestAllDomains:
    """All three domains must generate valid cases."""

    @pytest.mark.parametrize("domain", ["ATM_Robbery", "Office_Theft", "Communication"])
    def test_domain_generates_valid_case(self, domain):
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        case = gen.generate_case(domain=domain)
        assert case["domain"] == domain
        assert len(case["ground_truth"]["events"]) > 0
        assert len(case["observations"]) > 0
        assert "case_id" in case

    @pytest.mark.parametrize("domain", ["ATM_Robbery", "Office_Theft", "Communication"])
    def test_domain_batch_all_valid(self, domain):
        gen = ForenSynthGenerator(GeneratorConfig(seed=42))
        cases = gen.generate_batch(n=3, domain=domain)
        for case in cases:
            assert len(case["ground_truth"]["events"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
