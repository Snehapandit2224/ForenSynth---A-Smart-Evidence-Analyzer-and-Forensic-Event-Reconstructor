"""Test suite for entity resolution schemas and services."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from .schemas.observation import Observation, CaseInput, Modality, NormalizedObservation
from .schemas.entity import NormalizedAlias, CanonicalEntity
from .services.entity_mapping.intake import IntakeValidator, IntakeReport
from .services.entity_mapping.normalizer import Normalizer, AliasNormalizer


class TestObservationSchema:
    """Tests for Observation schema."""

    def test_valid_observation(self):
        """Test creating a valid observation."""
        obs = Observation(
            obs_id="O1",
            entity="Person_05",
            role="suspect",
            modality=Modality.VIDEO,
            source="camera_1",
            location="ATM booth",
            content="Test content",
            timestamp="2024-01-15T10:15:30Z",
            time_offset=180,
            confidence=0.95,
            noise_tags=["blur"],
        )
        assert obs.obs_id == "O1"
        assert obs.entity == "Person_05"
        assert obs.confidence == 0.95

    def test_observation_with_no_noise_tags(self):
        """Test observation without noise_tags."""
        obs = Observation(
            obs_id="O1",
            entity="Person_05",
            role="suspect",
            modality=Modality.AUDIO,
            source="mic_1",
            location="room",
            content="Test",
            timestamp="2024-01-15T10:15:30Z",
            time_offset=180,
            confidence=0.9,
        )
        assert obs.noise_tags == []

    def test_confidence_bounds(self):
        """Test confidence score validation."""
        with pytest.raises(ValidationError):
            Observation(
                obs_id="O1",
                entity="Person_05",
                role="suspect",
                modality=Modality.VIDEO,
                source="cam",
                location="room",
                content="Test",
                timestamp="2024-01-15T10:15:30Z",
                time_offset=180,
                confidence=1.5,  # Invalid: > 1.0
            )

        with pytest.raises(ValidationError):
            Observation(
                obs_id="O1",
                entity="Person_05",
                role="suspect",
                modality=Modality.VIDEO,
                source="cam",
                location="room",
                content="Test",
                timestamp="2024-01-15T10:15:30Z",
                time_offset=180,
                confidence=-0.1,  # Invalid: < 0.0
            )

    def test_invalid_timestamp(self):
        """Test invalid timestamp format."""
        with pytest.raises(ValidationError):
            Observation(
                obs_id="O1",
                entity="Person_05",
                role="suspect",
                modality=Modality.VIDEO,
                source="cam",
                location="room",
                content="Test",
                timestamp="2024-01-15 10:15:30",  # Invalid: not ISO 8601
                time_offset=180,
                confidence=0.9,
            )

    def test_empty_entity(self):
        """Test empty entity field."""
        with pytest.raises(ValidationError):
            Observation(
                obs_id="O1",
                entity="",  # Invalid: empty
                role="suspect",
                modality=Modality.VIDEO,
                source="cam",
                location="room",
                content="Test",
                timestamp="2024-01-15T10:15:30Z",
                time_offset=180,
                confidence=0.9,
            )


class TestCaseInputSchema:
    """Tests for CaseInput schema."""

    def test_valid_case_input(self):
        """Test creating valid case input."""
        case = CaseInput(
            case_id="CASE_001",
            observations=[
                Observation(
                    obs_id="O1",
                    entity="Person_05",
                    role="suspect",
                    modality=Modality.VIDEO,
                    source="cam",
                    location="room",
                    content="Test",
                    timestamp="2024-01-15T10:15:30Z",
                    time_offset=180,
                    confidence=0.9,
                )
            ],
        )
        assert case.case_id == "CASE_001"
        assert len(case.observations) == 1

    def test_no_observations(self):
        """Test case with no observations."""
        with pytest.raises(ValidationError):
            CaseInput(case_id="CASE_001", observations=[])


class TestNormalizedAliasSchema:
    """Tests for NormalizedAlias schema."""

    def test_valid_alias(self):
        """Test creating valid alias."""
        alias = NormalizedAlias(
            original="Person_05",
            alias_type="Person",
            alias_id="05",
            modality_hint="video",
            confidence=1.0,
        )
        assert alias.original == "Person_05"
        assert alias.alias_type == "Person"

    def test_default_confidence(self):
        """Test alias with default confidence."""
        alias = NormalizedAlias(
            original="Person_05",
            alias_type="Person",
            alias_id="05",
            modality_hint="video",
        )
        assert alias.confidence == 1.0


class TestCanonicalEntitySchema:
    """Tests for CanonicalEntity schema."""

    def test_valid_canonical_entity(self):
        """Test creating valid canonical entity."""
        entity = CanonicalEntity(
            entity_id="PERSON_0001",
            merged_aliases=["Person_05", "Speaker_A"],
            dominant_label="Person_05",
            entity_confidence=0.87,
            mention_count=12,
        )
        assert entity.entity_id == "PERSON_0001"
        assert len(entity.merged_aliases) == 2

    def test_empty_merged_aliases(self):
        """Test entity with empty merged aliases."""
        with pytest.raises(ValidationError):
            CanonicalEntity(
                entity_id="PERSON_0001",
                merged_aliases=[],  # Invalid: empty
                dominant_label="Person_05",
                entity_confidence=0.87,
                mention_count=12,
            )


class TestIntakeValidator:
    """Tests for IntakeValidator."""

    def test_validate_valid_case(self):
        """Test validating valid case."""
        validator = IntakeValidator()
        case_input = {
            "case_id": "CASE_001",
            "observations": [
                {
                    "obs_id": "O1",
                    "entity": "Person_05",
                    "role": "suspect",
                    "modality": "video",
                    "source": "cam",
                    "location": "room",
                    "content": "Test",
                    "timestamp": "2024-01-15T10:15:30Z",
                    "time_offset": 180,
                    "confidence": 0.9,
                }
            ],
        }

        observations, report = validator.validate_case(case_input)
        assert len(observations) == 1
        assert report.valid_observations == 1
        assert report.invalid_observations == 0

    def test_duplicate_detection(self):
        """Test duplicate observation detection."""
        validator = IntakeValidator()
        case_input = {
            "case_id": "CASE_001",
            "observations": [
                {
                    "obs_id": "O1",
                    "entity": "Person_05",
                    "role": "suspect",
                    "modality": "video",
                    "source": "cam",
                    "location": "room",
                    "content": "Test content",
                    "timestamp": "2024-01-15T10:15:30Z",
                    "time_offset": 180,
                    "confidence": 0.85,
                },
                {
                    "obs_id": "O1_dup",  # Different ID
                    "entity": "Person_05",
                    "role": "suspect",
                    "modality": "video",
                    "source": "cam",
                    "location": "room",
                    "content": "Test content",  # Same content
                    "timestamp": "2024-01-15T10:15:30Z",
                    "time_offset": 180,
                    "confidence": 0.95,  # Higher confidence
                },
            ],
        }

        observations, report = validator.validate_case(case_input)
        assert report.duplicate_observations == 1
        assert report.valid_observations == 1  # Only one kept
        assert report.warnings  # Warning about duplicate

    def test_invalid_schema(self):
        """Test invalid schema."""
        validator = IntakeValidator()
        case_input = {
            "case_id": "CASE_001",
            "observations": [
                {
                    "obs_id": "O1",
                    # Missing 'entity' field
                    "role": "suspect",
                    "modality": "video",
                    "source": "cam",
                    "location": "room",
                    "content": "Test",
                    "timestamp": "2024-01-15T10:15:30Z",
                    "time_offset": 180,
                    "confidence": 0.9,
                }
            ],
        }

        observations, report = validator.validate_case(case_input)
        assert len(observations) == 0
        assert len(report.errors) > 0


class TestAliasNormalizer:
    """Tests for AliasNormalizer."""

    def test_person_pattern(self):
        """Test Person pattern parsing."""
        alias = AliasNormalizer.normalize_alias("Person_05")
        assert alias.alias_type == "Person"
        assert alias.alias_id == "05"
        assert alias.modality_hint == "video"
        assert alias.confidence == 1.0

    def test_speaker_pattern(self):
        """Test Speaker pattern parsing."""
        alias = AliasNormalizer.normalize_alias("Speaker_A")
        assert alias.alias_type == "Speaker"
        assert alias.alias_id == "A"
        assert alias.modality_hint == "audio"

    def test_sms_pattern(self):
        """Test SMS pattern parsing."""
        alias = AliasNormalizer.normalize_alias("sms_15")
        assert alias.alias_type == "sms"
        assert alias.alias_id == "15"
        assert alias.modality_hint == "text"

    def test_case_insensitive_patterns(self):
        """Test case-insensitive pattern matching."""
        patterns = [
            ("person_05", "Person"),
            ("PERSON_05", "Person"),
            ("speaker_a", "Speaker"),
            ("SMS_99", "sms"),
        ]
        for pattern, expected_type in patterns:
            alias = AliasNormalizer.normalize_alias(pattern)
            assert alias.alias_type == expected_type


class TestNormalizer:
    """Tests for Normalizer."""

    def test_normalize_content(self):
        """Test content normalization."""
        Normalizer._normalize_content("  TEST  CONTENT  ") == "test content"

    def test_normalize_with_case_base_time(self):
        """Test normalization with base time."""
        base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")
        normalizer = Normalizer(case_base_time=base_time)

        obs = Observation(
            obs_id="O1",
            entity="Person_05",
            role="suspect",
            modality=Modality.VIDEO,
            source="cam",
            location="room",
            content="  Test  ",
            timestamp="2024-01-15T10:05:00Z",
            time_offset=0,  # Will be recalculated
            confidence=0.9,
        )

        normalized, report = normalizer.normalize_observations([obs])
        assert len(normalized) == 1
        assert normalized[0].timestamp_offset_sec == 300  # 5 minutes
        assert normalized[0].content_normalized == "test"

    def test_normalization_report_with_errors(self):
        """Test normalization report generation."""
        normalizer = Normalizer()

        # Create an observation with whitespace
        obs = Observation(
            obs_id="O1",
            entity="Person_05",
            role="suspect",
            modality=Modality.VIDEO,
            source="cam",
            location="room",
            content="  Multiple   spaces  ",
            timestamp="2024-01-15T10:15:30Z",
            time_offset=180,
            confidence=0.9,
        )

        normalized, report = normalizer.normalize_observations([obs])
        assert report.total_observations == 1
        assert report.successfully_normalized == 1
        assert len(report.errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
