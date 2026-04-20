"""README for Entity Resolution Modules."""

# Entity Resolution — Schemas & Services

Production-ready implementation of the intake validation and normalization modules for the Entity Resolution Layer.

## Modules Implemented

### 1. `schemas/observation.py`
Core observation and case-level schemas using Pydantic.

**Classes:**
- `Modality(Enum)` — video, audio, text
- `Observation` — Individual observation (input contract)
- `CaseInput` — Top-level case with observations list
- `NormalizedObservation` — Observation with computed/normalized fields

**Features:**
- Full Pydantic validation
- ISO 8601 timestamp validation
- Confidence score validation [0.0, 1.0]
- Non-empty string enforcement
- All original fields preserved

**Example:**
```python
from schemas.observation import Observation, CaseInput, Modality

obs = Observation(
    obs_id="O1",
    entity="Person_05",
    role="suspect",
    modality=Modality.VIDEO,
    source="camera_1",
    location="ATM booth",
    content="...",
    timestamp="2024-01-15T10:15:30Z",
    time_offset=180,
    confidence=0.95,
    noise_tags=["blur"],
)

case = CaseInput(
    case_id="CASE_001",
    observations=[obs, ...]
)
```

### 2. `schemas/entity.py`
Entity resolution output schemas.

**Classes:**
- `NormalizedAlias` — Parsed alias structure
- `CanonicalEntity` — Resolved entity (output schema)

**Example:**
```python
from schemas.entity import CanonicalEntity

entity = CanonicalEntity(
    entity_id="PERSON_0001",
    merged_aliases=["Person_05", "Speaker_A"],
    dominant_label="Person_05",
    entity_confidence=0.87,
    mention_count=12,
    modality_distribution={"video": 8, "audio": 4},
    source_distribution={"camera_1": 5, ...},
    location_distribution={...},
)
```

### 3. `services/entity_mapping/intake.py`
Intake validation service for observations.

**Classes:**
- `IntakeValidator` — Validates case input
- `IntakeReport` — Validation report with stats and errors
- `IntakeError` — Individual error record

**Features:**
- Pydantic schema validation
- Semantic deduplication (hash-based)
- Confidence score validation
- Timestamp parsing
- Comprehensive error reporting
- Keeps highest confidence when duplicates found

**Example:**
```python
from services.entity_mapping.intake import IntakeValidator

validator = IntakeValidator()
case_input = {
    "case_id": "CASE_001",
    "observations": [...]
}

observations, report = validator.validate_case(case_input)

print(f"Valid: {report.valid_observations}")
print(f"Invalid: {report.invalid_observations}")
print(f"Duplicates: {report.duplicate_observations}")
for error in report.errors:
    print(f"{error.error_type}: {error.message}")
```

### 4. `services/entity_mapping/normalizer.py`
Normalization service for observations.

**Classes:**
- `AliasNormalizer` — Parses alias patterns
- `Normalizer` — Normalizes observations
- `NormalizationReport` — Normalization report

**Features:**
- Alias pattern parsing (regex):
  - Person_*, Speaker_*, sms_*, email_*, report_*, log_*
- Modality hint extraction from alias
- Content normalization:
  - Lowercase conversion
  - Extra whitespace removal
  - Whitespace trimming
- Timestamp parsing to datetime
- Case window alignment
- Deterministic, production-ready logic

**Example:**
```python
from services.entity_mapping.normalizer import Normalizer, AliasNormalizer
from datetime import datetime

# Parse alias
alias = AliasNormalizer.normalize_alias("Person_05")
print(f"Type: {alias.alias_type}, ID: {alias.alias_id}, Modality: {alias.modality_hint}")

# Normalize observations
base_time = datetime.fromisoformat("2024-01-15T10:00:00+00:00")
normalizer = Normalizer(case_base_time=base_time)

normalized, report = normalizer.normalize_observations(observations)

for norm_obs in normalized:
    print(f"Original: '{norm_obs.content}'")
    print(f"Normalized: '{norm_obs.content_normalized}'")
    print(f"Timestamp: {norm_obs.timestamp_dt}")
    print(f"Offset: {norm_obs.timestamp_offset_sec}s")
```

## Validation & Normalization Pipeline

```
Raw Input
    ↓
[1] INTAKE VALIDATION (intake.py)
    ├→ Pydantic schema validation
    ├→ Observation deduplication 
    ├→ Timestamp validation
    ├→ Confidence validation [0.0, 1.0]
    └→ IntakeReport (errors, warnings)
    ↓
[2] NORMALIZATION (normalizer.py)
    ├→ Alias pattern parsing
    ├→ Modality hint extraction
    ├→ Content normalization (lowercase, whitespace)
    ├→ Timestamp parsing to datetime
    ├→ Case window alignment
    └→ NormalizationReport
    ↓
Validated & Normalized Observations (ready for blocking)
```

## Error Handling

All modules implement comprehensive error handling:

### Intake Errors
- `SCHEMA_ERROR` — Pydantic validation failure
- `CONFIDENCE_ERROR` — Confidence outside [0.0, 1.0]
- `TIMESTAMP_ERROR` — Invalid ISO 8601 format
- `UNEXPECTED_ERROR` — Unhandled exceptions

### Normalization Errors
- Logged with observation ID
- Original data preserved on error
- Warnings for partial failures

## Design Principles

1. **Pydantic Validation** — All input validated strictly
2. **Preservation** — All original fields preserved exactly
3. **Normalization** — Deterministic, repeatable transformations
4. **Error Reporting** — Comprehensive, actionable diagnostics
5. **No Ground Truth** — No event_ref or event field usage
6. **Deduplication** — Semantic hash-based duplicate detection
7. **Audit Trail** — All decisions traceable

## Usage Example

```python
from services.entity_mapping.intake import IntakeValidator
from services.entity_mapping.normalizer import Normalizer
from datetime import datetime

# Step 1: Load raw case input (from API or file)
case_input = {
    "case_id": "CASE_20250315_001",
    "observations": [...]
}

# Step 2: Validate intake
validator = IntakeValidator()
observations, intake_report = validator.validate_case(case_input)

if intake_report.invalid_observations > 0:
    print("Intake validation failed:")
    for error in intake_report.errors:
        print(f"  {error.obs_id}: {error.message}")
    exit(1)

print(f"✓ Intake: {intake_report.valid_observations} valid observations")

# Step 3: Normalize observations
base_time = datetime.fromisoformat(case_input["case_id"] + "T10:00:00Z")
normalizer = Normalizer(case_base_time=base_time)
normalized_obs, norm_report = normalizer.normalize_observations(observations)

if norm_report.errors:
    print("Normalization errors:")
    for error in norm_report.errors:
        print(f"  {error}")

print(f"✓ Normalized: {norm_report.successfully_normalized} observations")

# Step 4: Ready for blocking stage
print("✓ Ready for entity mapping pipeline")
```

## Testing

Run examples and validation:

```bash
cd c:/Users/pandi/Documents/Capstone\ Project/code
python -m entity_resolution.examples
```

**Output includes:**
- Intake validation with valid/invalid/duplicate counts
- Alias pattern parsing for all modality types
- Content normalization (whitespace, case)
- Timestamp datetime conversion and alignment
- Error handling with missing fields
- Duplicate detection with confidence-based merge

## Production Notes

- **Deterministic:** All operations are deterministic and reproducible
- **Scalable:** O(n) validation and normalization, no O(n²) operations
- **Robust:** Handles missing fields, invalid formats, duplicates gracefully
- **Auditable:** Every decision logged with rationale
- **Extensible:** Modality and alias patterns easily customizable

## Dependencies

- `pydantic>=2.0` — Data validation
- `python>=3.9` — Async support (if needed)

## Next Phases

1. ✓ Schemas (observation, entity) — **DONE**
2. ✓ Intake validation — **DONE**
3. ✓ Normalization — **DONE**
4. → Blocking (Stage 3)
5. → Feature computation (Stage 4)
6. → Similarity scoring (Stage 5)
7. → Classification (Stage 6)
... (remaining 6 stages)

---

**Implementation Status:** Ready for production use. Clean, validated, documented code.
