"""Observation and case-level schemas."""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from enum import Enum
from datetime import datetime


class Modality(str, Enum):
    """Observation modality types."""

    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class Observation(BaseModel):
    """Individual observation (must match input contract exactly)."""

    obs_id: str = Field(..., description="Unique observation identifier (e.g., O1, O2)")
    entity: str = Field(..., description="Alias label (e.g., Person_05, Speaker_A)")
    role: str = Field(..., description="Entity role (e.g., suspect, witness, system)")
    modality: Modality = Field(..., description="Observation modality")
    source: str = Field(..., description="Source device/channel label (e.g., camera_1)")
    location: str = Field(..., description="Spatial location string")
    content: str = Field(..., description="Natural language description")
    timestamp: str = Field(..., description="ISO 8601 UTC datetime string")
    time_offset: int = Field(..., ge=0, description="Seconds from case window start")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence [0.0, 1.0]")
    noise_tags: List[str] = Field(default_factory=list, description="Noise labels")

    class Config:
        use_enum_values = True
        extra = "ignore"

    @field_validator("entity", "role", "source", "location", "content", mode="before")
    @classmethod
    def validate_non_empty_string(cls, v):
        """Ensure string fields are non-empty."""
        if not isinstance(v, str):
            raise ValueError("Must be a string")
        if not v.strip():
            raise ValueError("Must not be empty")
        return v

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_iso_timestamp(cls, v):
        """Validate ISO 8601 timestamp format."""
        if not isinstance(v, str):
            raise ValueError("Timestamp must be string")
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {v}") from e
        return v

    @field_validator("obs_id", mode="before")
    @classmethod
    def validate_obs_id(cls, v):
        """Ensure obs_id is non-empty."""
        if not isinstance(v, str):
            raise ValueError("obs_id must be string")
        if not v.strip():
            raise ValueError("obs_id must not be empty")
        return v


class CaseInput(BaseModel):
    """Top-level input case data."""

    case_id: str = Field(..., description="Case identifier")
    observations: List[Observation] = Field(..., description="List of observations")

    class Config:
        extra = "ignore"

    @field_validator("case_id", mode="before")
    @classmethod
    def validate_case_id(cls, v):
        """Ensure case_id is non-empty."""
        if not isinstance(v, str):
            raise ValueError("case_id must be string")
        if not v.strip():
            raise ValueError("case_id must not be empty")
        return v

    @model_validator(mode="after")
    def validate_at_least_one_observation(self):
        """Ensure at least one observation."""
        if len(self.observations) == 0:
            raise ValueError("Must have at least 1 observation")
        return self


class NormalizedObservation(BaseModel):
    """Observation with normalized and computed fields."""

    # Original fields (preserved)
    obs_id: str
    entity: str  # Original alias string
    role: str
    modality: Modality
    source: str
    location: str
    content: str  # Will be lowercased and whitespace-cleaned
    timestamp: str  # Original ISO string
    time_offset: int
    confidence: float
    noise_tags: List[str]

    # Computed/normalized fields
    timestamp_dt: datetime = Field(..., description="Parsed datetime object")
    timestamp_offset_sec: int = Field(..., description="Unix offset from epoch start")
    content_normalized: str = Field(..., description="Lowercase, whitespace-cleaned content")

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = "forbid"
