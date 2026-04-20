"""Entity resolution schemas."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
from datetime import datetime


class NormalizedAlias(BaseModel):
    """Parsed alias structure."""

    original: str = Field(..., description="Original alias string (e.g., Person_05)")
    alias_type: str = Field(..., description="Parsed type (Person, Speaker, sms, etc.)")
    alias_id: str = Field(
        ..., description="Parsed ID component (05, A, 15, etc.)"
    )
    modality_hint: str = Field(
        ..., description="Inferred modality from alias pattern"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Parsing confidence"
    )

    class Config:
        extra = "forbid"

    @field_validator("original", "alias_type", "alias_id", "modality_hint", mode="before")
    @classmethod
    def validate_non_empty_string(cls, v):
        """Ensure string fields are non-empty."""
        if not isinstance(v, str):
            raise ValueError("Must be a string")
        if not v.strip():
            raise ValueError("Must not be empty")
        return v


class CanonicalEntitySchema(BaseModel):
    """Resolved canonical entity (output schema)."""

    entity_id: str = Field(..., description="Unique entity ID (e.g., PERSON_0001)")
    merged_aliases: List[str] = Field(
        ..., description="All aliases merged into this entity"
    )
    dominant_label: str = Field(
        ..., description="Preferred alias for human readability"
    )
    role: Optional[str] = Field(default=None, description="Primary role if consistent")

    # Confidence and statistics
    entity_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall entity confidence [0.0, 1.0]"
    )
    mention_count: int = Field(ge=0, description="Number of observations for this entity")
    modality_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Count per modality"
    )
    source_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Count per source"
    )
    location_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Count per location"
    )
    temporal_span_sec: Optional[int] = Field(
        default=None, description="Time span from first to last mention"
    )

    class Config:
        extra = "forbid"

    @field_validator(
        "entity_id", "dominant_label", mode="before"
    )
    @classmethod
    def validate_non_empty_string(cls, v):
        """Ensure string fields are non-empty."""
        if not isinstance(v, str):
            raise ValueError("Must be a string")
        if not v.strip():
            raise ValueError("Must not be empty")
        return v

    @field_validator("merged_aliases", mode="before")
    @classmethod
    def validate_merged_aliases(cls, v):
        """Ensure at least one merged alias."""
        if not isinstance(v, list):
            raise ValueError("merged_aliases must be list")
        if len(v) == 0:
            raise ValueError("merged_aliases must not be empty")
        if not all(isinstance(a, str) and a.strip() for a in v):
            raise ValueError("All aliases must be non-empty strings")
        return v
