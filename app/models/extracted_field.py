"""Models representing individual extracted fields with confidence and evidence."""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from app.models.enums import CostImpactType

T = TypeVar("T")


class CostImpact(BaseModel):
    """Normalized financial impact of the change order."""

    amount: float = Field(
        description="Signed or normalized numerical monetary amount. Positive for addition, negative for deduction, 0.0 for no-cost.",
    )
    currency: str = Field(
        default="USD",
        description="Currency code (e.g., USD, CAD, EUR).",
    )
    impact_type: CostImpactType = Field(
        description="Classification of change: addition, deduction, or no_cost.",
    )


class ExtractedField(BaseModel, Generic[T]):
    """Standard container for an extracted field containing value, confidence, and source evidence."""

    value: T | None = Field(
        default=None,
        description="Extracted and normalized value, or null if missing from document.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Heuristic confidence estimate from 0.0 to 1.0. Missing fields must have 0.0.",
    )
    evidence: str | None = Field(
        default=None,
        description="Supporting verbatim source text justifying the extracted value.",
    )
    page: int | None = Field(
        default=None,
        description="Source document page number (1-indexed) where the evidence was located.",
    )
