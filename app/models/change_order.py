"""Final and intermediate change-order extraction schemas."""

from pydantic import BaseModel, Field

from app.models.enums import ApprovalStatus, CostImpactType, DocumentStatus, SourceType
from app.models.extracted_field import CostImpact, ExtractedField


class RawFieldCandidate(BaseModel):
    """Raw extraction item from the LLM prior to validation and scoring."""

    value: str | None = Field(default=None, description="Extracted raw value or null if not found.")
    evidence: str | None = Field(
        default=None, description="Exact quotation from document supporting this value."
    )
    page: int | None = Field(default=None, description="Page number where evidence appears.")
    is_inferred: bool = Field(
        default=False, description="True if value was deduced rather than explicitly stated."
    )


class RawCostImpactCandidate(BaseModel):
    """Raw financial impact candidate from LLM."""

    amount: float | None = Field(default=None, description="Numerical amount extracted.")
    currency: str = Field(default="USD", description="Currency code, e.g. USD.")
    impact_type: CostImpactType = Field(
        default=CostImpactType.NO_COST,
        description="addition, deduction, or no_cost.",
    )
    evidence: str | None = Field(
        default=None, description="Exact quotation supporting the cost impact."
    )
    page: int | None = Field(default=None, description="Page number where evidence appears.")
    is_inferred: bool = Field(default=False, description="True if cost impact was inferred.")


class RawScheduleImpactCandidate(BaseModel):
    """Raw schedule impact candidate from LLM."""

    days: int | None = Field(default=None, description="Calendar or working days impact.")
    evidence: str | None = Field(
        default=None, description="Exact quotation supporting schedule impact."
    )
    page: int | None = Field(default=None, description="Page number where evidence appears.")
    is_inferred: bool = Field(default=False, description="True if schedule impact was inferred.")


class RawChangeOrderExtraction(BaseModel):
    """Intermediate structured extraction produced directly by the LLM."""

    project_name: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    project_address: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    change_order_number: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    original_contract_number: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    title: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    description: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    contractor_name: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    owner_name: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    issue_date: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    requested_by: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    reason_for_change: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    cost_impact: RawCostImpactCandidate = Field(default_factory=RawCostImpactCandidate)
    schedule_impact_days: RawScheduleImpactCandidate = Field(
        default_factory=RawScheduleImpactCandidate
    )
    revised_contract_value: RawFieldCandidate = Field(default_factory=RawFieldCandidate)
    approval_status: RawFieldCandidate = Field(default_factory=RawFieldCandidate)

    # Contextual fields helpful for consistency checks
    previous_contract_sum: float | None = Field(
        default=None,
        description="Stated contract sum prior to this change order, if explicitly provided in the document.",
    )
    detected_conflicts: list[str] = Field(
        default_factory=list,
        description="Any conflicting statements, contradictory dates, or conflicting amounts noted in the text.",
    )


class ChangeOrderExtractionResponse(BaseModel):
    """Final validated and scored change order extraction response."""

    document_status: DocumentStatus = Field(
        description="Overall processing status: success, partial_success, or failed.",
    )
    source_type: SourceType = Field(
        description="Document ingestion mode: digital_pdf, scanned_pdf, or raw_text.",
    )
    extraction_warnings: list[str] = Field(
        default_factory=list,
        description="List of warnings, discrepancies, or anomalies flagged during validation.",
    )

    # 15 Required Extracted Fields
    project_name: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    project_address: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    change_order_number: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    original_contract_number: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField[str]()
    )
    title: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    description: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    contractor_name: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    owner_name: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    issue_date: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    requested_by: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    reason_for_change: ExtractedField[str] = Field(default_factory=lambda: ExtractedField[str]())
    cost_impact: ExtractedField[CostImpact] = Field(
        default_factory=lambda: ExtractedField[CostImpact]()
    )
    schedule_impact_days: ExtractedField[int] = Field(default_factory=lambda: ExtractedField[int]())
    revised_contract_value: ExtractedField[float] = Field(
        default_factory=lambda: ExtractedField[float]()
    )
    approval_status: ExtractedField[ApprovalStatus] = Field(
        default_factory=lambda: ExtractedField[ApprovalStatus]()
    )
