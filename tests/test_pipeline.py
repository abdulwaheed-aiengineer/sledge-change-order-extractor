"""End-to-end tests for the Extraction Pipeline."""

from unittest.mock import MagicMock

from app.models.change_order import (
    ChangeOrderExtractionResponse,
    RawChangeOrderExtraction,
    RawCostImpactCandidate,
    RawFieldCandidate,
    RawScheduleImpactCandidate,
)
from app.models.enums import ApprovalStatus, CostImpactType, DocumentStatus, SourceType
from app.services.confidence_scorer import ConfidenceScorer
from app.services.llm_extractor import LLMExtractor
from app.services.ocr_service import OCRService
from app.services.pdf_extractor import PDFExtractor
from app.services.pipeline import ExtractionPipeline
from app.services.text_normalizer import TextNormalizer
from app.services.validator import ChangeOrderValidator


def test_pipeline_extract_from_text_success(
    sample_standard_text: str,
    mock_standard_raw_extraction: RawChangeOrderExtraction,
) -> None:
    """Verifies end-to-end extraction from raw text with clean inputs."""
    mock_llm = LLMExtractor()
    mock_llm.extract = MagicMock(return_value=mock_standard_raw_extraction)  # type: ignore

    pipeline = ExtractionPipeline(
        pdf_extractor=PDFExtractor(),
        ocr_service=OCRService(),
        text_normalizer=TextNormalizer(),
        llm_extractor=mock_llm,
        validator=ChangeOrderValidator(),
        confidence_scorer=ConfidenceScorer(),
    )

    response = pipeline.extract_from_text(sample_standard_text)

    assert isinstance(response, ChangeOrderExtractionResponse)
    assert response.document_status == DocumentStatus.SUCCESS
    assert response.source_type == SourceType.RAW_TEXT
    assert len(response.extraction_warnings) == 0

    # Field checks
    assert response.project_name.value == "Oakridge Medical Pavilion"
    assert response.project_name.confidence >= 0.90
    assert response.project_name.evidence is not None

    assert response.change_order_number.value == "CO-007"
    assert response.cost_impact.value is not None
    assert response.cost_impact.value.amount == 14500.0
    assert response.cost_impact.value.impact_type == CostImpactType.ADDITION
    assert response.schedule_impact_days.value == 7
    assert response.revised_contract_value.value == 1309500.00
    assert response.approval_status.value == ApprovalStatus.APPROVED


def test_pipeline_extract_from_pdf_success(
    standard_digital_pdf_bytes: bytes,
    mock_standard_raw_extraction: RawChangeOrderExtraction,
) -> None:
    """Verifies end-to-end extraction from a digital PDF document."""
    mock_llm = LLMExtractor()
    mock_llm.extract = MagicMock(return_value=mock_standard_raw_extraction)  # type: ignore

    pipeline = ExtractionPipeline(
        pdf_extractor=PDFExtractor(),
        ocr_service=OCRService(),
        text_normalizer=TextNormalizer(),
        llm_extractor=mock_llm,
        validator=ChangeOrderValidator(),
        confidence_scorer=ConfidenceScorer(),
    )

    response = pipeline.extract_from_pdf(standard_digital_pdf_bytes)

    assert response.document_status == DocumentStatus.SUCCESS
    assert response.source_type == SourceType.DIGITAL_PDF
    assert response.change_order_number.value == "CO-007"
    assert response.cost_impact.value.amount == 14500.0


def test_pipeline_handles_deduction_and_no_cost(
    sample_deduction_text: str,
) -> None:
    """Verifies extraction and normalization of a financial deduction change order."""
    deduction_raw = RawChangeOrderExtraction(
        project_name=RawFieldCandidate(
            value="Westside High School Renovation",
            evidence="PROJECT: Westside High School Renovation",
        ),
        change_order_number=RawFieldCandidate(
            value="CO-012",
            evidence="CHANGE ORDER #: CO-012",
        ),
        cost_impact=RawCostImpactCandidate(
            amount=32500.0,  # Positive in candidate, must normalize to negative
            currency="USD",
            impact_type=CostImpactType.DEDUCTION,
            evidence="Cost Impact: Deduction of ($32,500.00)",
        ),
        schedule_impact_days=RawScheduleImpactCandidate(
            days=0,
            evidence="The Contract Time will be unchanged: 0 days impact.",
        ),
        revised_contract_value=RawFieldCandidate(
            value="4787500.00",
            evidence="Revised Contract Value: $4,787,500.00",
        ),
        previous_contract_sum=4820000.0,
    )

    mock_llm = LLMExtractor()
    mock_llm.extract = MagicMock(return_value=deduction_raw)  # type: ignore

    pipeline = ExtractionPipeline(
        pdf_extractor=PDFExtractor(),
        ocr_service=OCRService(),
        text_normalizer=TextNormalizer(),
        llm_extractor=mock_llm,
        validator=ChangeOrderValidator(),
        confidence_scorer=ConfidenceScorer(),
    )

    response = pipeline.extract_from_text(sample_deduction_text)

    assert response.cost_impact.value is not None
    assert response.cost_impact.value.impact_type == CostImpactType.DEDUCTION
    assert response.cost_impact.value.amount == -32500.0  # Signed negative
    assert response.schedule_impact_days.value == 0
    # Consistency check: 4,820,000 + (-32,500) == 4,787,500
    assert response.revised_contract_value.value == 4787500.00


def test_pipeline_handles_missing_fields_cleanly(sample_missing_fields_text: str) -> None:
    """Verifies that missing fields are set to None with 0.0 confidence and partial success status."""
    sparse_raw = RawChangeOrderExtraction(
        project_name=RawFieldCandidate(
            value="Riverside Plaza Retail Shell",
            evidence="Project: Riverside Plaza Retail Shell",
        ),
        cost_impact=RawCostImpactCandidate(
            amount=0.0,
            currency="USD",
            impact_type=CostImpactType.NO_COST,
            evidence="Cost Impact: No cost ($0.00 change)",
        ),
        # All other fields omitted / default null
    )

    mock_llm = LLMExtractor()
    mock_llm.extract = MagicMock(return_value=sparse_raw)  # type: ignore

    pipeline = ExtractionPipeline(
        pdf_extractor=PDFExtractor(),
        ocr_service=OCRService(),
        text_normalizer=TextNormalizer(),
        llm_extractor=mock_llm,
        validator=ChangeOrderValidator(),
        confidence_scorer=ConfidenceScorer(),
    )

    response = pipeline.extract_from_text(sample_missing_fields_text)

    assert response.document_status == DocumentStatus.PARTIAL_SUCCESS
    assert response.project_name.value == "Riverside Plaza Retail Shell"
    assert response.project_name.confidence > 0.8

    # Missing fields must strictly have null value, 0.0 confidence, null evidence
    assert response.contractor_name.value is None
    assert response.contractor_name.confidence == 0.0
    assert response.contractor_name.evidence is None

    assert response.original_contract_number.value is None
    assert response.original_contract_number.confidence == 0.0


def test_pipeline_flags_conflicts_and_arithmetic_errors(sample_conflict_text: str) -> None:
    """Verifies that conflicting fields and arithmetic errors generate warnings and penalize confidence."""
    conflict_raw = RawChangeOrderExtraction(
        project_name=RawFieldCandidate(
            value="Harbor Point Warehouse",
            evidence="PROJECT: Harbor Point Warehouse",
        ),
        change_order_number=RawFieldCandidate(
            value="CO-003",
            evidence="CHANGE ORDER NUMBER: CO-003",
        ),
        issue_date=RawFieldCandidate(
            value="2024-02-30",  # Invalid calendar date
            evidence="DATE OF ISSUANCE: 2024-02-30",
        ),
        cost_impact=RawCostImpactCandidate(
            amount=25000.0,
            currency="USD",
            impact_type=CostImpactType.ADDITION,
            evidence="Cost Impact: Net addition of $25,000.00",
        ),
        revised_contract_value=RawFieldCandidate(
            value="540000.00",  # Discrepant ($500,000 + $25,000 != $540,000)
            evidence="Revised Contract Value: $540,000.00",
        ),
        previous_contract_sum=500000.0,
        detected_conflicts=[
            "Section 4 notes conflict with header: stated $28,000 instead of $25,000.",
        ],
    )

    mock_llm = LLMExtractor()
    mock_llm.extract = MagicMock(return_value=conflict_raw)  # type: ignore

    pipeline = ExtractionPipeline(
        pdf_extractor=PDFExtractor(),
        ocr_service=OCRService(),
        text_normalizer=TextNormalizer(),
        llm_extractor=mock_llm,
        validator=ChangeOrderValidator(),
        confidence_scorer=ConfidenceScorer(),
    )

    response = pipeline.extract_from_text(sample_conflict_text)

    assert response.document_status == DocumentStatus.PARTIAL_SUCCESS
    assert len(response.extraction_warnings) > 0

    # Check for date warning
    assert any("2024-02-30" in w for w in response.extraction_warnings)
    # Check for arithmetic discrepancy warning
    assert any("arithmetic discrepancy" in w.lower() for w in response.extraction_warnings)
    # Check for conflict warning
    assert any("conflict detected" in w.lower() for w in response.extraction_warnings)

    # Discrepant revised contract value should have reduced confidence
    assert response.revised_contract_value.confidence < 0.80
