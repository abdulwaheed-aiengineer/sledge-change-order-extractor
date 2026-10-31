"""Pytest fixtures and test helpers for the change order extraction pipeline."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.api.dependencies import get_extraction_pipeline
from app.main import app
from app.models.change_order import (
    RawChangeOrderExtraction,
    RawCostImpactCandidate,
    RawFieldCandidate,
    RawScheduleImpactCandidate,
)
from app.models.enums import CostImpactType
from app.services.confidence_scorer import ConfidenceScorer
from app.services.llm_extractor import LLMExtractor
from app.services.ocr_service import OCRService
from app.services.pdf_extractor import PDFExtractor
from app.services.pipeline import ExtractionPipeline
from app.services.text_normalizer import TextNormalizer
from app.services.validator import ChangeOrderValidator

SAMPLES_DIR = Path(__file__).parent / "samples"


@pytest.fixture
def sample_standard_text() -> str:
    """Loads realistic standard change order sample text."""
    path = SAMPLES_DIR / "change_order_1_standard.txt"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def sample_deduction_text() -> str:
    """Loads realistic deduction change order sample text."""
    path = SAMPLES_DIR / "change_order_2_deduction.txt"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def sample_conflict_text() -> str:
    """Loads sample text with conflicting data and arithmetic errors."""
    path = SAMPLES_DIR / "change_order_3_conflict.txt"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def sample_missing_fields_text() -> str:
    """Loads sample text with minimal fields and missing information."""
    path = SAMPLES_DIR / "change_order_4_missing_fields.txt"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def standard_digital_pdf_bytes(sample_standard_text: str) -> bytes:
    """Generates an in-memory single-page digital PDF containing standard change order text."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_text((50, 50), sample_standard_text, fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def multi_page_pdf_bytes() -> bytes:
    """Generates an in-memory 2-page digital PDF."""
    doc = pymupdf.open()
    # Page 1
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 50),
        "PROJECT: Highland Hospital Expansion\nCHANGE ORDER NO: CO-005\nDate: 2024-06-01",
        fontsize=11,
    )
    # Page 2
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 50),
        "Contract Sum will be increased by $35,000.00.\nSchedule impact: 10 days.\nSTATUS: Approved",
        fontsize=11,
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def scanned_image_pdf_bytes() -> bytes:
    """Generates a PDF containing an image with no digital text (simulating a scanned document)."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    # Create a simple PIL image containing text
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "SCANNED CHANGE ORDER CO-999\nCost Impact: $5000", fill=(0, 0, 0))

    import io

    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()

    rect = pymupdf.Rect(50, 50, 450, 250)
    page.insert_image(rect, stream=img_bytes)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def mock_standard_raw_extraction() -> RawChangeOrderExtraction:
    """Returns a mock RawChangeOrderExtraction matching sample 1."""
    return RawChangeOrderExtraction(
        project_name=RawFieldCandidate(
            value="Oakridge Medical Pavilion",
            evidence="PROJECT: Oakridge Medical Pavilion",
            page=1,
            is_inferred=False,
        ),
        project_address=RawFieldCandidate(
            value="1044 Highland Parkway, Denver, CO 80202",
            evidence="1044 Highland Parkway, Denver, CO 80202",
            page=1,
            is_inferred=False,
        ),
        change_order_number=RawFieldCandidate(
            value="CO-007",
            evidence="CHANGE ORDER NUMBER: CO-007",
            page=1,
            is_inferred=False,
        ),
        original_contract_number=RawFieldCandidate(
            value="CNT-2023-8891",
            evidence="ORIGINAL CONTRACT NUMBER: CNT-2023-8891",
            page=1,
            is_inferred=False,
        ),
        title=RawFieldCandidate(
            value="Electrical conduit rerouting due to unforeseen structural grade beam",
            evidence="Title: Electrical conduit rerouting due to unforeseen structural grade beam",
            page=1,
            is_inferred=False,
        ),
        description=RawFieldCandidate(
            value="Reroute electrical power feeds and branch circuits around newly exposed subsurface grade beam discovered during basement slab excavation, per RFI-084 response and Engineer Sketch SK-012. Includes core drilling, additional 3-inch rigid conduit, junction boxes, and pulling conductors.",
            evidence="Description: Reroute electrical power feeds and branch circuits around newly exposed subsurface grade beam discovered during basement slab excavation, per RFI-084 response and Engineer Sketch SK-012. Includes core drilling, additional 3-inch rigid conduit, junction boxes, and pulling conductors.",
            page=1,
            is_inferred=False,
        ),
        contractor_name=RawFieldCandidate(
            value="Apex Commercial Builders LLC",
            evidence="CONTRACTOR: Apex Commercial Builders LLC",
            page=1,
            is_inferred=False,
        ),
        owner_name=RawFieldCandidate(
            value="Colorado Health Properties Inc.",
            evidence="OWNER: Colorado Health Properties Inc.",
            page=1,
            is_inferred=False,
        ),
        issue_date=RawFieldCandidate(
            value="2024-05-15",
            evidence="DATE OF ISSUANCE: 2024-05-15",
            page=1,
            is_inferred=False,
        ),
        requested_by=RawFieldCandidate(
            value="Structural Engineer (Thornton Tomasetti)",
            evidence="Requested by: Structural Engineer (Thornton Tomasetti)",
            page=1,
            is_inferred=False,
        ),
        reason_for_change=RawFieldCandidate(
            value="Unforeseen existing structural conditions not shown on civil record drawings.",
            evidence="Reason for change: Unforeseen existing structural conditions not shown on civil record drawings.",
            page=1,
            is_inferred=False,
        ),
        cost_impact=RawCostImpactCandidate(
            amount=14500.0,
            currency="USD",
            impact_type=CostImpactType.ADDITION,
            evidence="The Contract Sum will be increased by this Change Order in the amount of: $14,500.00",
            page=1,
            is_inferred=False,
        ),
        schedule_impact_days=RawScheduleImpactCandidate(
            days=7,
            evidence="The Contract Time will be increased by: 7 calendar days",
            page=1,
            is_inferred=False,
        ),
        revised_contract_value=RawFieldCandidate(
            value="1309500.00",
            evidence="The new Contract Sum including this Change Order will be: $1,309,500.00",
            page=1,
            is_inferred=False,
        ),
        approval_status=RawFieldCandidate(
            value="approved",
            evidence="STATUS: Fully Executed and Approved",
            page=1,
            is_inferred=False,
        ),
        previous_contract_sum=1295000.0,
        detected_conflicts=[],
    )


@pytest.fixture
def mock_llm_extractor(mock_standard_raw_extraction: RawChangeOrderExtraction) -> LLMExtractor:
    """Provides a mocked LLMExtractor that returns standard extraction without network calls."""
    extractor = LLMExtractor()
    extractor.extract = MagicMock(return_value=mock_standard_raw_extraction)  # type: ignore
    return extractor


@pytest.fixture
def mock_pipeline(mock_llm_extractor: LLMExtractor) -> ExtractionPipeline:
    """Assembles a working ExtractionPipeline with mocked LLM."""
    return ExtractionPipeline(
        pdf_extractor=PDFExtractor(max_pages=50, ocr_min_char_density=50),
        ocr_service=OCRService(),
        text_normalizer=TextNormalizer(),
        llm_extractor=mock_llm_extractor,
        validator=ChangeOrderValidator(),
        confidence_scorer=ConfidenceScorer(),
    )


@pytest.fixture
def client(mock_pipeline: ExtractionPipeline) -> Generator[TestClient, None, None]:
    """TestClient with mocked pipeline dependency injected."""
    app.dependency_overrides[get_extraction_pipeline] = lambda: mock_pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
