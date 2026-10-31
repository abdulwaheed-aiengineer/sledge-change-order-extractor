"""Tests for FastAPI HTTP endpoints and error handling."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.dependencies import get_extraction_pipeline, get_settings
from app.config import Settings
from app.main import app
from app.services.llm_extractor import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.services.pipeline import ExtractionPipeline


def test_health_check_endpoint(client: TestClient) -> None:
    """Verifies GET /health returns service readiness and limits."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "ocr_available" in data
    assert "llm_configured" in data
    assert data["max_upload_size_bytes"] > 0
    assert data["max_pdf_pages"] > 0


def test_extract_text_endpoint_success(client: TestClient, sample_standard_text: str) -> None:
    """Verifies POST /extract/text returns 200 with structured response."""
    payload = {"text": sample_standard_text}
    response = client.post("/extract/text", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["document_status"] == "success"
    assert data["project_name"]["value"] == "Oakridge Medical Pavilion"
    assert data["change_order_number"]["value"] == "CO-007"
    assert data["cost_impact"]["value"]["amount"] == 14500.0


def test_extract_text_empty_input_returns_400(client: TestClient) -> None:
    """Verifies POST /extract/text returns 400 for empty or whitespace-only input."""
    # Whitespace only
    response = client.post("/extract/text", json={"text": "   \n\t  "})
    assert response.status_code == 400
    assert "empty or contains only whitespace" in response.json()["detail"]


def test_extract_pdf_endpoint_success(
    client: TestClient,
    standard_digital_pdf_bytes: bytes,
) -> None:
    """Verifies POST /extract/pdf successfully accepts and processes PDF upload."""
    files = {"file": ("change_order.pdf", standard_digital_pdf_bytes, "application/pdf")}
    response = client.post("/extract/pdf", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["source_type"] == "digital_pdf"
    assert data["change_order_number"]["value"] == "CO-007"


def test_extract_pdf_rejects_non_pdf_file(client: TestClient) -> None:
    """Verifies POST /extract/pdf returns 400 when an invalid file extension is uploaded."""
    files = {"file": ("document.txt", b"plain text", "text/plain")}
    response = client.post("/extract/pdf", files=files)

    assert response.status_code == 400
    assert "must be a PDF document" in response.json()["detail"]


def test_extract_pdf_enforces_size_limit(client: TestClient) -> None:
    """Verifies POST /extract/pdf returns 413 when file exceeds max upload limit."""
    # Override settings with a tiny 100-byte limit
    tiny_settings = Settings(max_upload_size_bytes=100)
    app.dependency_overrides[get_settings] = lambda: tiny_settings

    try:
        oversized_bytes = b"%PDF-1.4 " + b"X" * 200
        files = {"file": ("large.pdf", oversized_bytes, "application/pdf")}
        response = client.post("/extract/pdf", files=files)

        assert response.status_code == 413
        assert "exceeds maximum allowable size limit" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_settings, None)


def test_extract_pdf_malformed_bytes_returns_400(client: TestClient) -> None:
    """Verifies POST /extract/pdf returns 400 when PDF bytes are corrupted."""
    files = {"file": ("corrupt.pdf", b"corrupted bytes not a pdf", "application/pdf")}
    response = client.post("/extract/pdf", files=files)

    assert response.status_code == 400
    assert "Malformed or unreadable PDF" in response.json()["detail"]


def test_extract_text_llm_timeout_returns_504(client: TestClient) -> None:
    """Verifies that an LLM timeout triggers HTTP 504 Gateway Timeout."""
    mock_pipeline = MagicMock(spec=ExtractionPipeline)
    mock_pipeline.extract_from_text.side_effect = LLMTimeoutError("Request timed out.")
    app.dependency_overrides[get_extraction_pipeline] = lambda: mock_pipeline

    try:
        response = client.post("/extract/text", json={"text": "Some contract text"})
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_extraction_pipeline, None)


def test_extract_text_llm_auth_error_returns_500(client: TestClient) -> None:
    """Verifies that an LLM authentication failure triggers HTTP 500."""
    mock_pipeline = MagicMock(spec=ExtractionPipeline)
    mock_pipeline.extract_from_text.side_effect = LLMAuthenticationError("Invalid API key.")
    app.dependency_overrides[get_extraction_pipeline] = lambda: mock_pipeline

    try:
        response = client.post("/extract/text", json={"text": "Some contract text"})
        assert response.status_code == 500
        assert "authentication failure" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_extraction_pipeline, None)


def test_extract_text_rate_limit_returns_429(client: TestClient) -> None:
    """Verifies that LLM rate limit triggers HTTP 429 Too Many Requests."""
    mock_pipeline = MagicMock(spec=ExtractionPipeline)
    mock_pipeline.extract_from_text.side_effect = LLMRateLimitError("Rate limit reached.")
    app.dependency_overrides[get_extraction_pipeline] = lambda: mock_pipeline

    try:
        response = client.post("/extract/text", json={"text": "Some contract text"})
        assert response.status_code == 429
        assert "rate limit exceeded" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_extraction_pipeline, None)
