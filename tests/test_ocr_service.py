"""Tests for OCR fallback service and empirical OCR confidence calculation."""

from unittest.mock import patch

import pytesseract
from PIL import Image

from app.services.ocr_service import OCRPageResult, OCRService


def test_ocr_service_handles_missing_tesseract() -> None:
    """Verifies graceful degradation when Tesseract binary is absent."""
    service = OCRService(tesseract_cmd="/nonexistent/path/to/tesseract")
    with patch.object(service, "is_available", return_value=False):
        result = service.ocr_page(pdf_bytes=b"dummy", page_number=1)

        assert isinstance(result, OCRPageResult)
        assert not result.is_successful
        assert result.confidence == 0.0
        assert result.text == ""
        assert "not installed or available" in (result.warning or "")


def test_ocr_confidence_calculation_with_word_data() -> None:
    """Verifies that empirical OCR confidence is computed as mean word-level confidence."""
    service = OCRService()

    fake_ocr_data = {
        "text": ["CHANGE", "ORDER", "", "CO-101", "   "],
        "conf": [90, 80, -1, 70, -1],
    }

    img = Image.new("RGB", (100, 50), color=(255, 255, 255))
    with (
        patch("pytesseract.image_to_data", return_value=fake_ocr_data),
        patch("pytesseract.image_to_string", return_value="CHANGE ORDER CO-101"),
    ):
        result = service.ocr_image(img, page_number=1)

        assert result.is_successful
        assert "CHANGE ORDER CO-101" in result.text
        # Mean of [90, 80, 70] = 80 -> normalized to 0.80
        assert result.confidence == 0.80


def test_ocr_page_invalid_page_index(standard_digital_pdf_bytes: bytes) -> None:
    """Verifies error handling when requesting an out-of-bounds page index."""
    service = OCRService()
    with patch.object(service, "is_available", return_value=True):
        result = service.ocr_page(standard_digital_pdf_bytes, page_number=99)

        assert not result.is_successful
        assert "Invalid page index 99" in (result.warning or "")


def test_ocr_image_exception_handling() -> None:
    """Verifies that unexpected pytesseract exceptions are captured in warning."""
    service = OCRService()
    img = Image.new("RGB", (50, 50))

    with patch("pytesseract.image_to_data", side_effect=pytesseract.TesseractError(1, "OCR Crash")):
        result = service.ocr_image(img, page_number=1)

        assert not result.is_successful
        assert result.confidence == 0.0
        assert "Tesseract processing error" in (result.warning or "")
