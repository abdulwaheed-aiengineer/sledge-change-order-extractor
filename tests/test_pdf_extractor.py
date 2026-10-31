"""Tests for PyMuPDF PDF extraction service."""

import pymupdf
import pytest

from app.services.pdf_extractor import (
    PDFExtractionResult,
    PDFExtractor,
    PDFPageLimitExceededError,
    PDFProcessingError,
)


def test_extract_digital_pdf_standard(standard_digital_pdf_bytes: bytes) -> None:
    """Verifies text extraction from a standard single-page digital PDF."""
    extractor = PDFExtractor(max_pages=50, ocr_min_char_density=50)
    result = extractor.extract_from_bytes(standard_digital_pdf_bytes)

    assert isinstance(result, PDFExtractionResult)
    assert result.total_pages == 1
    assert not result.is_scanned
    assert not result.needs_ocr_fallback
    assert len(result.pages) == 1

    page = result.pages[0]
    assert page.page_number == 1
    assert "Oakridge Medical Pavilion" in page.text
    assert "CO-007" in page.text
    assert page.char_count > 100
    assert not page.needs_ocr


def test_extract_multi_page_pdf(multi_page_pdf_bytes: bytes) -> None:
    """Verifies that multi-page documents maintain page-level order and boundaries."""
    extractor = PDFExtractor(max_pages=50, ocr_min_char_density=20)
    result = extractor.extract_from_bytes(multi_page_pdf_bytes)

    assert result.total_pages == 2
    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert "Highland Hospital" in result.pages[0].text
    assert result.pages[1].page_number == 2
    assert "increased by $35,000.00" in result.pages[1].text


def test_ocr_fallback_triggered_on_scanned_pdf(scanned_image_pdf_bytes: bytes) -> None:
    """Verifies that an image-only scanned PDF is flagged for OCR fallback."""
    extractor = PDFExtractor(max_pages=50, ocr_min_char_density=50)
    result = extractor.extract_from_bytes(scanned_image_pdf_bytes)

    assert result.total_pages == 1
    assert result.pages[0].char_count == 0
    assert result.pages[0].has_images
    assert result.pages[0].needs_ocr
    assert result.needs_ocr_fallback
    assert result.is_scanned


def test_pdf_page_limit_enforcement() -> None:
    """Verifies that PDFs exceeding the max_pages limit raise an exception."""
    doc = pymupdf.open()
    for _ in range(5):
        doc.new_page()
    five_page_bytes = doc.tobytes()
    doc.close()

    extractor = PDFExtractor(max_pages=3)
    with pytest.raises(PDFPageLimitExceededError) as exc_info:
        extractor.extract_from_bytes(five_page_bytes)

    assert "exceeds the limit of 3 pages" in str(exc_info.value)


def test_empty_pdf_bytes_handling() -> None:
    """Verifies that empty bytes input raises PDFProcessingError."""
    extractor = PDFExtractor()
    with pytest.raises(PDFProcessingError) as exc_info:
        extractor.extract_from_bytes(b"")

    assert "Empty PDF file provided" in str(exc_info.value)


def test_malformed_pdf_bytes_handling() -> None:
    """Verifies that corrupted bytes input raises PDFProcessingError."""
    extractor = PDFExtractor()
    with pytest.raises(PDFProcessingError) as exc_info:
        extractor.extract_from_bytes(b"NOT_A_REAL_PDF_DATA_STREAM")

    assert "Malformed or unreadable PDF" in str(exc_info.value)
