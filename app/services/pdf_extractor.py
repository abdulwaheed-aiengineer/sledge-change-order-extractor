"""PDF extraction service using PyMuPDF."""

import logging
from dataclasses import dataclass

import pymupdf

logger = logging.getLogger("sledge.extractor.pdf")


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed or is corrupted."""


class PDFPageLimitExceededError(PDFProcessingError):
    """Raised when the document exceeds the maximum allowable page count."""


@dataclass
class ExtractedPage:
    """Represents text and metadata extracted from a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    char_count: int
    has_images: bool
    needs_ocr: bool


@dataclass
class PDFExtractionResult:
    """Aggregated output from PDF text extraction."""

    pages: list[ExtractedPage]
    total_pages: int
    is_scanned: bool
    needs_ocr_fallback: bool


class PDFExtractor:
    """Extracts text content and layout metadata from PDF documents using PyMuPDF."""

    def __init__(self, max_pages: int = 50, ocr_min_char_density: int = 50) -> None:
        self.max_pages = max_pages
        self.ocr_min_char_density = ocr_min_char_density

    def extract_from_bytes(self, pdf_bytes: bytes) -> PDFExtractionResult:
        """Parses PDF bytes, extracts digital text per page, and checks OCR readiness.

        Args:
            pdf_bytes: Raw bytes of the uploaded PDF file.

        Returns:
            PDFExtractionResult containing per-page text and OCR fallback flags.

        Raises:
            PDFProcessingError: If the document is corrupted or password-protected.
            PDFPageLimitExceededError: If the document exceeds max allowable pages.
        """
        if not pdf_bytes:
            raise PDFProcessingError("Empty PDF file provided.")

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.warning("Failed to parse PDF document bytes: %s", exc.__class__.__name__)
            raise PDFProcessingError(f"Malformed or unreadable PDF: {exc}") from exc

        try:
            if doc.is_encrypted:
                raise PDFProcessingError("Password-protected or encrypted PDFs are not supported.")

            total_pages = len(doc)
            if total_pages == 0:
                raise PDFProcessingError("PDF document contains no pages.")

            if total_pages > self.max_pages:
                raise PDFPageLimitExceededError(
                    f"Document has {total_pages} pages, which exceeds the limit of {self.max_pages} pages."
                )

            extracted_pages: list[ExtractedPage] = []
            pages_needing_ocr = 0

            for page_index in range(total_pages):
                page = doc[page_index]
                text = page.get_text("text") or ""
                clean_text = text.strip()
                char_count = len(clean_text)

                image_list = page.get_images(full=True)
                has_images = len(image_list) > 0

                # A page requires OCR fallback if digital text density is below threshold
                # and either has images or the document appears to be scanned.
                needs_ocr = char_count < self.ocr_min_char_density

                if needs_ocr:
                    pages_needing_ocr += 1

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_index + 1,
                        text=clean_text,
                        char_count=char_count,
                        has_images=has_images,
                        needs_ocr=needs_ocr,
                    )
                )

            # Document is considered scanned if the majority of pages need OCR
            is_scanned = (pages_needing_ocr / total_pages) >= 0.5
            needs_ocr_fallback = pages_needing_ocr > 0

            return PDFExtractionResult(
                pages=extracted_pages,
                total_pages=total_pages,
                is_scanned=is_scanned,
                needs_ocr_fallback=needs_ocr_fallback,
            )
        finally:
            doc.close()
