"""OCR Fallback Service using Tesseract OCR and Pillow."""

import io
import logging
import os
import shutil
from dataclasses import dataclass

import pymupdf
import pytesseract
from PIL import Image

logger = logging.getLogger("sledge.extractor.ocr")


@dataclass
class OCRPageResult:
    """Result of performing OCR on a single document page or image."""

    page_number: int
    text: str
    confidence: float  # 0.0 to 1.0 (mean word confidence where measurable)
    is_successful: bool
    warning: str | None = None


class OCRService:
    """Service handling OCR execution and quality metrics via Tesseract and Pillow."""

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        self.tesseract_cmd = tesseract_cmd or self._find_tesseract_binary()
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def _find_tesseract_binary(self) -> str | None:
        """Attempts to locate the tesseract executable on the host system."""
        # Check standard PATH
        in_path = shutil.which("tesseract")
        if in_path:
            return in_path

        # Standard Windows installation paths
        common_windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\Tesseract-OCR\tesseract.exe"),
        ]
        for path in common_windows_paths:
            if os.path.isfile(path):
                return path

        return None

    def is_available(self) -> bool:
        """Checks if Tesseract binary is callable on the host system."""
        if not self.tesseract_cmd and not shutil.which("tesseract"):
            return False
        try:
            version = pytesseract.get_tesseract_version()
            return bool(version)
        except Exception:
            return False

    def ocr_page(self, pdf_bytes: bytes, page_number: int) -> OCRPageResult:
        """Renders a PDF page to image and runs Tesseract OCR.

        Args:
            pdf_bytes: Raw PDF bytes.
            page_number: 1-indexed page number to render and scan.

        Returns:
            OCRPageResult with extracted text, measured confidence, and warnings if any.
        """
        if not self.is_available():
            msg = "OCR fallback requested but Tesseract is not installed or available on the host system."
            logger.warning(msg)
            return OCRPageResult(
                page_number=page_number,
                text="",
                confidence=0.0,
                is_successful=False,
                warning=msg,
            )

        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            try:
                page_idx = page_number - 1
                if page_idx < 0 or page_idx >= len(doc):
                    return OCRPageResult(
                        page_number=page_number,
                        text="",
                        confidence=0.0,
                        is_successful=False,
                        warning=f"Invalid page index {page_number} for PDF with {len(doc)} pages.",
                    )

                page = doc[page_idx]
                # Render to 300 DPI for high-quality OCR recognition
                pixmap = page.get_pixmap(dpi=300)
                image_bytes = pixmap.tobytes("png")
                image = Image.open(io.BytesIO(image_bytes))

                return self.ocr_image(image, page_number=page_number)
            finally:
                doc.close()
        except Exception as exc:
            logger.error("OCR rendering failed for page %d: %s", page_number, exc)
            return OCRPageResult(
                page_number=page_number,
                text="",
                confidence=0.0,
                is_successful=False,
                warning=f"OCR execution failed on page {page_number}: {exc}",
            )

    def ocr_image(self, image: Image.Image, page_number: int = 1) -> OCRPageResult:
        """Runs Tesseract on a Pillow Image object, calculating word-level confidence metrics."""
        try:
            # Extract detailed word-level data to compute empirical confidence
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            words: list[str] = []
            confidences: list[float] = []

            for word, conf in zip(data["text"], data["conf"], strict=False):
                clean_word = word.strip()
                if clean_word and conf != -1 and conf != "-1":
                    try:
                        numeric_conf = float(conf)
                        # Valid confidence is 0 to 100
                        if numeric_conf >= 0:
                            words.append(clean_word)
                            confidences.append(numeric_conf)
                    except ValueError:
                        continue

            text = " ".join(words).strip()
            # If word-level text is sparse, also get full string format
            if not text:
                text = pytesseract.image_to_string(image).strip()

            # Calculate measurable mean OCR confidence normalized to 0.0 - 1.0
            if confidences:
                mean_conf = (sum(confidences) / len(confidences)) / 100.0
                confidence = max(0.0, min(1.0, round(mean_conf, 4)))
            else:
                confidence = 0.5 if text else 0.0

            return OCRPageResult(
                page_number=page_number,
                text=text,
                confidence=confidence,
                is_successful=True,
            )
        except Exception as exc:
            logger.error("Pytesseract processing error: %s", exc)
            return OCRPageResult(
                page_number=page_number,
                text="",
                confidence=0.0,
                is_successful=False,
                warning=f"Tesseract processing error: {exc}",
            )
