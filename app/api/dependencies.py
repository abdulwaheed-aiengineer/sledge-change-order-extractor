"""FastAPI dependency injection providers."""

from functools import lru_cache

from fastapi import Depends

from app.config import Settings, settings
from app.services.confidence_scorer import ConfidenceScorer
from app.services.llm_extractor import LLMExtractor
from app.services.ocr_service import OCRService
from app.services.pdf_extractor import PDFExtractor
from app.services.pipeline import ExtractionPipeline
from app.services.text_normalizer import TextNormalizer
from app.services.validator import ChangeOrderValidator


@lru_cache
def get_settings() -> Settings:
    """Provides application configuration settings."""
    return settings


def get_pdf_extractor(
    cfg: Settings = Depends(get_settings),
) -> PDFExtractor:
    """Provides PDFExtractor instance."""
    return PDFExtractor(
        max_pages=cfg.max_pdf_pages,
        ocr_min_char_density=cfg.ocr_min_char_density,
    )


def get_ocr_service(
    cfg: Settings = Depends(get_settings),
) -> OCRService:
    """Provides OCRService instance."""
    return OCRService(tesseract_cmd=cfg.tesseract_cmd)


@lru_cache
def get_text_normalizer() -> TextNormalizer:
    """Provides singleton TextNormalizer instance."""
    return TextNormalizer()


@lru_cache
def get_validator() -> ChangeOrderValidator:
    """Provides singleton ChangeOrderValidator instance."""
    return ChangeOrderValidator()


@lru_cache
def get_confidence_scorer() -> ConfidenceScorer:
    """Provides singleton ConfidenceScorer instance."""
    return ConfidenceScorer()


def get_llm_extractor(
    cfg: Settings = Depends(get_settings),
) -> LLMExtractor:
    """Provides LLMExtractor instance with configured OpenAI credentials."""
    return LLMExtractor(
        api_key=cfg.openai_api_key,
        model=cfg.openai_model,
        timeout_seconds=cfg.openai_timeout_seconds,
    )


def get_extraction_pipeline(
    pdf_extractor: PDFExtractor = Depends(get_pdf_extractor),
    ocr_service: OCRService = Depends(get_ocr_service),
    text_normalizer: TextNormalizer = Depends(get_text_normalizer),
    llm_extractor: LLMExtractor = Depends(get_llm_extractor),
    validator: ChangeOrderValidator = Depends(get_validator),
    confidence_scorer: ConfidenceScorer = Depends(get_confidence_scorer),
) -> ExtractionPipeline:
    """Assembles the full ExtractionPipeline orchestrator."""
    return ExtractionPipeline(
        pdf_extractor=pdf_extractor,
        ocr_service=ocr_service,
        text_normalizer=text_normalizer,
        llm_extractor=llm_extractor,
        validator=validator,
        confidence_scorer=confidence_scorer,
    )
