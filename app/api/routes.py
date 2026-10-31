"""FastAPI router endpoints for health check, text extraction, and PDF extraction."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import (
    get_extraction_pipeline,
    get_ocr_service,
    get_settings,
)
from app.config import Settings
from app.models.api_requests import HealthResponse, TextExtractionRequest
from app.models.change_order import ChangeOrderExtractionResponse
from app.services.llm_extractor import (
    LLMAuthenticationError,
    LLMExtractionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.services.ocr_service import OCRService
from app.services.pdf_extractor import (
    PDFPageLimitExceededError,
    PDFProcessingError,
)
from app.services.pipeline import ExtractionPipeline

logger = logging.getLogger("sledge.extractor.api")
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health and Diagnostic Status",
    description="Returns runtime status, OCR binary reachability, and document constraint limits.",
)
def health_check(
    cfg: Settings = Depends(get_settings),
    ocr_service: OCRService = Depends(get_ocr_service),
) -> HealthResponse:
    """Verifies service health and configuration readiness."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        ocr_available=ocr_service.is_available(),
        llm_configured=bool(cfg.openai_api_key and cfg.openai_api_key.strip()),
        max_upload_size_bytes=cfg.max_upload_size_bytes,
        max_pdf_pages=cfg.max_pdf_pages,
    )


@router.post(
    "/extract/text",
    response_model=ChangeOrderExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Change Order from Raw Text",
    description="Processes raw text or transcript of a construction change order and extracts structured fields.",
)
def extract_from_text(
    request: TextExtractionRequest,
    pipeline: ExtractionPipeline = Depends(get_extraction_pipeline),
) -> ChangeOrderExtractionResponse:
    """Extracts structured change-order data from raw document text."""
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided text input is empty or contains only whitespace.",
        )

    try:
        return pipeline.extract_from_text(request.text)
    except LLMAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM authentication failure: {exc}",
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Extraction timed out: {exc}",
        ) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"LLM provider rate limit exceeded: {exc}",
        ) from exc
    except LLMExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM extraction error: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unhandled error during text extraction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during extraction processing.",
        ) from exc


@router.post(
    "/extract/pdf",
    response_model=ChangeOrderExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Change Order from PDF Document",
    description="Processes uploaded PDF file using digital extraction or OCR fallback for scanned pages.",
)
async def extract_from_pdf(
    file: UploadFile = File(..., description="Change-order PDF file to extract."),
    cfg: Settings = Depends(get_settings),
    pipeline: ExtractionPipeline = Depends(get_extraction_pipeline),
) -> ChangeOrderExtractionResponse:
    """Extracts structured change-order data from uploaded PDF file."""
    # Verify file extension or content-type
    filename = file.filename or ""
    content_type = file.content_type or ""

    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{content_type}'. Uploaded file must be a PDF document.",
        )

    # Read file content safely, enforcing maximum size limit
    max_bytes = cfg.max_upload_size_bytes
    pdf_bytes = await file.read(max_bytes + 1)

    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowable size limit of {max_bytes / (1024 * 1024):.1f} MB.",
        )

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded PDF file is empty.",
        )

    try:
        return pipeline.extract_from_pdf(pdf_bytes)
    except PDFPageLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except PDFProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except LLMAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM authentication failure: {exc}",
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Extraction timed out: {exc}",
        ) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"LLM provider rate limit exceeded: {exc}",
        ) from exc
    except LLMExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM extraction error: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Unhandled error during PDF extraction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during document processing.",
        ) from exc
