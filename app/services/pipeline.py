"""End-to-end extraction pipeline coordinating PDF, OCR, LLM, validation, and scoring services."""

import logging

from app.models.change_order import (
    ChangeOrderExtractionResponse,
    RawChangeOrderExtraction,
    RawFieldCandidate,
)
from app.models.enums import DocumentStatus, SourceType
from app.models.extracted_field import ExtractedField
from app.services.confidence_scorer import ConfidenceScorer
from app.services.llm_extractor import LLMExtractor
from app.services.ocr_service import OCRService
from app.services.pdf_extractor import PDFExtractor, PDFProcessingError
from app.services.text_normalizer import TextNormalizer
from app.services.validator import ChangeOrderValidator, ValidationResult

logger = logging.getLogger("sledge.extractor.pipeline")


class ExtractionPipeline:
    """Orchestrates the entire change order extraction lifecycle."""

    def __init__(
        self,
        pdf_extractor: PDFExtractor,
        ocr_service: OCRService,
        text_normalizer: TextNormalizer,
        llm_extractor: LLMExtractor,
        validator: ChangeOrderValidator,
        confidence_scorer: ConfidenceScorer,
    ) -> None:
        self.pdf_extractor = pdf_extractor
        self.ocr_service = ocr_service
        self.text_normalizer = text_normalizer
        self.llm_extractor = llm_extractor
        self.validator = validator
        self.confidence_scorer = confidence_scorer

    def extract_from_text(self, text: str) -> ChangeOrderExtractionResponse:
        """Extracts structured change-order data from raw text."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only.")

        normalized_doc = self.text_normalizer.normalize_raw_text(text)
        return self._run_extraction_pipeline(
            document_text=normalized_doc.full_text,
            source_type=SourceType.RAW_TEXT,
            ocr_confidence=None,
            initial_warnings=[],
        )

    def extract_from_pdf(self, pdf_bytes: bytes) -> ChangeOrderExtractionResponse:
        """Extracts structured change-order data from PDF bytes using digital or OCR extraction."""
        pdf_result = self.pdf_extractor.extract_from_bytes(pdf_bytes)
        warnings: list[str] = []
        page_texts: list[tuple[int, str]] = []
        ocr_confidences: list[float] = []

        for page in pdf_result.pages:
            if page.needs_ocr:
                logger.info("Page %d has low text density; running OCR fallback.", page.page_number)
                ocr_res = self.ocr_service.ocr_page(pdf_bytes, page_number=page.page_number)
                if ocr_res.is_successful and ocr_res.text:
                    page_texts.append((page.page_number, ocr_res.text))
                    ocr_confidences.append(ocr_res.confidence)
                else:
                    # Keep whatever partial digital text existed, and log warning
                    page_texts.append((page.page_number, page.text))
                    if ocr_res.warning:
                        warnings.append(ocr_res.warning)
            else:
                page_texts.append((page.page_number, page.text))

        mean_ocr_conf = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None
        source_type = SourceType.SCANNED_PDF if pdf_result.is_scanned else SourceType.DIGITAL_PDF

        normalized_doc = self.text_normalizer.normalize_pages(page_texts)
        if not normalized_doc.full_text.strip():
            raise PDFProcessingError(
                "No readable text could be extracted from the PDF document via digital extraction or OCR."
            )

        return self._run_extraction_pipeline(
            document_text=normalized_doc.full_text,
            source_type=source_type,
            ocr_confidence=mean_ocr_conf,
            initial_warnings=warnings,
        )

    def _run_extraction_pipeline(
        self,
        document_text: str,
        source_type: SourceType,
        ocr_confidence: float | None,
        initial_warnings: list[str],
    ) -> ChangeOrderExtractionResponse:
        """Core pipeline workflow for structured LLM extraction, validation, and scoring."""
        warnings: list[str] = list(initial_warnings)

        # 1. LLM Extraction
        raw_extraction: RawChangeOrderExtraction = self.llm_extractor.extract(document_text)

        # 2. Validation
        val_result: ValidationResult = self.validator.validate_extraction(raw_extraction)
        warnings.extend(val_result.warnings)

        # 3. Field Building & Heuristic Confidence Scoring
        has_conflicts = len(raw_extraction.detected_conflicts) > 0 or any(
            "discrepancy" in w.lower() for w in val_result.warnings
        )

        def build_string_field(field_name: str) -> ExtractedField[str]:
            candidate: RawFieldCandidate = getattr(raw_extraction, field_name)
            raw_val = candidate.value.strip() if candidate.value else None
            is_valid = val_result.field_validation_flags.get(field_name, True)
            conf = self.confidence_scorer.compute_field_confidence(
                field_name=field_name,
                value=raw_val,
                raw_candidate=candidate,
                document_text=document_text,
                is_valid=is_valid,
                has_conflict=has_conflicts,
                ocr_confidence=ocr_confidence,
            )
            return ExtractedField[str](
                value=raw_val,
                confidence=conf,
                evidence=candidate.evidence if raw_val is not None else None,
                page=candidate.page if raw_val is not None else None,
            )

        # Build issue_date using normalized value
        date_candidate = raw_extraction.issue_date
        date_valid = val_result.field_validation_flags.get("issue_date", True)
        date_conf = self.confidence_scorer.compute_field_confidence(
            field_name="issue_date",
            value=val_result.normalized_date,
            raw_candidate=date_candidate,
            document_text=document_text,
            is_valid=date_valid,
            has_conflict=has_conflicts,
            ocr_confidence=ocr_confidence,
        )
        issue_date_field = ExtractedField[str](
            value=val_result.normalized_date,
            confidence=date_conf,
            evidence=date_candidate.evidence if val_result.normalized_date else None,
            page=date_candidate.page if val_result.normalized_date else None,
        )

        # Build cost_impact using normalized CostImpact model
        cost_candidate = raw_extraction.cost_impact
        cost_valid = val_result.field_validation_flags.get(
            "cost_impact", True
        ) and val_result.field_validation_flags.get("cost_impact_consistency", True)
        cost_conf = self.confidence_scorer.compute_field_confidence(
            field_name="cost_impact",
            value=val_result.normalized_cost,
            raw_candidate=cost_candidate,
            document_text=document_text,
            is_valid=cost_valid,
            has_conflict=has_conflicts
            or not val_result.field_validation_flags.get("cost_impact_consistency", True),
            ocr_confidence=ocr_confidence,
        )
        cost_impact_field = ExtractedField(
            value=val_result.normalized_cost,
            confidence=cost_conf,
            evidence=cost_candidate.evidence if val_result.normalized_cost else None,
            page=cost_candidate.page if val_result.normalized_cost else None,
        )

        # Build schedule_impact_days
        sched_candidate = raw_extraction.schedule_impact_days
        sched_valid = val_result.field_validation_flags.get("schedule_impact_days", True)
        sched_conf = self.confidence_scorer.compute_field_confidence(
            field_name="schedule_impact_days",
            value=val_result.normalized_schedule_days,
            raw_candidate=sched_candidate,
            document_text=document_text,
            is_valid=sched_valid,
            has_conflict=has_conflicts,
            ocr_confidence=ocr_confidence,
        )
        schedule_impact_field = ExtractedField[int](
            value=val_result.normalized_schedule_days,
            confidence=sched_conf,
            evidence=sched_candidate.evidence
            if val_result.normalized_schedule_days is not None
            else None,
            page=sched_candidate.page if val_result.normalized_schedule_days is not None else None,
        )

        # Build revised_contract_value
        revised_candidate = raw_extraction.revised_contract_value
        revised_valid = val_result.field_validation_flags.get("revised_contract_value", True)
        revised_conf = self.confidence_scorer.compute_field_confidence(
            field_name="revised_contract_value",
            value=val_result.normalized_revised_contract,
            raw_candidate=revised_candidate,
            document_text=document_text,
            is_valid=revised_valid,
            has_conflict=has_conflicts or not revised_valid,
            ocr_confidence=ocr_confidence,
        )
        revised_contract_field = ExtractedField[float](
            value=val_result.normalized_revised_contract,
            confidence=revised_conf,
            evidence=revised_candidate.evidence
            if val_result.normalized_revised_contract is not None
            else None,
            page=revised_candidate.page
            if val_result.normalized_revised_contract is not None
            else None,
        )

        # Build approval_status
        approval_candidate = raw_extraction.approval_status
        approval_valid = val_result.field_validation_flags.get("approval_status", True)
        approval_conf = self.confidence_scorer.compute_field_confidence(
            field_name="approval_status",
            value=val_result.normalized_approval_status,
            raw_candidate=approval_candidate,
            document_text=document_text,
            is_valid=approval_valid,
            has_conflict=has_conflicts,
            ocr_confidence=ocr_confidence,
        )
        approval_status_field = ExtractedField(
            value=val_result.normalized_approval_status,
            confidence=approval_conf,
            evidence=approval_candidate.evidence if val_result.normalized_approval_status else None,
            page=approval_candidate.page if val_result.normalized_approval_status else None,
        )

        # Determine overall document status
        # If critical fields (change_order_number or project_name or cost_impact) are missing, or there are warnings:
        has_critical_missing = (
            raw_extraction.change_order_number.value is None
            or raw_extraction.project_name.value is None
            or raw_extraction.cost_impact.amount is None
        )
        if has_critical_missing or len(warnings) > 0:
            doc_status = DocumentStatus.PARTIAL_SUCCESS
        else:
            doc_status = DocumentStatus.SUCCESS

        return ChangeOrderExtractionResponse(
            document_status=doc_status,
            source_type=source_type,
            extraction_warnings=warnings,
            project_name=build_string_field("project_name"),
            project_address=build_string_field("project_address"),
            change_order_number=build_string_field("change_order_number"),
            original_contract_number=build_string_field("original_contract_number"),
            title=build_string_field("title"),
            description=build_string_field("description"),
            contractor_name=build_string_field("contractor_name"),
            owner_name=build_string_field("owner_name"),
            issue_date=issue_date_field,
            requested_by=build_string_field("requested_by"),
            reason_for_change=build_string_field("reason_for_change"),
            cost_impact=cost_impact_field,
            schedule_impact_days=schedule_impact_field,
            revised_contract_value=revised_contract_field,
            approval_status=approval_status_field,
        )
