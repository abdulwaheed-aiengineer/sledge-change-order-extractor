"""Enumeration definitions for change-order extraction models."""

from enum import StrEnum


class DocumentStatus(StrEnum):
    """Overall document processing and extraction status."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class SourceType(StrEnum):
    """Source medium or ingestion pathway of the document."""

    DIGITAL_PDF = "digital_pdf"
    SCANNED_PDF = "scanned_pdf"
    RAW_TEXT = "raw_text"


class CostImpactType(StrEnum):
    """Categorization of financial adjustment in the change order."""

    ADDITION = "addition"
    DEDUCTION = "deduction"
    NO_COST = "no_cost"


class ApprovalStatus(StrEnum):
    """Workflow approval state of the change order document."""

    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"
    DRAFT = "draft"
    UNKNOWN = "unknown"
