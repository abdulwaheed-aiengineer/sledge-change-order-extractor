"""API request and health check schema definitions."""

from pydantic import BaseModel, Field


class TextExtractionRequest(BaseModel):
    """Payload for extracting structured data from raw text."""

    text: str = Field(
        min_length=1,
        description="Raw change-order document text content to be extracted.",
        examples=["CHANGE ORDER NO: 004\nProject: Metro Transit Center\nCost Impact: +$14,500.00"],
    )


class HealthResponse(BaseModel):
    """Health check diagnostic response."""

    status: str = Field(default="ok", description="Overall service status.")
    version: str = Field(default="0.1.0", description="Application version.")
    ocr_available: bool = Field(description="Indicates whether Tesseract OCR binary is reachable.")
    llm_configured: bool = Field(description="Indicates whether an OpenAI API key is set.")
    max_upload_size_bytes: int = Field(description="Maximum upload size allowed in bytes.")
    max_pdf_pages: int = Field(description="Maximum PDF pages allowed.")
