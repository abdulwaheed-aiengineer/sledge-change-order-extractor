"""Application configuration module using Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. If omitted, LLM service will require mock or explicit injection.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name with structured output support.",
    )
    openai_timeout_seconds: float = Field(
        default=60.0,
        description="Maximum seconds to wait for LLM completion before timing out.",
    )

    # Ingestion Constraints
    max_upload_size_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum allowable file upload size in bytes.",
    )
    max_pdf_pages: int = Field(
        default=50,
        description="Maximum allowed PDF pages per document.",
    )
    ocr_min_char_density: int = Field(
        default=50,
        description="Minimum character count per page below which OCR fallback is triggered.",
    )

    # OCR Configuration
    tesseract_cmd: str | None = Field(
        default=None,
        description="Optional absolute path to tesseract binary if not in system PATH.",
    )

    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host bind address.")
    port: int = Field(default=8000, description="Server port.")
    log_level: str = Field(default="INFO", description="Standard logging level.")
    debug: bool = Field(default=False, description="Debug mode flag.")


settings = Settings()
