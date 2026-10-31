"""LLM Extraction service leveraging OpenAI structured outputs."""

import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from app.models.change_order import RawChangeOrderExtraction
from app.prompts.extraction_prompt import EXTRACTION_SYSTEM_PROMPT, format_user_prompt

logger = logging.getLogger("sledge.extractor.llm")


class LLMExtractionError(Exception):
    """Base exception for LLM extraction failures."""


class LLMAuthenticationError(LLMExtractionError):
    """Raised when the OpenAI API key is missing or invalid."""


class LLMTimeoutError(LLMExtractionError):
    """Raised when the LLM API request times out."""


class LLMRateLimitError(LLMExtractionError):
    """Raised when OpenAI rate limit is exceeded."""


class LLMExtractor:
    """Extracts structured change-order information using OpenAI structured outputs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds

        if client is not None:
            self.client = client
        elif api_key:
            self.client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        else:
            self.client = None

    def extract(self, document_text: str) -> RawChangeOrderExtraction:
        """Invokes OpenAI with structured outputs to extract change-order data.

        Args:
            document_text: Normalized document text including page markers.

        Returns:
            RawChangeOrderExtraction parsed and validated against Pydantic schema.

        Raises:
            LLMAuthenticationError: If API credentials are not configured or invalid.
            LLMTimeoutError: If the call times out.
            LLMRateLimitError: If rate limited.
            LLMExtractionError: If parsing or generation fails.
        """
        if not self.client:
            raise LLMAuthenticationError(
                "OpenAI client is not configured. Provide OPENAI_API_KEY in environment or inject client."
            )

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(document_text)},
        ]

        logger.info(
            "Sending extraction request to LLM (model=%s, doc_chars=%d)",
            self.model,
            len(document_text),
        )

        try:
            # Use OpenAI beta structured outputs with Pydantic model
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=RawChangeOrderExtraction,
                timeout=self.timeout_seconds,
            )

            choice = response.choices[0]
            if choice.finish_reason == "length":
                logger.warning("LLM response truncated due to context length limit.")

            parsed_extraction: RawChangeOrderExtraction | None = choice.message.parsed
            if not parsed_extraction:
                refusal = getattr(choice.message, "refusal", None)
                raise LLMExtractionError(f"LLM did not return structured data. Refusal: {refusal}")

            return parsed_extraction

        except AuthenticationError as exc:
            logger.error("OpenAI authentication error: %s", exc)
            raise LLMAuthenticationError(f"Authentication failure: {exc}") from exc

        except APITimeoutError as exc:
            logger.error("OpenAI request timed out after %.1fs", self.timeout_seconds)
            raise LLMTimeoutError(
                f"OpenAI extraction timed out after {self.timeout_seconds}s"
            ) from exc

        except RateLimitError as exc:
            logger.error("OpenAI rate limit encountered: %s", exc)
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {exc}") from exc

        except APIConnectionError as exc:
            logger.error("OpenAI connection error: %s", exc)
            raise LLMExtractionError(f"Failed to connect to OpenAI API: {exc}") from exc

        except APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise LLMExtractionError(f"OpenAI API returned an error: {exc}") from exc

        except Exception as exc:
            logger.error("Unexpected error during LLM extraction: %s", exc)
            raise LLMExtractionError(f"LLM extraction error: {exc}") from exc
