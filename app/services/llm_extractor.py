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

# Pricing per 1,000,000 tokens in USD (Standard OpenAI rates)
MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.150, "completion": 0.600},
    "gpt-4o": {"prompt": 2.500, "completion": 10.000},
    "gpt-4o-2024-08-06": {"prompt": 2.500, "completion": 10.000},
    "gpt-4-turbo": {"prompt": 10.000, "completion": 30.000},
    "gpt-3.5-turbo": {"prompt": 0.500, "completion": 1.500},
}



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

    def _log_token_usage_and_cost(self, usage: Any) -> None:
        """Logs token usage breakdown and calculates estimated cost in USD."""
        if not usage:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if not isinstance(prompt_tokens, (int, float)) or not isinstance(completion_tokens, (int, float)):
            return

        total = int(total_tokens) if isinstance(total_tokens, (int, float)) else int(prompt_tokens + completion_tokens)
        rates = MODEL_PRICING_PER_1M.get(self.model, MODEL_PRICING_PER_1M["gpt-4o-mini"])
        cost = (prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]) / 1_000_000.0

        logger.info(
            "LLM Usage & Cost [%s]: %d prompt + %d completion = %d total tokens | Est. Cost: $%.6f USD",
            self.model,
            int(prompt_tokens),
            int(completion_tokens),
            total,
            cost,
        )

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

            # Log token usage and cost per request
            self._log_token_usage_and_cost(getattr(response, "usage", None))

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
