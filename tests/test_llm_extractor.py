"""Tests for OpenAI structured output extraction service and error handling."""

from unittest.mock import MagicMock

import pytest
from openai import APITimeoutError, AuthenticationError, RateLimitError

from app.models.change_order import RawChangeOrderExtraction, RawFieldCandidate
from app.services.llm_extractor import (
    LLMAuthenticationError,
    LLMExtractionError,
    LLMExtractor,
    LLMRateLimitError,
    LLMTimeoutError,
)


def test_missing_client_raises_auth_error() -> None:
    """Verifies that attempting extraction without configured credentials raises LLMAuthenticationError."""
    extractor = LLMExtractor(api_key=None, client=None)
    with pytest.raises(LLMAuthenticationError):
        extractor.extract("Dummy document text.")


def test_successful_structured_extraction() -> None:
    """Verifies that structured response parsing maps to RawChangeOrderExtraction correctly."""
    mock_client = MagicMock()
    mock_parsed_data = RawChangeOrderExtraction(
        project_name=RawFieldCandidate(value="Metro Station", evidence="Project: Metro Station"),
    )

    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.parsed = mock_parsed_data

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.beta.chat.completions.parse.return_value = mock_response

    extractor = LLMExtractor(client=mock_client)
    result = extractor.extract("Project: Metro Station")

    assert isinstance(result, RawChangeOrderExtraction)
    assert result.project_name.value == "Metro Station"
    assert result.project_name.evidence == "Project: Metro Station"


def test_timeout_error_handling() -> None:
    """Verifies that APITimeoutError is caught and raised as LLMTimeoutError."""
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = APITimeoutError(request=MagicMock())

    extractor = LLMExtractor(client=mock_client, timeout_seconds=10.0)
    with pytest.raises(LLMTimeoutError) as exc_info:
        extractor.extract("Doc text")

    assert "timed out after 10.0s" in str(exc_info.value)


def test_authentication_error_handling() -> None:
    """Verifies that OpenAI AuthenticationError maps to LLMAuthenticationError."""
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = AuthenticationError(
        message="Invalid API Key",
        response=MagicMock(),
        body={},
    )

    extractor = LLMExtractor(client=mock_client)
    with pytest.raises(LLMAuthenticationError):
        extractor.extract("Doc text")


def test_rate_limit_error_handling() -> None:
    """Verifies that RateLimitError maps to LLMRateLimitError."""
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(),
        body={},
    )

    extractor = LLMExtractor(client=mock_client)
    with pytest.raises(LLMRateLimitError):
        extractor.extract("Doc text")


def test_null_parsed_response_handling() -> None:
    """Verifies handling when the LLM refuses or returns empty parsed data."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.parsed = None
    mock_choice.message.refusal = "Safety policy refusal."

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.beta.chat.completions.parse.return_value = mock_response

    extractor = LLMExtractor(client=mock_client)
    with pytest.raises(LLMExtractionError) as exc_info:
        extractor.extract("Doc text")

    assert "did not return structured data" in str(exc_info.value)


def test_token_usage_and_cost_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that token usage and cost calculation are logged accurately."""
    mock_client = MagicMock()
    mock_parsed_data = RawChangeOrderExtraction(
        project_name=RawFieldCandidate(value="Airport Terminal"),
    )

    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.parsed = mock_parsed_data

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 1000
    mock_usage.completion_tokens = 500
    mock_usage.total_tokens = 1500

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client.beta.chat.completions.parse.return_value = mock_response

    extractor = LLMExtractor(client=mock_client, model="gpt-4o-mini")
    with caplog.at_level("INFO", logger="sledge.extractor.llm"):
        extractor.extract("Airport Terminal expansion")

    # gpt-4o-mini: 1000 * 0.150/1M + 500 * 0.600/1M = 0.00015 + 0.00030 = 0.000450
    assert any("LLM Usage & Cost [gpt-4o-mini]" in record.message for record in caplog.records)
    assert any("$0.000450 USD" in record.message for record in caplog.records)

