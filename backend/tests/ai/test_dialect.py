"""Parameter negotiation against an OpenAI-compatible endpoint.

The first real gpt-5.6-luna call failed with exactly the error reproduced
below, so these cases are regressions, not hypotheticals.
"""

from __future__ import annotations

import pytest

from app.ai.dialect import EndpointDialect, rejected_parameter


class FakeAPIError(Exception):
    """Shaped like openai's BadRequestError: message text plus a `body`."""

    def __init__(self, message: str, body: dict | None = None) -> None:
        super().__init__(message)
        self.body = body


def _unsupported_param_error(param: str) -> FakeAPIError:
    return FakeAPIError(
        f"Error code: 400 - Unsupported parameter: '{param}' is not supported with this model.",
        {
            "error": {
                "message": f"Unsupported parameter: '{param}' is not supported with this model.",
                "type": "invalid_request_error",
                "param": param,
                "code": "unsupported_parameter",
            }
        },
    )


def _unsupported_temperature_error() -> FakeAPIError:
    return FakeAPIError(
        "Error code: 400 - Unsupported value: 'temperature' does not support 0.0 with this model.",
        {
            "error": {
                "message": "Unsupported value: 'temperature' does not support 0.0.",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": "unsupported_value",
            }
        },
    )


def test_reads_the_offending_parameter_from_the_error_body():
    assert rejected_parameter(_unsupported_param_error("max_tokens")) == "max_tokens"
    assert rejected_parameter(_unsupported_temperature_error()) == "temperature"


def test_falls_back_to_the_message_when_no_structured_body():
    exc = FakeAPIError(
        "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'max_tokens' is not "
        "supported with this model. Use 'max_completion_tokens' instead.\"}}"
    )
    assert rejected_parameter(exc) == "max_tokens"


def test_unrelated_errors_yield_no_parameter():
    assert rejected_parameter(FakeAPIError("Error code: 401 - invalid api key")) is None
    assert rejected_parameter(FakeAPIError("Connection error.")) is None


def test_defaults_to_the_modern_token_parameter():
    kwargs: dict = {}
    EndpointDialect().apply(kwargs, max_tokens=200, temperature=0.5)
    assert kwargs == {"max_completion_tokens": 200, "temperature": 0.5}


def test_flips_to_legacy_max_tokens_when_the_modern_name_is_rejected():
    dialect = EndpointDialect()
    assert dialect.adapt(_unsupported_param_error("max_completion_tokens")) is True
    kwargs: dict = {}
    dialect.apply(kwargs, max_tokens=200, temperature=0.5)
    assert "max_tokens" in kwargs
    assert "max_completion_tokens" not in kwargs


def test_flips_to_modern_name_when_legacy_is_rejected():
    """The exact failure from the first real run: the endpoint demanded
    max_completion_tokens."""
    dialect = EndpointDialect(token_param="max_tokens")
    assert dialect.adapt(_unsupported_param_error("max_tokens")) is True
    kwargs: dict = {}
    dialect.apply(kwargs, max_tokens=200, temperature=0.5)
    assert kwargs["max_completion_tokens"] == 200


def test_does_not_flip_back_and_forth_forever():
    """Once already on the other spelling, a repeat rejection must not be
    treated as adaptable -- otherwise the retry loop never terminates."""
    dialect = EndpointDialect(token_param="max_tokens")
    assert dialect.adapt(_unsupported_param_error("max_completion_tokens")) is False


def test_drops_temperature_when_only_the_default_is_supported():
    dialect = EndpointDialect()
    assert dialect.adapt(_unsupported_temperature_error()) is True
    kwargs: dict = {"temperature": 0.9}
    dialect.apply(kwargs, max_tokens=200, temperature=0.9)
    assert "temperature" not in kwargs
    # A second rejection has nothing left to change.
    assert dialect.adapt(_unsupported_temperature_error()) is False


def test_unrelated_error_is_not_adapted():
    dialect = EndpointDialect()
    assert dialect.adapt(FakeAPIError("Error code: 401 - invalid api key")) is False


@pytest.mark.asyncio
async def test_provider_retries_once_and_then_succeeds():
    """End-to-end through the provider: the endpoint rejects max_tokens on
    the first call, and the retry with max_completion_tokens works."""
    from types import SimpleNamespace

    from app.ai.provider.base import Message
    from app.ai.provider.luna_openai import LunaOpenAIProvider
    from app.ai.schemas import VoteOutput

    provider = LunaOpenAIProvider(
        api_key="sk-test", base_url="https://example.invalid/v1", model="gpt-5.6-luna"
    )
    seen: list[dict] = []

    async def fake_create(**kwargs):
        seen.append(kwargs)
        if "max_tokens" in kwargs:
            raise _unsupported_param_error("max_tokens")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"vote_target": "p1", "reason": "ok"}')
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    provider._dialect.token_param = "max_tokens"  # simulate the old default
    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign]

    result = await provider.generate_structured(
        system="sys", messages=[Message(role="user", content="hi")], response_schema=VoteOutput
    )
    assert result == VoteOutput(vote_target="p1", reason="ok")
    assert "max_tokens" in seen[0]
    assert "max_completion_tokens" in seen[1]
    # The learned setting sticks, so later calls do not pay the rejection again.
    assert provider.dialect.token_param == "max_completion_tokens"
