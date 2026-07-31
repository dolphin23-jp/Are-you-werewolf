import pytest

from app.ai.provider.factory import LLMProviderConfigError, build_llm_provider
from app.ai.provider.luna_openai import LunaOpenAIProvider
from app.ai.provider.mock import MockProvider
from app.config import Settings


def test_mock_provider_selected_by_default():
    settings = Settings(werewolf_llm_provider="mock")
    provider = build_llm_provider(settings)
    assert isinstance(provider, MockProvider)


def test_luna_provider_requires_api_key():
    settings = Settings(werewolf_llm_provider="luna", luna_api_key="")
    with pytest.raises(LLMProviderConfigError):
        build_llm_provider(settings)


def test_luna_provider_builds_when_key_present():
    settings = Settings(
        werewolf_llm_provider="luna",
        luna_api_key="sk-test",
        luna_base_url="https://example.invalid/v1",
        luna_model="gpt-5.6-luna",
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, LunaOpenAIProvider)


def test_unknown_provider_raises():
    settings = Settings(werewolf_llm_provider="bogus")
    with pytest.raises(LLMProviderConfigError):
        build_llm_provider(settings)
