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


def test_luna_provider_asks_for_the_current_model_when_none_is_configured():
    """The default is what a fresh checkout, the Actions fallback and any
    deployment that never set LUNA_MODEL will call. It moved from gpt-5.6-luna
    to gpt-6.0-luna, so a rename that missed one place shows up here."""
    settings = Settings(
        werewolf_llm_provider="luna",
        luna_api_key="sk-test",
        luna_base_url="https://example.invalid/v1",
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, LunaOpenAIProvider)
    assert settings.luna_model == "gpt-6.0-luna"
    assert provider._model == "gpt-6.0-luna"


def test_an_explicit_luna_model_still_wins_over_the_default():
    """A deployment pinned to the previous model keeps working until it is moved."""
    settings = Settings(
        werewolf_llm_provider="luna",
        luna_api_key="sk-test",
        luna_base_url="https://example.invalid/v1",
        luna_model="gpt-5.6-luna",
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, LunaOpenAIProvider)
    assert provider._model == "gpt-5.6-luna"


@pytest.mark.parametrize(
    "configured",
    [
        " luna ",
        '"luna"',
        "LUNA",
        # The model name typed into the provider field, old and new generation,
        # in any case -- matched by shape, so the next one needs no code change.
        "gpt-5.6-luna",
        "gpt-6.0-luna",
        " GPT-6.0-Luna ",
        '"gpt-6.1-luna"',
    ],
)
def test_luna_provider_normalizes_common_codespaces_secret_values(configured: str):
    settings = Settings(werewolf_llm_provider=configured, luna_api_key="sk-test")
    provider = build_llm_provider(settings)
    assert settings.werewolf_llm_provider == "luna"
    assert isinstance(provider, LunaOpenAIProvider)


def test_unknown_provider_raises():
    settings = Settings(werewolf_llm_provider="bogus")
    with pytest.raises(LLMProviderConfigError):
        build_llm_provider(settings)


@pytest.mark.parametrize(
    "configured", ["gpt-6.0-terra", "gpt-luna", "gpt-6.0-luna-mini", "6.0-luna"]
)
def test_only_luna_shaped_model_names_are_aliased(configured: str):
    """The alias must not swallow a typo: anything else stays an unknown provider."""
    settings = Settings(werewolf_llm_provider=configured)
    assert settings.werewolf_llm_provider == configured
    with pytest.raises(LLMProviderConfigError):
        build_llm_provider(settings)
