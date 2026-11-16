import pytest

from app.ai.provider.anthropic_claude import AnthropicClaudeProvider
from app.ai.provider.factory import LLMProviderConfigError, build_llm_provider
from app.ai.provider.luna_openai import LunaOpenAIProvider
from app.ai.provider.mock import MockProvider
from app.ai.schemas import DiscussionOutput, VoteOutput
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
    to gpt-6-luna, so a rename that missed one place shows up here. The exact
    spelling matters: the dotted guess gpt-6.0-luna is refused by the endpoint
    with model_not_found."""
    settings = Settings(
        werewolf_llm_provider="luna",
        luna_api_key="sk-test",
        luna_base_url="https://example.invalid/v1",
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, LunaOpenAIProvider)
    assert settings.luna_model == "gpt-6-luna"
    assert provider._model == "gpt-6-luna"


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
        "gpt-6-luna",
        " GPT-6-Luna ",
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
    "configured", ["gpt-6-terra", "gpt-luna", "gpt-6-luna-mini", "6-luna"]
)
def test_only_luna_shaped_model_names_are_aliased(configured: str):
    """The alias must not swallow a typo: anything else stays an unknown provider."""
    settings = Settings(werewolf_llm_provider=configured)
    assert settings.werewolf_llm_provider == configured
    with pytest.raises(LLMProviderConfigError):
        build_llm_provider(settings)


# --- anthropic -----------------------------------------------------------------


def test_anthropic_provider_requires_its_own_namespaced_key(monkeypatch):
    """The SDK's ambient ANTHROPIC_API_KEY usually belongs to some other tool
    on the machine; it must not stand in for the game's key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key-of-another-tool")
    settings = Settings(werewolf_llm_provider="anthropic", werewolf_anthropic_api_key="")
    with pytest.raises(LLMProviderConfigError, match="WEREWOLF_ANTHROPIC_API_KEY"):
        build_llm_provider(settings)


def test_anthropic_provider_builds_from_settings():
    settings = Settings(
        werewolf_llm_provider="anthropic",
        werewolf_anthropic_api_key="test-key",
        werewolf_anthropic_base_url="https://example.invalid",
        werewolf_anthropic_max_retries=3,
        werewolf_anthropic_max_tokens=16000,
        werewolf_anthropic_effort_vote="high",
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, AnthropicClaudeProvider)
    assert provider.model == "claude-opus-5-5"
    assert provider.fallbacks_enabled is True
    assert provider._max_retries == 3
    assert provider._max_tokens_floor == 16000
    assert str(provider._client.base_url).startswith("https://example.invalid")
    assert provider.effort_for(VoteOutput) == "high"
    assert provider.effort_for(DiscussionOutput) == "low"


def test_anthropic_fallbacks_can_be_switched_off():
    settings = Settings(
        werewolf_llm_provider="anthropic",
        werewolf_anthropic_api_key="test-key",
        werewolf_anthropic_fallbacks="off",
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, AnthropicClaudeProvider)
    assert provider.fallbacks_enabled is False


@pytest.mark.parametrize(
    "configured",
    [
        "anthropic",
        " Anthropic ",
        "claude",
        '"Claude"',
        # A model id typed into the provider field.
        "claude-opus-5-5",
        " CLAUDE-SONNET-5-5 ",
        "claude-opus-5-5-20261101",
    ],
)
def test_anthropic_aliases_select_the_anthropic_provider(configured: str):
    settings = Settings(werewolf_llm_provider=configured, werewolf_anthropic_api_key="test-key")
    assert settings.werewolf_llm_provider == "anthropic"
    assert isinstance(build_llm_provider(settings), AnthropicClaudeProvider)


@pytest.mark.parametrize("configured", ["claude-", "claude_opus", "claud", "opus-5-5"])
def test_only_claude_shaped_names_are_aliased(configured: str):
    settings = Settings(werewolf_llm_provider=configured)
    assert settings.werewolf_llm_provider == configured
    with pytest.raises(LLMProviderConfigError):
        build_llm_provider(settings)
