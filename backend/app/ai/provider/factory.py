"""Provider selection via an explicit env var (`WEREWOLF_LLM_PROVIDER`), not
a heuristic sniff of the API key string like the prior implementation. If a
real provider (`anthropic`, or the dormant `luna`) is selected without its API
key configured, fail loud at startup rather than silently degrading to mock.

The real SDKs are imported only inside their branch, so the offline providers
(and the test suite) never pay for importing them."""

from __future__ import annotations

from app.ai.metrics import MetricsCollector
from app.ai.provider.base import LLMProvider
from app.ai.provider.mock import MockProvider
from app.ai.provider.scenario import ScenarioProvider
from app.config import Settings


class LLMProviderConfigError(RuntimeError):
    pass


def build_llm_provider(
    settings: Settings,
    *,
    seed: int | None = None,
    metrics: MetricsCollector | None = None,
) -> LLMProvider:
    provider = settings.werewolf_llm_provider.lower()

    if provider == "mock":
        return MockProvider(seed=seed, metrics=metrics)

    if provider == "scenario":
        # Also offline and free -- a richer double than `mock`, used to drive the
        # reply/quote, pending-question and pressured-candidate paths end to end.
        return ScenarioProvider(seed=seed, metrics=metrics)

    if provider == "anthropic":
        if not settings.werewolf_anthropic_api_key:
            raise LLMProviderConfigError(
                "WEREWOLF_LLM_PROVIDER=anthropic but WEREWOLF_ANTHROPIC_API_KEY is not set. "
                "Set WEREWOLF_ANTHROPIC_API_KEY (the SDK's own ANTHROPIC_API_KEY is "
                "deliberately not read), or switch to WEREWOLF_LLM_PROVIDER=mock for "
                "offline development."
            )
        from app.ai.provider.anthropic_claude import AnthropicClaudeProvider

        return AnthropicClaudeProvider(
            api_key=settings.werewolf_anthropic_api_key,
            base_url=settings.werewolf_anthropic_base_url,
            model=settings.werewolf_anthropic_model,
            max_concurrency=settings.werewolf_anthropic_max_concurrency,
            timeout_seconds=settings.werewolf_anthropic_timeout_seconds,
            max_retries=settings.werewolf_anthropic_max_retries,
            max_tokens_floor=settings.werewolf_anthropic_max_tokens,
            effort_by_kind=settings.anthropic_effort_by_kind,
            fallbacks=settings.werewolf_anthropic_fallbacks != "off",
            metrics=metrics,
        )

    if provider == "luna":
        if not settings.luna_api_key:
            raise LLMProviderConfigError(
                "WEREWOLF_LLM_PROVIDER=luna but LUNA_API_KEY is not set. "
                "Set LUNA_API_KEY/LUNA_BASE_URL/LUNA_MODEL, or switch to "
                "WEREWOLF_LLM_PROVIDER=mock for offline development."
            )
        from app.ai.provider.luna_openai import LunaOpenAIProvider

        return LunaOpenAIProvider(
            api_key=settings.luna_api_key,
            base_url=settings.luna_base_url,
            model=settings.luna_model,
            max_concurrency=settings.luna_max_concurrency,
            timeout_seconds=settings.luna_timeout_seconds,
            max_retries=settings.luna_max_retries,
            metrics=metrics,
            reasoning_effort=settings.luna_reasoning_effort,
        )

    raise LLMProviderConfigError(
        f"unknown WEREWOLF_LLM_PROVIDER={settings.werewolf_llm_provider!r}"
    )
