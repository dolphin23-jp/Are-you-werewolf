"""Provider selection via an explicit env var (`WEREWOLF_LLM_PROVIDER`), not
a heuristic sniff of the API key string like the prior implementation. If
`luna` is selected without an API key configured, fail loud at startup
rather than silently degrading to mock."""

from __future__ import annotations

from app.ai.metrics import MetricsCollector
from app.ai.provider.base import LLMProvider
from app.ai.provider.mock import MockProvider
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
        )

    raise LLMProviderConfigError(
        f"unknown WEREWOLF_LLM_PROVIDER={settings.werewolf_llm_provider!r}"
    )
