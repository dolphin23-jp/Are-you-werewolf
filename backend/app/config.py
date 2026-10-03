import re
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Shared by the field default below and by `_normalize_reasoning_engine`'s
# blank-value fallback, so the two can never drift apart again -- a blank
# `WEREWOLF_REASONING_ENGINE=` line in a `.env` (the form `.env.example` uses
# for every other optional setting) must resolve to whatever the real default
# is, not to a value hardcoded independently of it.
_DEFAULT_REASONING_ENGINE: Literal["legacy", "v2", "v3"] = "v2"

# The OpenAI-compatible model the `luna` provider asks for when `LUNA_MODEL` is
# not set. One constant, so a model upgrade is a one-line change here and the
# alias matcher below never has to be taught the new name.
#
# Exactly `gpt-6-luna`. The dotted `gpt-6.0-luna`, guessed from the previous
# model's `gpt-5.6-luna`, is refused by the endpoint with `model_not_found`.
DEFAULT_LUNA_MODEL = "gpt-6-luna"

# A `gpt-<version>-luna` model name typed into the *provider* field. Matching the
# shape rather than one literal means the next model generation, and the one
# already deployed in somebody's Codespaces secret, both keep working.
_LUNA_MODEL_NAME_RE = re.compile(r"gpt-[0-9]+(?:\.[0-9]+)*-luna")


class Settings(BaseSettings):
    # `.env` is read relative to the process's working directory (i.e. run
    # uvicorn from `backend/`). Real environment variables always win over
    # the file, so a deployment can inject secrets without a .env at all.
    # The file itself is gitignored -- never commit real API keys.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    werewolf_env: str = "development"
    werewolf_cors_origins: str = "http://localhost:5173"
    werewolf_log_level: str = "INFO"
    werewolf_session_store: str = "memory"
    werewolf_rng_seed: int | None = None
    werewolf_access_password: str = ""
    werewolf_discussion_segment_size: int = 4
    werewolf_ai_pacing_scale: float = 1.0
    werewolf_discussion_wait_seconds: float = 45.0

    werewolf_llm_provider: str = "mock"
    # legacy: every decision goes through the model, with no fact ledger.
    # v2: the reasoning layer decides votes, night actions and the speaking
    # order in code, and the model is left with wording. Defaulted to legacy
    # while v2 was only exercised through manual live-evaluation scripts;
    # now the default so real gameplay (not just those scripts) is what gets
    # manually playtested. Set WEREWOLF_REASONING_ENGINE=legacy to compare.
    # v3: the reasoning layer keeps the facts, the solver, the validation and
    # the speaking order, but the model decides whom to suspect, how to vote
    # and what to do at night, and speaks in a chat register with no injected
    # wording. See docs/approach-reset-2026-10.md for why.
    werewolf_reasoning_engine: Literal["legacy", "v2", "v3"] = _DEFAULT_REASONING_ENGINE

    luna_api_key: str = ""
    luna_base_url: str = "https://api.example.com/v1"
    luna_model: str = DEFAULT_LUNA_MODEL
    luna_max_concurrency: int = 6
    luna_timeout_seconds: float = 30.0
    luna_max_retries: int = 2
    # Reasoning-model effort hint (`low` / `medium` / `high`), sent as
    # `reasoning_effort` when set. Empty means the parameter is not sent at
    # all, so an endpoint that never heard of it sees no change. On the
    # seed-11 live game 70% of every completion budget was spent on hidden
    # reasoning and half of all calls were cut off before the JSON closed;
    # this is the first knob to turn for that.
    luna_reasoning_effort: str = ""

    @field_validator("werewolf_llm_provider", mode="before")
    @classmethod
    def _normalize_llm_provider(cls, value: object) -> object:
        """Tolerate the common Codespaces-secret forms without hiding bad providers."""
        if not isinstance(value, str):
            return value
        normalized = value.strip().strip('"\'').lower()
        # The model name is frequently entered in the provider field. There is
        # currently only one real provider, so this unambiguous alias is safe.
        if _LUNA_MODEL_NAME_RE.fullmatch(normalized):
            return "luna"
        return normalized

    @field_validator("werewolf_reasoning_engine", mode="before")
    @classmethod
    def _normalize_reasoning_engine(cls, value: object) -> object:
        """Normalize the spelling, then let the Literal reject anything else.

        A typo used to fall through to a hardcoded engine in silence, so a
        deployment that meant to pick one engine would run the other and look
        like nothing had changed. Failing at startup is the only honest
        option -- and the blank-value fallback has to be the same default the
        field itself uses, or the two silently disagree.
        """
        if not isinstance(value, str):
            return value
        normalized = value.strip().strip('"\'').lower()
        return normalized or _DEFAULT_REASONING_ENGINE

    @field_validator("werewolf_rng_seed", mode="before")
    @classmethod
    def _empty_string_is_none(cls, value: object) -> object:
        # `.env.example` ships `WEREWOLF_RNG_SEED=` (left blank to mean "no
        # fixed seed"), and a blank line in a .env arrives as "" rather than
        # being absent -- which would otherwise fail int parsing at startup.
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.werewolf_cors_origins.split(",") if o.strip()]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
