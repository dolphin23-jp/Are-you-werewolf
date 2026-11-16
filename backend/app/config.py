import re
from typing import Any, Literal

from pydantic import ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Shared by the field default below and by `_normalize_reasoning_engine`'s
# blank-value fallback, so the two can never drift apart again -- a blank
# `WEREWOLF_REASONING_ENGINE=` line in a `.env` (the form `.env.example` uses
# for every other optional setting) must resolve to whatever the real default
# is, not to a value hardcoded independently of it.
_DEFAULT_REASONING_ENGINE: Literal["legacy", "v2", "v3"] = "v3"

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

# The Claude model the `anthropic` provider asks for when
# `WEREWOLF_ANTHROPIC_MODEL` is not set: Sonnet 5.5, the owner's decision of
# 2026-11-16 (docs/approach-reset-2026-10.md §11.8). On the game's own
# discussion prompts at `low` effort it finished a turn in 4.0 s at the median
# (Opus 5.5: 9.2 s; the target is about 5 s) at about 0.44x the cost; its game
# quality is read on the next paired-seed evaluation. `claude-opus-5-5` stays
# selectable here. Neither model takes a thinking budget or `disabled`
# thinking, so the provider never sends `thinking` (adaptive by default) and
# steers depth with `output_config.effort` (below).
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"

# A Claude model id (`claude-opus-5-5`, `claude-sonnet-5-5-20261101`, ...) typed
# into the *provider* field. Unambiguous for the same reason as the luna
# pattern: no other provider's models are named `claude-*`.
_CLAUDE_MODEL_NAME_RE = re.compile(r"claude-[a-z0-9-]+")
# A model id with a version (`claude-opus-5-5`, `claude-haiku-5-5`, a dated
# snapshot). Only such an id typed into the provider field becomes the model;
# a bare family name (`claude-opus`, `claude-code`) names no callable model and
# would 404 every call, so it selects the provider and keeps the default model.
_VERSIONED_CLAUDE_MODEL_RE = re.compile(r"claude-[a-z]+-\d+(?:-\d+)*(?:-\d{8})?")

# Every value `output_config.effort` accepts. Validated at startup so a typo
# fails here instead of as a 400 on the first paid request of a game.
ANTHROPIC_EFFORT_LEVELS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})

# Per call kind, how hard the model thinks. One mapping, shared by the field
# defaults below, their blank-value fallback and the provider's own default,
# so none of them can drift from the others. Discussion and wolf chat are many
# short turns where latency is what the player feels; a vote or a night action
# is one decision per day that the whole game turns on.
DEFAULT_ANTHROPIC_EFFORT_BY_KIND: dict[str, str] = {
    "discussion": "low",
    "vote": "medium",
    "night": "medium",
    "chat": "low",
    "default": "medium",
}

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

_ANTHROPIC_FALLBACK_MODES = ("default", "off")


def _unquote(value: str) -> str:
    """The value inside the quotes and whitespace a Codespaces secret is often
    pasted with (`"sk-..."`, a trailing newline). A `.env` file is unquoted by
    its parser already; a real environment variable arrives verbatim."""
    return value.strip().strip("\"'").strip()


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
        # A validation error normally quotes its input, and for a check that
        # spans fields (`_model_id_in_the_provider_field_is_the_model`) the
        # input is every setting -- API keys included -- printed to the
        # startup log. The messages here name the offending setting instead.
        hide_input_in_errors=True,
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
    # order in code, and the model is left with wording. Kept for comparison
    # and maintained, not improved.
    # v3 (the default): the reasoning layer keeps the facts, the solver, the
    # validation and the speaking order, but the model decides whom to suspect,
    # how to vote and what to do at night, and speaks in a chat register with
    # no injected wording. On the same model and seed it ran about four times
    # faster than v2 with no failed calls (docs/approach-reset-2026-10.md,
    # sections 2.1b and 6.1).
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

    # --- Anthropic (Claude) -- the real provider. ---
    # Namespaced on purpose rather than the SDK's own `ANTHROPIC_API_KEY` /
    # `ANTHROPIC_BASE_URL`: those are routinely set by other tools on the same
    # machine (a coding agent, a proxy), and picking one up would silently bill
    # the wrong account or send hidden-role prompts to the wrong host. The
    # provider passes the key and the base URL to the SDK explicitly, so the
    # SDK never falls back to its ambient variables either.
    werewolf_anthropic_api_key: str = ""
    werewolf_anthropic_base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    werewolf_anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    # One limiter for the whole process, shared by every provider instance
    # (each game builds its own): the rate limit is per API key, not per game.
    werewolf_anthropic_max_concurrency: int = 8
    # Generous because a thinking model at `max_tokens` 12000 can legitimately
    # run for a minute or more; a short timeout would bill the call and then
    # discard it.
    werewolf_anthropic_timeout_seconds: float = 120.0
    werewolf_anthropic_max_retries: int = 2
    # Floor for every request's `max_tokens`: the larger of this and the
    # caller's value is sent. Sonnet 5.5 and Opus 5.5 think adaptively by
    # default, and thinking is billed against the same budget as the visible
    # JSON, so the small budgets the agent asks for (sized for a non-thinking
    # model) would truncate mid-object.
    # Only tokens actually produced are billed, so a high ceiling costs nothing
    # by itself.
    werewolf_anthropic_max_tokens: int = 12000
    werewolf_anthropic_effort_discussion: str = DEFAULT_ANTHROPIC_EFFORT_BY_KIND["discussion"]
    werewolf_anthropic_effort_vote: str = DEFAULT_ANTHROPIC_EFFORT_BY_KIND["vote"]
    werewolf_anthropic_effort_night: str = DEFAULT_ANTHROPIC_EFFORT_BY_KIND["night"]
    werewolf_anthropic_effort_chat: str = DEFAULT_ANTHROPIC_EFFORT_BY_KIND["chat"]
    werewolf_anthropic_effort_default: str = DEFAULT_ANTHROPIC_EFFORT_BY_KIND["default"]
    # "default": a refused turn is re-run server-side on the fallback model
    # inside the same call (beta). "off": a refusal just fails the call, and
    # the agent's own fallback line takes over.
    werewolf_anthropic_fallbacks: str = "default"

    # A real provider costs money per call, so a publicly reachable server
    # with no access password is an open tab on the owner's account. Starting
    # with provider=anthropic and an empty WEREWOLF_ACCESS_PASSWORD has to be
    # opted into with this flag; it is never the default.
    werewolf_allow_open_access: bool = False
    # Ceiling on the estimated spend of one human-play session, in USD.
    werewolf_session_max_cost_usd: float = 30.0

    @model_validator(mode="before")
    @classmethod
    def _model_id_in_the_provider_field_is_the_model(cls, data: Any) -> Any:
        """`WEREWOLF_LLM_PROVIDER=claude-haiku-5-5` means Claude, *that* model.

        `_normalize_llm_provider` turns a Claude model id in the provider
        field into `anthropic`; without this the id itself was dropped and
        the run billed the default model -- then Opus 5.5, forty times
        Haiku's price -- with nothing on screen to show the swap. This runs
        before the field validators, so it still sees the id as typed.

        A blank or absent `WEREWOLF_ANTHROPIC_MODEL` takes the id. One that
        names the same model is fine. One that names another model means two
        settings disagree about what to bill, and either guess bills a model
        the owner did not pick, so startup fails and says which two.
        """
        if not isinstance(data, dict):
            return data
        provider = data.get("werewolf_llm_provider")
        if not isinstance(provider, str):
            return data
        model_id = _unquote(provider).lower()
        if not _VERSIONED_CLAUDE_MODEL_RE.fullmatch(model_id):
            return data
        configured = data.get("werewolf_anthropic_model")
        if configured is None or (isinstance(configured, str) and not _unquote(configured)):
            return {**data, "werewolf_anthropic_model": model_id}
        if isinstance(configured, str) and _unquote(configured).lower() != model_id:
            raise ValueError(
                f"WEREWOLF_LLM_PROVIDER={provider!r} names the model {model_id!r}, but "
                f"WEREWOLF_ANTHROPIC_MODEL={configured!r} names another. Name the model "
                "in one place: remove WEREWOLF_ANTHROPIC_MODEL, or set "
                "WEREWOLF_LLM_PROVIDER=anthropic."
            )
        return data

    @field_validator("werewolf_llm_provider", mode="before")
    @classmethod
    def _normalize_llm_provider(cls, value: object) -> object:
        """Tolerate the common Codespaces-secret forms without hiding bad providers."""
        if not isinstance(value, str):
            return value
        normalized = value.strip().strip('"\'').lower()
        # The model name is frequently entered in the provider field. Each
        # real provider's models have a name shape no other provider uses
        # (`gpt-*-luna`, `claude-*`), so these aliases cannot pick the wrong
        # one; anything else stays as typed and fails as an unknown provider.
        # A Claude id is also the model asked for: see
        # `_model_id_in_the_provider_field_is_the_model`.
        if _LUNA_MODEL_NAME_RE.fullmatch(normalized):
            return "luna"
        if normalized == "claude" or _CLAUDE_MODEL_NAME_RE.fullmatch(normalized):
            return "anthropic"
        return normalized

    @field_validator(
        "werewolf_anthropic_effort_discussion",
        "werewolf_anthropic_effort_vote",
        "werewolf_anthropic_effort_night",
        "werewolf_anthropic_effort_chat",
        "werewolf_anthropic_effort_default",
        mode="before",
    )
    @classmethod
    def _normalize_anthropic_effort(cls, value: object, info: ValidationInfo) -> object:
        """Fail at startup on an effort the API would reject with a 400.

        A blank line (the `.env.example` form) means "the default for this
        call kind", taken from the same mapping as the field default.
        """
        if not isinstance(value, str):
            return value
        normalized = value.strip().strip('"\'').lower()
        if not normalized:
            kind = (info.field_name or "").removeprefix("werewolf_anthropic_effort_")
            return DEFAULT_ANTHROPIC_EFFORT_BY_KIND[kind]
        if normalized not in ANTHROPIC_EFFORT_LEVELS:
            raise ValueError(
                f"{info.field_name.upper() if info.field_name else 'effort'}={value!r} is not "
                f"one of {sorted(ANTHROPIC_EFFORT_LEVELS)}"
            )
        return normalized

    @field_validator("werewolf_anthropic_fallbacks", mode="before")
    @classmethod
    def _normalize_anthropic_fallbacks(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().strip('"\'').lower() or "default"
        if normalized not in _ANTHROPIC_FALLBACK_MODES:
            raise ValueError(
                f"WEREWOLF_ANTHROPIC_FALLBACKS={value!r} is not one of {_ANTHROPIC_FALLBACK_MODES}"
            )
        return normalized

    @field_validator("werewolf_access_password", mode="before")
    @classmethod
    def _drop_captured_comment(cls, value: object) -> object:
        """Treat a value that is only a captured inline comment as no password.

        `docker run --env-file` keeps everything after `=` verbatim, so a hand
        edited `.env` line such as `WEREWOLF_ACCESS_PASSWORD=   # set me`
        arrives as a non-empty "password". The server would then start "with
        a password" nobody knows, and the refusal to serve a paid provider
        without one would never fire. Whitespace before `#` is the signature of
        that capture; a real password is also stripped of the padding and
        trailing newline a pasted secret tends to carry.
        """
        if not isinstance(value, str):
            return value
        if re.match(r"\s+#", value):
            return ""
        return value.strip()

    @field_validator("werewolf_anthropic_api_key", "luna_api_key", mode="before")
    @classmethod
    def _unquote_api_key(cls, value: object) -> object:
        """A key never contains quotes or whitespace, so they are dropped.

        A Codespaces secret pasted as `"sk-..."` or with a trailing newline
        otherwise reached the API verbatim, as `Bearer "sk-..."`: a 401 on
        every call, from a server that had started fine because the key was
        not blank. A key that is nothing but quotes and spaces becomes blank,
        so the provider factory's missing-key check stops startup instead.
        """
        if not isinstance(value, str):
            return value
        return _unquote(value)

    @field_validator("werewolf_anthropic_model", "werewolf_anthropic_base_url", mode="before")
    @classmethod
    def _blank_anthropic_string_is_default(cls, value: object, info: ValidationInfo) -> object:
        # A blank `WEREWOLF_ANTHROPIC_MODEL=` must mean "the default model",
        # not an empty model id that only fails on the first paid request.
        # Quotes are stripped as for the provider field (Codespaces secrets).
        if not isinstance(value, str):
            return value
        normalized = value.strip().strip('"\'')
        if normalized:
            return normalized
        if info.field_name == "werewolf_anthropic_model":
            return DEFAULT_ANTHROPIC_MODEL
        return DEFAULT_ANTHROPIC_BASE_URL

    @property
    def anthropic_effort_by_kind(self) -> dict[str, str]:
        """The five effort settings in the shape the provider consumes."""
        return {
            "discussion": self.werewolf_anthropic_effort_discussion,
            "vote": self.werewolf_anthropic_effort_vote,
            "night": self.werewolf_anthropic_effort_night,
            "chat": self.werewolf_anthropic_effort_chat,
            "default": self.werewolf_anthropic_effort_default,
        }

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


def settings_for_provider(provider: str | None, **overrides: object) -> Settings:
    """Settings with `provider` forced, keeping the model the configuration chose.

    A script that passes the provider as a keyword hides the raw environment
    value, so a model id typed into WEREWOLF_LLM_PROVIDER (which Settings turns
    into the model) would silently fall back to the default model -- and the
    run would bill a different model from the one the owner named and the
    preflight checked. When the forced and the configured provider are both
    anthropic, the configured model is carried over unless overridden.
    """
    if provider is None:
        return Settings(**overrides)  # type: ignore[arg-type]
    configured = Settings()
    values: dict[str, object] = {"werewolf_llm_provider": provider}
    if (
        configured.werewolf_llm_provider == "anthropic"
        and Settings(werewolf_llm_provider=provider).werewolf_llm_provider == "anthropic"
    ):
        values["werewolf_anthropic_model"] = configured.werewolf_anthropic_model
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]

