"""Config loading regression tests.

The prior implementation declared `python-dotenv` as a dependency but never
actually called it, so a `.env` file silently did nothing. These tests lock
in that `.env` is really read, and that real environment variables still
win over it (so a deployment can inject secrets with no .env file present).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import DEFAULT_ANTHROPIC_MODEL, Settings


def _write_env(tmp_path: Path, body: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(body, encoding="utf-8")
    return env_file


def test_env_file_is_actually_read(tmp_path: Path):
    env_file = _write_env(
        tmp_path,
        "WEREWOLF_LLM_PROVIDER=luna\nLUNA_API_KEY=sk-from-env-file\nLUNA_MODEL=gpt-5.6-luna\n",
    )
    settings = Settings(_env_file=str(env_file))
    assert settings.werewolf_llm_provider == "luna"
    assert settings.luna_api_key == "sk-from-env-file"
    assert settings.luna_model == "gpt-5.6-luna"


def test_real_environment_variable_overrides_env_file(tmp_path: Path, monkeypatch):
    env_file = _write_env(tmp_path, "LUNA_API_KEY=sk-from-env-file\n")
    monkeypatch.setenv("LUNA_API_KEY", "sk-from-real-envvar")
    settings = Settings(_env_file=str(env_file))
    assert settings.luna_api_key == "sk-from-real-envvar"


def test_shipped_env_example_loads_without_error(monkeypatch):
    """Copying `.env.example` verbatim to `.env` is the documented setup
    step, so it must parse. This caught `WEREWOLF_RNG_SEED=` (blank, meaning
    "no fixed seed") failing int parsing at startup."""
    for var in ("WEREWOLF_LLM_PROVIDER", "LUNA_API_KEY", "WEREWOLF_RNG_SEED"):
        monkeypatch.delenv(var, raising=False)
    example = Path(__file__).resolve().parents[2] / ".env.example"
    assert example.exists()

    settings = Settings(_env_file=str(example))
    assert settings.werewolf_llm_provider == "mock"
    assert settings.werewolf_rng_seed is None
    # Inline `# comment` suffixes must not bleed into the parsed value.
    assert "#" not in settings.werewolf_llm_provider


def test_blank_optional_int_is_treated_as_none(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("WEREWOLF_RNG_SEED", raising=False)
    env_file = _write_env(tmp_path, "WEREWOLF_RNG_SEED=\n")
    assert Settings(_env_file=str(env_file)).werewolf_rng_seed is None

    env_file = _write_env(tmp_path, "WEREWOLF_RNG_SEED=42\n")
    assert Settings(_env_file=str(env_file)).werewolf_rng_seed == 42


def test_defaults_are_safe_when_no_env_file_exists(tmp_path: Path, monkeypatch):
    for var in ("WEREWOLF_LLM_PROVIDER", "LUNA_API_KEY", "LUNA_MODEL"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=str(tmp_path / "does-not-exist.env"))
    # Defaulting to the mock provider means a fresh checkout never attempts
    # a paid API call by accident.
    assert settings.werewolf_llm_provider == "mock"
    assert settings.luna_api_key == ""
    # What the luna provider asks for when LUNA_MODEL is not set anywhere.
    assert settings.luna_model == "gpt-6-luna"


def test_shipped_env_example_names_the_same_model_as_the_default():
    """`.env.example` is what people copy. If it names a different model than
    the code default, a fresh `cp .env.example .env` silently pins the old one."""
    from app.config import DEFAULT_LUNA_MODEL

    text = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    assert f"LUNA_MODEL={DEFAULT_LUNA_MODEL}" in text


def test_explicit_kwarg_outranks_the_environment(monkeypatch):
    """The precedence that made an evaluation run silently use the mock
    provider: a CLI flag passed as a kwarg beats WEREWOLF_LLM_PROVIDER. That
    is correct for an *explicit* flag -- the scripts must therefore pass no
    kwarg at all when the flag is omitted (see the next test)."""
    monkeypatch.setenv("WEREWOLF_LLM_PROVIDER", "luna")
    assert Settings(werewolf_llm_provider="mock").werewolf_llm_provider == "mock"


def test_v3_is_the_default_engine_and_an_environment_value_still_wins(monkeypatch):
    monkeypatch.delenv("WEREWOLF_REASONING_ENGINE", raising=False)
    assert Settings().werewolf_reasoning_engine == "v3"
    # A deployment that still carries the old value keeps running that engine
    # until it is removed -- which is why the README says to remove it.
    monkeypatch.setenv("WEREWOLF_REASONING_ENGINE", " V2 ")
    assert Settings().werewolf_reasoning_engine == "v2"


def test_reasoning_effort_is_off_unless_set(monkeypatch):
    assert Settings().luna_reasoning_effort == ""
    monkeypatch.setenv("LUNA_REASONING_EFFORT", "low")
    assert Settings().luna_reasoning_effort == "low"


def test_omitted_flag_leaves_the_environment_in_charge(monkeypatch):
    monkeypatch.setenv("WEREWOLF_LLM_PROVIDER", "luna")
    overrides: dict[str, str] = {}  # what the scripts build when --provider is absent
    assert Settings(**overrides).werewolf_llm_provider == "luna"


# --- anthropic -----------------------------------------------------------------


def test_anthropic_defaults(tmp_path: Path):
    settings = Settings(_env_file=str(tmp_path / "does-not-exist.env"))
    assert settings.werewolf_anthropic_api_key == ""
    assert settings.werewolf_anthropic_base_url == "https://api.anthropic.com"
    assert settings.werewolf_anthropic_model == DEFAULT_ANTHROPIC_MODEL == "claude-opus-5-5"
    assert settings.werewolf_anthropic_max_concurrency == 8
    assert settings.werewolf_anthropic_timeout_seconds == 120.0
    assert settings.werewolf_anthropic_max_retries == 2
    assert settings.werewolf_anthropic_max_tokens == 12000
    assert settings.anthropic_effort_by_kind == {
        "discussion": "low",
        "vote": "medium",
        "night": "medium",
        "chat": "low",
        "default": "medium",
    }
    assert settings.werewolf_anthropic_fallbacks == "default"
    # Paid provider + no password must be opted into, never the default.
    assert settings.werewolf_allow_open_access is False
    assert settings.werewolf_session_max_cost_usd == 30.0


def test_the_sdks_own_variables_are_never_read(monkeypatch):
    """`ANTHROPIC_*` on a developer machine usually belongs to another tool;
    the game reads only its namespaced settings."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://ambient.invalid")
    settings = Settings()
    assert settings.werewolf_anthropic_api_key == ""
    assert settings.werewolf_anthropic_base_url == "https://api.anthropic.com"

    monkeypatch.setenv("WEREWOLF_ANTHROPIC_API_KEY", "game-key")
    monkeypatch.setenv("WEREWOLF_ANTHROPIC_BASE_URL", "https://example.invalid")
    settings = Settings()
    assert settings.werewolf_anthropic_api_key == "game-key"
    assert settings.werewolf_anthropic_base_url == "https://example.invalid"


def test_effort_is_normalized_and_validated(monkeypatch):
    monkeypatch.setenv("WEREWOLF_ANTHROPIC_EFFORT_VOTE", " HIGH ")
    monkeypatch.setenv("WEREWOLF_ANTHROPIC_EFFORT_NIGHT", '"xhigh"')
    # Blank, as `.env.example` lines may be: the field's own default.
    monkeypatch.setenv("WEREWOLF_ANTHROPIC_EFFORT_DISCUSSION", "")
    settings = Settings()
    assert settings.werewolf_anthropic_effort_vote == "high"
    assert settings.werewolf_anthropic_effort_night == "xhigh"
    assert settings.werewolf_anthropic_effort_discussion == "low"

    # A typo fails at startup, not as a 400 on the first paid request.
    monkeypatch.setenv("WEREWOLF_ANTHROPIC_EFFORT_CHAT", "extreme")
    with pytest.raises(ValidationError, match="WEREWOLF_ANTHROPIC_EFFORT_CHAT"):
        Settings()


def test_fallbacks_accept_only_default_or_off(monkeypatch):
    monkeypatch.setenv("WEREWOLF_ANTHROPIC_FALLBACKS", " OFF ")
    assert Settings().werewolf_anthropic_fallbacks == "off"
    monkeypatch.setenv("WEREWOLF_ANTHROPIC_FALLBACKS", "")
    assert Settings().werewolf_anthropic_fallbacks == "default"
    monkeypatch.setenv("WEREWOLF_ANTHROPIC_FALLBACKS", "sometimes")
    with pytest.raises(ValidationError, match="WEREWOLF_ANTHROPIC_FALLBACKS"):
        Settings()


def test_a_blank_model_or_base_url_means_the_default(tmp_path: Path):
    env_file = _write_env(tmp_path, "WEREWOLF_ANTHROPIC_MODEL=\nWEREWOLF_ANTHROPIC_BASE_URL=\n")
    settings = Settings(_env_file=str(env_file))
    assert settings.werewolf_anthropic_model == DEFAULT_ANTHROPIC_MODEL
    assert settings.werewolf_anthropic_base_url == "https://api.anthropic.com"


def test_a_quoted_model_is_unquoted(monkeypatch):
    monkeypatch.setenv("WEREWOLF_ANTHROPIC_MODEL", ' "claude-sonnet-5-5" ')
    assert Settings().werewolf_anthropic_model == "claude-sonnet-5-5"


def test_shipped_env_example_configures_anthropic_with_placeholders_only():
    """What a fresh `cp .env.example .env` gets: the default model, no key,
    the mock provider still selected, and the open-access guard on."""
    example = Path(__file__).resolve().parents[2] / ".env.example"
    text = example.read_text(encoding="utf-8")
    assert f"WEREWOLF_ANTHROPIC_MODEL={DEFAULT_ANTHROPIC_MODEL}" in text

    settings = Settings(_env_file=str(example))
    assert settings.werewolf_llm_provider == "mock"
    assert settings.werewolf_anthropic_api_key == ""
    assert settings.werewolf_anthropic_model == DEFAULT_ANTHROPIC_MODEL
    assert settings.werewolf_anthropic_fallbacks == "default"
    assert settings.werewolf_allow_open_access is False
