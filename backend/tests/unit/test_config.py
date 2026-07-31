"""Config loading regression tests.

The prior implementation declared `python-dotenv` as a dependency but never
actually called it, so a `.env` file silently did nothing. These tests lock
in that `.env` is really read, and that real environment variables still
win over it (so a deployment can inject secrets with no .env file present).
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings


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
    for var in ("WEREWOLF_LLM_PROVIDER", "LUNA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=str(tmp_path / "does-not-exist.env"))
    # Defaulting to the mock provider means a fresh checkout never attempts
    # a paid API call by accident.
    assert settings.werewolf_llm_provider == "mock"
    assert settings.luna_api_key == ""


def test_explicit_kwarg_outranks_the_environment(monkeypatch):
    """The precedence that made an evaluation run silently use the mock
    provider: a CLI flag passed as a kwarg beats WEREWOLF_LLM_PROVIDER. That
    is correct for an *explicit* flag -- the scripts must therefore pass no
    kwarg at all when the flag is omitted (see the next test)."""
    monkeypatch.setenv("WEREWOLF_LLM_PROVIDER", "luna")
    assert Settings(werewolf_llm_provider="mock").werewolf_llm_provider == "mock"


def test_omitted_flag_leaves_the_environment_in_charge(monkeypatch):
    monkeypatch.setenv("WEREWOLF_LLM_PROVIDER", "luna")
    overrides: dict[str, str] = {}  # what the scripts build when --provider is absent
    assert Settings(**overrides).werewolf_llm_provider == "luna"
