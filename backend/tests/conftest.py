import os

import pytest

import app.config as app_config
from app.config import Settings
from app.engine.game import GameController, PlayerSpec

# The app's own settings all carry one of these prefixes. Every one is
# scrubbed, not only the provider and its key: the README has a Codespaces
# owner register `WEREWOLF_ACCESS_PASSWORD` as a secret too, and an app built
# behind it answers every test request with a 401.
_APP_SETTING_PREFIXES = ("WEREWOLF_", "LUNA_")
# The Anthropic SDK's own variables. The provider never reads them, but other
# tools on a developer machine (a coding agent, a proxy) commonly set them,
# and a test that builds a real SDK client must see the same clean
# environment CI does.
_SDK_KEYS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
)


def _is_developer_setting(name: str) -> bool:
    """Would this environment variable change what `Settings` reads?

    `Settings` matches names case-insensitively and with no prefix, so the
    test is on the upper-cased name and on every field the class declares,
    whatever it is called.
    """
    upper = name.upper()
    return (
        upper.startswith(_APP_SETTING_PREFIXES)
        or upper in _SDK_KEYS
        or name.lower() in Settings.model_fields
    )


_original_env_file = Settings.model_config.get("env_file")
_removed_environment: dict[str, str] = {}


def _isolate_from_developer_settings() -> None:
    """Run the suite against the defaults, never the developer's live config.

    Without this, following the documented live-evaluation setup silently arms
    the whole test suite with a real provider: `test_api_smoke` asserts
    `MockProvider` and fails, and the full-game e2e tests stop being free and
    offline -- they start issuing paid API calls, one every twelve seconds, for
    a suite that is supposed to cost nothing and touch no network.

    It runs when pytest imports this file, not in a fixture. pytest imports a
    directory's conftest before any test module under it, and several test
    modules build the app at import (`from app.main import app` runs
    `create_app()`), which is during collection, before even a session
    fixture runs. A fixture was too late for them: with a paid provider and no
    password the app refused to start and collection aborted, and with a
    password the module-level app was built behind it.

    The `.env` file is neutralised rather than deleted, and only the process's
    own copy of the settings is touched, so `backend/.env` keeps working for
    the live scripts. Tests that pass `_env_file=` explicitly are unaffected --
    `tests/unit/test_config.py` still proves the file is read and that a real
    environment variable outranks it.
    """
    for key in [key for key in os.environ if _is_developer_setting(key)]:
        _removed_environment[key] = os.environ.pop(key)
    Settings.model_config["env_file"] = None
    app_config._settings = None


_isolate_from_developer_settings()


def pytest_unconfigure(config: pytest.Config) -> None:
    """Hand the process its own settings back once the session is over."""
    Settings.model_config["env_file"] = _original_env_file
    os.environ.update(_removed_environment)
    app_config._settings = None


def make_player_specs(n: int = 17) -> list[PlayerSpec]:
    return [PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0)) for i in range(n)]


def make_controller(seed: int | None = 1, session_id: str = "test-session") -> GameController:
    return GameController(session_id=session_id, player_specs=make_player_specs(), seed=seed)
