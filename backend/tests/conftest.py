import os

import pytest

import app.config as config
from app.config import Settings
from app.engine.game import GameController, PlayerSpec

# Settings that select and authenticate a real LLM provider. `CLAUDE.md` and
# `docs/architecture.md` both tell a developer to put these in `backend/.env`
# to run the live evaluation, and `Settings` reads that file relative to the
# working directory -- which is `backend/` for pytest too.
_LIVE_PROVIDER_KEYS = (
    "WEREWOLF_LLM_PROVIDER",
    "WEREWOLF_REASONING_ENGINE",
    "LUNA_API_KEY",
    "LUNA_BASE_URL",
    "LUNA_MODEL",
)


@pytest.fixture(autouse=True, scope="session")
def isolate_tests_from_developer_credentials():
    """Run the suite against the defaults, never the developer's live config.

    Without this, following the documented live-evaluation setup silently arms
    the whole test suite with a real provider: `test_api_smoke` asserts
    `MockProvider` and fails, and the full-game e2e tests stop being free and
    offline -- they start issuing paid API calls, one every twelve seconds, for
    a suite that is supposed to cost nothing and touch no network.

    The `.env` file is neutralised rather than deleted, and only the process's
    own copy of the settings is touched, so `backend/.env` keeps working for
    the live scripts. Tests that pass `_env_file=` explicitly are unaffected --
    `tests/unit/test_config.py` still proves the file is read and that a real
    environment variable outranks it.
    """
    original_env_file = Settings.model_config.get("env_file")
    removed = {key: os.environ.pop(key) for key in _LIVE_PROVIDER_KEYS if key in os.environ}
    Settings.model_config["env_file"] = None
    config._settings = None
    try:
        yield
    finally:
        Settings.model_config["env_file"] = original_env_file
        os.environ.update(removed)
        config._settings = None


def make_player_specs(n: int = 17) -> list[PlayerSpec]:
    return [PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0)) for i in range(n)]


def make_controller(seed: int | None = 1, session_id: str = "test-session") -> GameController:
    return GameController(session_id=session_id, player_specs=make_player_specs(), seed=seed)
