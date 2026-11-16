"""The suite runs offline on the defaults, whatever the developer's shell holds.

The README tells a Codespaces owner to register `WEREWOLF_LLM_PROVIDER`,
`WEREWOLF_ANTHROPIC_API_KEY` and `WEREWOLF_ACCESS_PASSWORD` as secrets, and
Codespaces turns secrets into environment variables. The documented
`cd backend && pytest` then has to behave exactly as it does in CI.

The hard part is collection time: `test_api_smoke`, `test_spa_serving` and
`test_websocket_events` do `from app.main import app`, which builds the app
while pytest is still importing test modules -- before any fixture runs. With
a password in the environment that app was built behind the password and
every request was a 401; with a paid provider and no password the app refused
to start and collection aborted. Each case here runs those modules in a fresh
pytest process with the developer's settings in place, so the import order is
the real one. Nothing reaches a network: the key is fake, and with the
isolation working the games run on the mock provider.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2]

# One quick test from each module that builds the app at import time, i.e.
# during collection. Each sends a request to that module-level app, so an app
# built behind a password fails it with a 401; naming one test still imports
# its whole module. The full games in these modules take seconds each and
# would add nothing here.
COLLECTION_TIME_APP_TESTS = (
    "tests/e2e/test_api_smoke.py::test_an_ambiguous_human_sentence_does_not_become_a_co",
    "tests/e2e/test_spa_serving.py",
    "tests/e2e/test_websocket_events.py::test_a_wolf_channel_frame_never_reaches_a_non_wolf_socket",
)

FAKE_ANTHROPIC_KEY = "sk-ant-test-isolation-not-a-real-key"
FAKE_LUNA_KEY = "luna-test-isolation-not-a-real-key"

CODESPACES_SECRETS = {
    "WEREWOLF_LLM_PROVIDER": "anthropic",
    "WEREWOLF_ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY,
    "WEREWOLF_ACCESS_PASSWORD": "codespaces-password",
}
LIVE_EVALUATION_ENV = {
    # What the evaluation scripts need, and no password: they do not serve.
    "WEREWOLF_LLM_PROVIDER": "anthropic",
    "WEREWOLF_ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY,
}
LUNA_ENV = {
    "WEREWOLF_LLM_PROVIDER": "luna",
    "LUNA_API_KEY": FAKE_LUNA_KEY,
    "LUNA_BASE_URL": "https://luna.invalid/v1",
}


def _run_pytest(
    tmp_path: Path, env: dict[str, str], dotenv: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the collection-time tests in a new pytest process.

    The working directory is a fresh temporary one, so a real `backend/.env`
    on this machine plays no part; `dotenv` writes one there instead, which
    is where `Settings` looks for it.
    """
    if dotenv is not None:
        lines = [f"{key}={value}" for key, value in dotenv.items()]
        (tmp_path / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # This process's own environment is already scrubbed by tests/conftest.py,
    # so the child sees exactly the developer settings given here.
    child_env = {**os.environ, **env}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *(str(BACKEND / test) for test in COLLECTION_TIME_APP_TESTS),
        ],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize(
    ("env", "dotenv"),
    [
        pytest.param(CODESPACES_SECRETS, None, id="codespaces-secrets-with-password"),
        pytest.param(LIVE_EVALUATION_ENV, None, id="live-evaluation-env-without-password"),
        pytest.param({}, LIVE_EVALUATION_ENV, id="backend-dotenv-without-password"),
        pytest.param({}, CODESPACES_SECRETS, id="backend-dotenv-with-password"),
        pytest.param(LUNA_ENV, None, id="luna-env-without-password"),
        pytest.param(
            {"WEREWOLF_ALLOW_OPEN_ACCESS": "true", **LIVE_EVALUATION_ENV},
            None,
            id="open-access-opt-in",
        ),
    ],
)
def test_the_documented_developer_settings_do_not_change_the_suite(
    tmp_path: Path, env: dict[str, str], dotenv: dict[str, str] | None
) -> None:
    result = _run_pytest(tmp_path, env, dotenv)

    output = result.stdout + result.stderr
    lines = result.stdout.strip().splitlines()
    summary = lines[-1] if lines else ""
    assert result.returncode == 0, output[-4000:]
    assert " passed" in summary, output[-4000:]
    assert "error" not in summary and "failed" not in summary, output[-4000:]
    assert FAKE_ANTHROPIC_KEY not in output
    assert FAKE_LUNA_KEY not in output
