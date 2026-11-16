"""Startup guards, logging and the public health endpoint (`app/main.py`).

A paid provider behind no password is an open tab on the owner's account, so
the server must refuse to start that way unless told explicitly. The health
endpoint stays reachable without the password, so it may name the provider
and model but never a credential.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.config as config
from app.config import DEFAULT_ANTHROPIC_MODEL, DEFAULT_LUNA_MODEL, Settings
from app.main import OPEN_ACCESS_REFUSAL, create_app

FAKE_ANTHROPIC_KEY = "sk-ant-test-unit-main-not-a-real-key"
FAKE_LUNA_KEY = "luna-test-unit-main-not-a-real-key"

UseSettings = Callable[..., Settings]


@pytest.fixture
def use_settings(monkeypatch: pytest.MonkeyPatch) -> UseSettings:
    """Install a Settings object for `get_settings()`; restored after the test."""

    def apply(**overrides: Any) -> Settings:
        settings = Settings(**overrides)
        monkeypatch.setattr(config, "_settings", settings)
        return settings

    return apply


@contextmanager
def bare_root_logger() -> Iterator[logging.Logger]:
    """A root logger with no handlers, as under a plain `uvicorn app.main:app`.

    pytest attaches its own capture handlers to the root logger for the call
    phase of every test -- after fixtures have run -- which would make
    `create_app` (correctly) leave logging alone. So this is entered inside the
    test body, and everything is put back on the way out.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for handler in saved_handlers:
        root.removeHandler(handler)
    try:
        yield root
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


# -- the open-access guard --


# Every provider that bills a key, under each spelling the provider field
# accepts for it. The dormant luna provider bills too, and its README section
# says nothing about a password, so it needs the guard as much as anthropic.
PAID_PROVIDER_SPELLINGS = [
    ("anthropic", {"werewolf_anthropic_api_key": FAKE_ANTHROPIC_KEY}),
    ("claude", {"werewolf_anthropic_api_key": FAKE_ANTHROPIC_KEY}),
    ("claude-opus-5-5", {"werewolf_anthropic_api_key": FAKE_ANTHROPIC_KEY}),
    ("luna", {"luna_api_key": FAKE_LUNA_KEY}),
    ("gpt-6-luna", {"luna_api_key": FAKE_LUNA_KEY}),
]


@pytest.mark.parametrize(("provider", "key"), PAID_PROVIDER_SPELLINGS)
def test_refuses_to_start_a_paid_provider_without_a_password(
    use_settings: UseSettings, provider: str, key: dict[str, str]
) -> None:
    use_settings(
        werewolf_llm_provider=provider,
        werewolf_access_password="",
        werewolf_allow_open_access=False,
        **key,
    )
    with pytest.raises(RuntimeError) as excinfo:
        create_app()
    message = str(excinfo.value)
    assert message == OPEN_ACCESS_REFUSAL
    # Both remedies are named, in both languages, and the key is not echoed.
    assert "WEREWOLF_ACCESS_PASSWORD" in message
    assert "WEREWOLF_ALLOW_OPEN_ACCESS" in message
    assert "起動を中止" in message
    assert "Refusing to start" in message
    assert FAKE_ANTHROPIC_KEY not in message
    assert FAKE_LUNA_KEY not in message


@pytest.mark.parametrize(("provider", "key"), PAID_PROVIDER_SPELLINGS)
def test_starts_a_paid_provider_behind_a_password(
    use_settings: UseSettings, provider: str, key: dict[str, str]
) -> None:
    settings = use_settings(
        werewolf_llm_provider=provider,
        werewolf_access_password="table-password",
        werewolf_allow_open_access=False,
        **key,
    )
    client = TestClient(create_app())
    # The game API is behind the password; health stays public.
    assert client.post("/api/games", json={}).status_code == 401
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["llm_provider"] == settings.werewolf_llm_provider


@pytest.mark.parametrize(("provider", "key"), PAID_PROVIDER_SPELLINGS)
def test_open_access_is_an_explicit_opt_in_and_is_logged(
    use_settings: UseSettings,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    key: dict[str, str],
) -> None:
    use_settings(
        werewolf_llm_provider=provider,
        werewolf_access_password="",
        werewolf_allow_open_access=True,
        **key,
    )
    with caplog.at_level(logging.WARNING, logger="app.main"):
        client = TestClient(create_app())
    assert client.get("/api/health").status_code == 200
    assert any(
        record.levelno == logging.WARNING and "WEREWOLF_ALLOW_OPEN_ACCESS" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("provider", ["mock", "scenario"])
def test_offline_providers_need_no_password(use_settings: UseSettings, provider: str) -> None:
    use_settings(
        werewolf_llm_provider=provider,
        werewolf_access_password="",
        werewolf_allow_open_access=False,
    )
    client = TestClient(create_app())
    assert client.get("/api/health").status_code == 200


# -- /api/health --


@pytest.mark.parametrize(
    ("overrides", "expected_model"),
    [
        ({"werewolf_llm_provider": "mock"}, None),
        ({"werewolf_llm_provider": "scenario"}, None),
        (
            {
                "werewolf_llm_provider": "anthropic",
                "werewolf_anthropic_api_key": FAKE_ANTHROPIC_KEY,
                "werewolf_allow_open_access": True,
            },
            DEFAULT_ANTHROPIC_MODEL,
        ),
        (
            {
                "werewolf_llm_provider": "anthropic",
                "werewolf_anthropic_api_key": FAKE_ANTHROPIC_KEY,
                "werewolf_anthropic_model": "claude-sonnet-5-5",
                "werewolf_allow_open_access": True,
            },
            "claude-sonnet-5-5",
        ),
        (
            {
                "werewolf_llm_provider": "luna",
                "luna_api_key": FAKE_LUNA_KEY,
                "werewolf_allow_open_access": True,
            },
            DEFAULT_LUNA_MODEL,
        ),
    ],
)
def test_health_names_the_model_and_never_a_key(
    use_settings: UseSettings, overrides: dict[str, Any], expected_model: str | None
) -> None:
    use_settings(werewolf_access_password="", **overrides)
    resp = TestClient(create_app()).get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["llm_model"] == expected_model
    assert FAKE_ANTHROPIC_KEY not in resp.text
    assert FAKE_LUNA_KEY not in resp.text
    assert "api.anthropic.com" not in resp.text


# -- logging --


def test_configures_logging_when_nothing_else_has(
    use_settings: UseSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    use_settings(werewolf_log_level="debug")
    with bare_root_logger() as root:
        create_app()
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG
        # The point of it: a provider warning reaches the server log, with its
        # logger name, instead of Python's bare last-resort output.
        logging.getLogger("app.ai.provider.anthropic").warning("retrying after 529")
    err = capsys.readouterr().err
    assert "WARNING app.ai.provider.anthropic: retrying after 529" in err


def test_an_unknown_log_level_falls_back_to_info(
    use_settings: UseSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    use_settings(werewolf_log_level="chatty")
    with bare_root_logger() as root:
        create_app()
        assert root.level == logging.INFO
    assert "unknown WEREWOLF_LOG_LEVEL 'chatty'" in capsys.readouterr().err


def test_leaves_an_already_configured_root_logger_alone(use_settings: UseSettings) -> None:
    use_settings(werewolf_log_level="DEBUG")
    with bare_root_logger() as root:
        existing = logging.NullHandler()
        root.addHandler(existing)
        root.setLevel(logging.ERROR)
        create_app()
        assert root.handlers == [existing]
        assert root.level == logging.ERROR
