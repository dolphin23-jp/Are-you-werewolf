"""Each human-play session carries its own metrics and, on a paid provider,
its own spend cap; a spent cap stops the AI seats without breaking the
player's requests, and the view says so.

Nothing here sends a request to any LLM: the Anthropic game is created but
never started, and the exhaustion test runs on the free mock provider with
the coordinator's entry points replaced.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.config as config
from app.ai.metrics import MetricsCollector
from app.ai.provider.anthropic_claude import AnthropicClaudeProvider
from app.ai.provider.budget import BudgetedProvider, BudgetExceeded, EvaluationBudget
from app.ai.provider.mock import MockProvider
from app.api.routes_game import SESSION_MAX_REQUESTS
from app.config import Settings
from app.main import create_app
from app.sessions.models import GameSession
from app.sessions.store import get_session_store

FAKE_KEY = "sk-ant-test-session-budget-not-a-real-key"


@pytest.fixture
def client_for(monkeypatch: pytest.MonkeyPatch) -> Callable[..., TestClient]:
    def build(**overrides: Any) -> TestClient:
        monkeypatch.setattr(config, "_settings", Settings(**overrides))
        return TestClient(create_app())

    return build


def _create(client: TestClient, seed: int = 5) -> GameSession:
    resp = client.post("/api/games", json={"human_name": "Tester", "seed": seed})
    assert resp.status_code == 200
    session = get_session_store().get(resp.json()["session_id"])
    assert session is not None
    return session


def _agent_provider(session: GameSession) -> Any:
    """The provider the AI seats actually call (one shared instance)."""
    providers = {id(agent._provider) for agent in session.coordinator._agents.values()}
    assert len(providers) == 1
    return next(iter(session.coordinator._agents.values()))._provider


def test_an_anthropic_session_is_capped_and_metered(client_for: Callable[..., TestClient]):
    client = client_for(
        werewolf_llm_provider="anthropic",
        werewolf_anthropic_api_key=FAKE_KEY,
        werewolf_allow_open_access=True,
        werewolf_session_max_cost_usd=12.5,
    )
    session = _create(client)

    budget = session.budget
    assert isinstance(budget, EvaluationBudget)
    assert budget.max_estimated_cost == 12.5
    assert budget.max_requests == SESSION_MAX_REQUESTS == 5000
    assert isinstance(session.metrics, MetricsCollector)

    provider = _agent_provider(session)
    assert isinstance(provider, BudgetedProvider)
    assert provider.budget is budget
    inner = provider.inner
    assert isinstance(inner, AnthropicClaudeProvider)
    # The cap is enforced where requests are sent, not one level up.
    assert inner._request_budget is budget
    # One collector for the provider, the coordinator and the session.
    assert inner._metrics is session.metrics
    assert session.coordinator._metrics is session.metrics

    view = client.get(f"/api/games/{session.session_id}/view").json()
    assert view["ai_budget_exhausted"] is False
    assert view["ai_cost_usd"] == 0.0

    budget.reserve(0.5)
    budget.settle(0.5, 1.25)
    view = client.get(f"/api/games/{session.session_id}/view").json()
    assert view["ai_cost_usd"] == 1.25

    for path in ("view", "debug"):
        assert FAKE_KEY not in client.get(f"/api/games/{session.session_id}/{path}").text


def test_two_sessions_never_share_a_budget(client_for: Callable[..., TestClient]):
    client = client_for(
        werewolf_llm_provider="anthropic",
        werewolf_anthropic_api_key=FAKE_KEY,
        werewolf_allow_open_access=True,
    )
    first, second = _create(client, seed=1), _create(client, seed=2)
    assert first.budget is not second.budget
    assert first.metrics is not second.metrics


def test_an_offline_session_is_metered_but_not_capped(client_for: Callable[..., TestClient]):
    client = client_for(werewolf_llm_provider="mock")
    session = _create(client)

    assert session.budget is None
    assert isinstance(session.metrics, MetricsCollector)
    provider = _agent_provider(session)
    assert isinstance(provider, MockProvider)
    assert provider._metrics is session.metrics

    view = client.get(f"/api/games/{session.session_id}/view").json()
    assert view["ai_budget_exhausted"] is False
    assert view["ai_cost_usd"] is None


def test_a_spent_budget_stops_the_ai_and_the_view_says_so(
    client_for: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    client = client_for(werewolf_llm_provider="mock")
    session = _create(client, seed=11)
    sid = session.session_id
    coordinator = session.coordinator
    calls: list[str] = []

    async def over_budget(_session: object) -> None:
        calls.append("run_night_phase")
        raise BudgetExceeded("estimated cost budget would be exceeded")

    def record(name: str) -> Callable[..., Any]:
        async def call(*_args: object, **_kwargs: object) -> None:
            calls.append(name)

        return call

    monkeypatch.setattr(coordinator, "run_night_phase", over_budget)
    for name in ("advance_discussion", "run_discussion_round", "generate_all_votes"):
        monkeypatch.setattr(coordinator, name, record(name))
    monkeypatch.setattr(coordinator, "resume_after_human", lambda *a, **k: calls.append("resume"))

    with caplog.at_level(logging.WARNING, logger="app.api.orchestrator"):
        assert client.post(f"/api/games/{sid}/start").status_code == 200
        if not calls:
            # The human is the seer, so the night waits for their divination.
            debug = client.get(f"/api/games/{sid}/debug").json()
            target = next(p["player_id"] for p in debug["players"] if p["player_id"] != "p0")
            resp = client.post(
                f"/api/games/{sid}/night-action",
                json={"action_type": "divine", "target_id": target},
            )
            assert resp.status_code == 200

    assert calls == ["run_night_phase"]
    assert session.ai_budget_exhausted is True
    assert any("AI budget exhausted" in r.getMessage() for r in caplog.records)
    view = client.get(f"/api/games/{sid}/view").json()
    assert view["ai_budget_exhausted"] is True

    # The player can still act; the AI seats no longer do.
    assert client.post(f"/api/games/{sid}/resolve-night").status_code == 200
    assert client.post(f"/api/games/{sid}/start-discussion").status_code == 200
    assert client.post(f"/api/games/{sid}/chat", json={"content": "誰もいない?"}).status_code == 200
    assert client.post(f"/api/games/{sid}/pass-turn").status_code == 200
    resp = client.post(f"/api/games/{sid}/discussion-control", json={"action": "resume"})
    assert resp.status_code == 200
    assert client.post(f"/api/games/{sid}/end-discussion").status_code == 200
    assert calls == ["run_night_phase"]
    assert client.get(f"/api/games/{sid}/view").json()["ai_budget_exhausted"] is True
