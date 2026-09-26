"""Development-only tools stay out of a non-development deployment.

`/debug` mid-game and acting as another seat are how curl/Swagger play-testing
drives a game with no LLM. Outside development they would hand one client every
role, every private channel, and the AI seats' votes and night actions.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app.config as config
from app.config import Settings
from app.engine.phases import Phase
from app.main import app
from app.sessions.store import get_session_store

client = TestClient(app)


@pytest.fixture
def production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "_settings", Settings(werewolf_env="production"))


def _new_game() -> tuple[str, str]:
    body = client.post("/api/games", json={"human_name": "Tester", "seed": 5}).json()
    return body["session_id"], body["human_player_id"]


def test_development_keeps_the_play_testing_tools():
    session_id, _ = _new_game()

    assert client.get(f"/api/games/{session_id}/debug").status_code == 200
    view = client.get(f"/api/games/{session_id}/view", params={"player_id": "p5"})
    assert view.status_code == 200
    assert view.json()["your_player_id"] == "p5"
    assert view.json()["debug_available"] is True
    unknown = client.get(f"/api/games/{session_id}/view", params={"player_id": "p99"})
    assert unknown.status_code == 404


@pytest.mark.usefixtures("production")
def test_production_hides_debug_until_the_game_is_over():
    session_id, _ = _new_game()

    response = client.get(f"/api/games/{session_id}/debug")
    assert response.status_code == 409
    assert client.get(f"/api/games/{session_id}/view").json()["debug_available"] is False

    session = get_session_store().get(session_id)
    assert session is not None
    session.controller.state.phase = Phase.GAME_OVER
    assert client.get(f"/api/games/{session_id}/debug").status_code == 200
    assert client.get(f"/api/games/{session_id}/view").json()["debug_available"] is True


@pytest.mark.usefixtures("production")
def test_production_refuses_to_act_or_look_as_another_seat():
    session_id, human_id = _new_game()

    other = client.get(f"/api/games/{session_id}/view", params={"player_id": "p5"})
    assert other.status_code == 403
    response = client.post(
        f"/api/games/{session_id}/chat", params={"player_id": "p5"}, json={"content": "hi"}
    )
    assert response.status_code == 403
    # The human's own seat, named explicitly or not, still works.
    own = client.get(f"/api/games/{session_id}/view", params={"player_id": human_id})
    assert own.status_code == 200


@pytest.mark.usefixtures("production")
def test_production_only_opens_the_humans_own_websocket():
    session_id, human_id = _new_game()

    # The refusal arrives instead of the handshake's accept, so entering the
    # context raises; an accepted socket would not.
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect(f"/ws/{session_id}/p5"):
            pass
    assert refused.value.code == 4403

    with client.websocket_connect(f"/ws/{session_id}/{human_id}"):
        pass
