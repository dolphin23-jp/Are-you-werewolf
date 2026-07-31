"""API-level smoke test: drive one full game via HTTP only, exactly as the
real frontend would (the human calls chat/vote/night-action for their own
seat; the AI coordinator -- backed by MockProvider by default, zero cost --
handles the other 16 seats automatically via the orchestrator hooks)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.engine.phases import Phase
from app.main import app
from app.sessions.store import get_session_store

client = TestClient(app)


def test_full_game_via_api_reaches_a_terminal_or_later_phase():
    resp = client.post("/api/games", json={"human_name": "Tester", "seed": 123})
    assert resp.status_code == 200
    body = resp.json()
    session_id = body["session_id"]
    human_id = body["human_player_id"]
    assert len(body["player_names"]) == 17

    resp = client.post(f"/api/games/{session_id}/start")
    assert resp.status_code == 200

    debug = client.get(f"/api/games/{session_id}/debug").json()
    # The AI coordinator (MockProvider) auto-advances the Day-0 night unless
    # the human happens to be the seer, in which case it waits for us.
    if debug["phase"] == "night" and debug["day"] == 0:
        human_role = next(p["role"] for p in debug["players"] if p["player_id"] == human_id)
        if human_role == "seer":
            other = next(p["player_id"] for p in debug["players"] if p["player_id"] != human_id)
            resp = client.post(
                f"/api/games/{session_id}/night-action",
                json={"action_type": "divine", "target_id": other},
            )
            assert resp.status_code == 200

    debug = client.get(f"/api/games/{session_id}/debug").json()
    assert debug["phase"] == "dawn"

    resp = client.post(f"/api/games/{session_id}/start-discussion")
    assert resp.status_code == 200

    resp = client.post(f"/api/games/{session_id}/chat", json={"content": "よろしくお願いします"})
    assert resp.status_code == 200

    resp = client.post(f"/api/games/{session_id}/end-discussion")
    assert resp.status_code == 200

    debug = client.get(f"/api/games/{session_id}/debug").json()
    assert debug["phase"] in ("voting", "runoff", "vote_result", "game_over")

    # The mock AI already talked during discussion.
    assert len(debug["chat_log"]) > 0


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_view_is_filtered_but_debug_is_not():
    resp = client.post("/api/games", json={"human_name": "Tester", "seed": 5})
    session_id = resp.json()["session_id"]
    human_id = resp.json()["human_player_id"]

    view = client.get(f"/api/games/{session_id}/view", params={"player_id": human_id}).json()
    assert "your_role" in view
    assert "role" not in view["players"][0]

    debug = client.get(f"/api/games/{session_id}/debug").json()
    assert "role" in debug["players"][0]


def test_public_chat_role_claim_is_added_to_public_information():
    resp = client.post("/api/games", json={"human_name": "Claimant", "seed": 15})
    session_id = resp.json()["session_id"]
    human_id = resp.json()["human_player_id"]
    session = get_session_store().get(session_id)
    assert session is not None
    session.controller.state.phase = Phase.DISCUSSION

    response = client.post(
        f"/api/games/{session_id}/chat", json={"content": "私が占い師です。結果を話します"}
    )

    assert response.status_code == 200
    view = client.get(f"/api/games/{session_id}/view").json()
    assert {"player_id": human_id, "claimed_role": "seer", "day": 0} in view[
        "co_declarations"
    ]


def test_analysis_transcript_is_available_only_after_game_over():
    resp = client.post("/api/games", json={"human_name": "Analyst", "seed": 9})
    session_id = resp.json()["session_id"]

    before_game_over = client.get(f"/api/games/{session_id}/transcript")
    assert before_game_over.status_code == 409

    session = get_session_store().get(session_id)
    assert session is not None
    session.controller.state.phase = Phase.GAME_OVER

    transcript = client.get(f"/api/games/{session_id}/transcript")
    assert transcript.status_code == 200
    body = transcript.json()
    assert body["seed"] == 9
    assert body["provider"] == "MockProvider"
    assert len(body["names"]) == 17
    assert len(body["roles"]) == 17
    assert body["final_state"]["phase"] == "game_over"
