"""Regression test: with a WebSocket client connected, every route that
mutates game state must still succeed. This specifically catches the bug
where a sync `def` route (dispatched by FastAPI on a worker thread with no
running event loop) crashed inside `SessionWSHub.on_event`'s
`asyncio.create_task` call -- invisible unless a client is actually
connected, since `on_event` is a no-op fan-out over an empty connection
dict otherwise."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_mutating_routes_succeed_with_a_websocket_client_connected():
    resp = client.post("/api/games", json={"human_name": "Tester", "seed": 42})
    body = resp.json()
    session_id = body["session_id"]
    human_id = body["human_player_id"]

    with client.websocket_connect(f"/ws/{session_id}/{human_id}"):
        resp = client.post(f"/api/games/{session_id}/start")
        assert resp.status_code == 200

        debug = client.get(f"/api/games/{session_id}/debug").json()
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

        resp = client.post(f"/api/games/{session_id}/chat", json={"content": "こんにちは"})
        assert resp.status_code == 200

        resp = client.post(f"/api/games/{session_id}/co", json={"claimed_role": "villager"})
        assert resp.status_code == 200

        resp = client.post(f"/api/games/{session_id}/end-discussion")
        assert resp.status_code == 200

        debug = client.get(f"/api/games/{session_id}/debug").json()
        assert debug["phase"] in ("voting", "runoff", "vote_result", "game_over")
