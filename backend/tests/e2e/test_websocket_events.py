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
from app.sessions.store import get_session_store

client = TestClient(app)


def test_a_wolf_channel_frame_never_reaches_a_non_wolf_socket():
    """Over real sockets: a non-wolf's next frame after a wolf-channel message
    is the public message that followed it, while a wolf receives both.

    The AI coordinator is detached so that only the two messages sent here
    produce frames, which keeps the receive order deterministic (a missing
    frame shows up as the wrong frame, never as a hang).
    """
    resp = client.post("/api/games", json={"human_name": "Tester", "seed": 42})
    session_id = resp.json()["session_id"]
    session = get_session_store().get(session_id)
    assert session is not None
    session.coordinator = None
    roles = {
        p["player_id"]: p["role"]
        for p in client.get(f"/api/games/{session_id}/debug").json()["players"]
    }
    wolf = next(pid for pid, role in roles.items() if role == "werewolf")
    outsider = next(pid for pid, role in roles.items() if role not in ("werewolf", "freemason"))

    with (
        client.websocket_connect(f"/ws/{session_id}/{outsider}") as outsider_ws,
        client.websocket_connect(f"/ws/{session_id}/{wolf}") as wolf_ws,
    ):
        resp = client.post(
            f"/api/games/{session_id}/chat",
            params={"player_id": wolf},
            json={"content": "今夜の噛み先を決めよう", "channel": "wolf"},
        )
        assert resp.status_code == 200
        resp = client.post(
            f"/api/games/{session_id}/chat",
            params={"player_id": outsider},
            json={"content": "おはようございます"},
        )
        assert resp.status_code == 200

        outsider_first = outsider_ws.receive_json()
        assert outsider_first["type"] == "chat_message"
        assert outsider_first["payload"]["channel"] == "public"
        assert outsider_first["payload"]["content"] == "おはようございます"

        wolf_first = wolf_ws.receive_json()
        assert wolf_first["payload"]["channel"] == "wolf"
        assert wolf_first["payload"]["content"] == "今夜の噛み先を決めよう"
        wolf_second = wolf_ws.receive_json()
        assert wolf_second["payload"]["channel"] == "public"


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
