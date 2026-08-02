"""API-level smoke test: drive one full game via HTTP only, exactly as the
real frontend would (the human calls chat/vote/night-action for their own
seat; the AI coordinator -- backed by MockProvider by default, zero cost --
handles the other 16 seats automatically via the orchestrator hooks)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.engine.phases import Phase
from app.main import app
from app.sessions.models import DiscussionRoundState
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


def test_discussion_runs_without_waiting_when_human_is_dead():
    response = client.post("/api/games", json={"human_name": "Spectator", "seed": 44})
    session_id = response.json()["session_id"]
    session = get_session_store().get(session_id)
    assert session is not None
    session.controller.state.phase = Phase.DAWN
    session.controller.state.day = 1
    session.controller.state.players[session.human_id].alive = False

    response = client.post(f"/api/games/{session_id}/start-discussion")

    assert response.status_code == 200
    assert any(
        message.author_id != session.human_id
        for message in session.controller.state.chat_log
    )
    view = client.get(f"/api/games/{session_id}/view").json()
    assert view["awaiting_your_speech"] is False
    assert view["discussion_progress"]["spoken"] > 0


def test_view_is_filtered_but_debug_is_not():
    resp = client.post("/api/games", json={"human_name": "Tester", "seed": 5})
    session_id = resp.json()["session_id"]
    human_id = resp.json()["human_player_id"]

    view = client.get(f"/api/games/{session_id}/view", params={"player_id": human_id}).json()
    assert "your_role" in view
    assert "role" not in view["players"][0]

    debug = client.get(f"/api/games/{session_id}/debug").json()
    assert "role" in debug["players"][0]


def test_view_reports_human_speech_deadline_countdown():
    response = client.post("/api/games", json={"human_name": "Waiting", "seed": 8})
    session_id = response.json()["session_id"]
    session = get_session_store().get(session_id)
    assert session is not None
    session.discussion_round = DiscussionRoundState(
        day=1,
        order=[],
        awaiting_human=True,
        awaiting_since=time.time() - 10,
    )

    view = client.get(f"/api/games/{session_id}/view").json()

    assert view["awaiting_your_speech"] is True
    assert 34 <= view["speech_wait_remaining_seconds"] <= 35


def test_discussion_can_be_paused_without_an_active_generation_task():
    response = client.post("/api/games", json={"human_name": "Reader", "seed": 18})
    session_id = response.json()["session_id"]

    response = client.post(
        f"/api/games/{session_id}/discussion-control", json={"action": "pause"}
    )

    assert response.status_code == 200
    view = client.get(f"/api/games/{session_id}/view").json()
    assert view["discussion_paused"] is True


def test_discussion_control_rejects_unknown_actions():
    response = client.post("/api/games", json={"human_name": "Reader", "seed": 19})
    session_id = response.json()["session_id"]

    response = client.post(
        f"/api/games/{session_id}/discussion-control", json={"action": "rewind"}
    )

    assert response.status_code == 400


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


def test_compact_human_seer_co_registers_target_and_result():
    response = client.post("/api/games", json={"human_name": "Compact", "seed": 27})
    session_id = response.json()["session_id"]
    human_id = response.json()["human_player_id"]
    session = get_session_store().get(session_id)
    assert session is not None
    session.controller.state.phase = Phase.DISCUSSION
    target_id = "p1"
    target_name = session.controller.state.players[target_id].name

    response = client.post(
        f"/api/games/{session_id}/chat",
        json={"content": f"占いCO{target_name}は人狼ではない。理由は初日なので特にない。"},
    )

    assert response.status_code == 200
    view = client.get(f"/api/games/{session_id}/view").json()
    assert {"player_id": human_id, "claimed_role": "seer", "day": 0} in view[
        "co_declarations"
    ]
    assert {
        "claimant_id": human_id,
        "result_type": "seer",
        "target_id": target_id,
        "is_werewolf": False,
        "day": 0,
    } in view["public_result_claims"]


def test_human_shared_reveal_and_partner_confirmation_are_tracked():
    response = client.post("/api/games", json={"human_name": "共有本人", "seed": 26})
    session_id = response.json()["session_id"]
    human_id = response.json()["human_player_id"]
    session = get_session_store().get(session_id)
    assert session is not None
    session.coordinator = None
    session.controller.state.phase = Phase.DISCUSSION
    partner_id = "p11"
    partner_name = session.controller.state.players[partner_id].name
    human_name = session.controller.state.players[human_id].name

    client.post(f"/api/games/{session_id}/chat", json={"content": "共有者CO、相方生存"})
    reveal = client.post(
        f"/api/games/{session_id}/chat",
        json={"content": f"相方は{partner_name}({partner_id})です"},
    )
    confirmation = client.post(
        f"/api/games/{session_id}/chat",
        params={"player_id": partner_id},
        json={
            "content": (
                f"{human_name}({human_id})の共有者CO、相方は私{partner_name}で間違いありません"
            )
        },
    )

    assert reveal.status_code == 200
    assert confirmation.status_code == 200
    view = client.get(f"/api/games/{session_id}/view").json()
    assert view["freemason_partner_claims"] == [
        {
            "claimant_id": human_id,
            "partner_id": partner_id,
            "day": 0,
            "confirmed": True,
        }
    ]


def test_public_view_conceals_night_death_cause():
    resp = client.post("/api/games", json={"human_name": "Viewer", "seed": 16})
    session_id = resp.json()["session_id"]
    session = get_session_store().get(session_id)
    assert session is not None
    from app.engine.state import DeathCause

    player = session.controller.state.players["p1"]
    player.alive = False
    player.death_cause = DeathCause.CURSED
    player.death_day = 1

    public = client.get(f"/api/games/{session_id}/view").json()
    debug = client.get(f"/api/games/{session_id}/debug").json()
    assert next(item for item in public["players"] if item["player_id"] == "p1")[
        "death_cause"
    ] == "night_death"
    assert next(item for item in debug["players"] if item["player_id"] == "p1")[
        "death_cause"
    ] == "cursed"


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
