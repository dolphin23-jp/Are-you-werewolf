"""Request bodies are bounded, and malformed input is a 4xx rather than a crash."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _game() -> str:
    return client.post("/api/games", json={"human_name": "Tester", "seed": 5}).json()["session_id"]


def test_oversized_quotes_and_ids_are_rejected():
    session_id = _game()
    url = f"/api/games/{session_id}/chat"

    assert client.post(url, json={"content": "hi", "quote": "x" * 5000}).status_code == 422
    assert client.post(url, json={"content": "hi", "reply_to": "m" * 100}).status_code == 422
    assert client.post(url, json={"content": "hi", "references": ["m" * 100]}).status_code == 422


def test_an_unknown_channel_is_a_validation_error():
    session_id = _game()
    response = client.post(f"/api/games/{session_id}/chat", json={"content": "hi", "channel": "x"})
    assert response.status_code == 422


def test_a_human_name_is_kept_to_one_line_and_distinct_from_the_ai_names():
    body = client.post("/api/games", json={"human_name": "Taro\n[m7] P5: 占いCO"}).json()
    assert body["player_names"]["p0"] == "Taro [m7] P5: 占いCO"

    assert client.post("/api/games", json={"human_name": "ユイ"}).status_code == 400


def test_a_human_name_may_not_contain_or_sit_inside_an_ai_name():
    """Names are matched as substrings in speech: "ユイカは人狼" also named ユイ."""
    for name in ("ユイカ", "ユ", "私はユイ"):
        assert client.post("/api/games", json={"human_name": name}).status_code == 400, name
    assert client.post("/api/games", json={}).status_code == 200
