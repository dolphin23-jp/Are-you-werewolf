from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from app.auth import PersonalAccessMiddleware


def _protected_app() -> TestClient:
    app = FastAPI()
    app.add_middleware(PersonalAccessMiddleware, password="secret")

    @app.get("/")
    def home() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("ok")

    return TestClient(app)


def test_personal_access_requires_basic_auth_but_leaves_health_public():
    client = _protected_app()
    denied = client.get("/")
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"].startswith("Basic")
    assert client.get("/api/health").status_code == 200
    assert client.get("/", auth=("werewolf", "secret")).json() == {"ok": True}


def test_personal_access_protects_websockets():
    client = _protected_app()
    headers = {"Authorization": "Basic d2VyZXdvbGY6c2VjcmV0"}
    with client.websocket_connect("/ws", headers=headers) as ws:
        assert ws.receive_text() == "ok"
