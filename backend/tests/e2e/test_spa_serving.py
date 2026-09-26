"""The backend serves the built SPA from the same origin, so the whole game
runs on one port (which is what makes the Codespaces single-forwarded-port
setup work). These tests pin the routing rules that make that safe.

They build a throwaway `dist` instead of relying on `pnpm build`: CI's backend
job never builds the frontend, so the SPA and path-traversal checks used to be
skipped on every run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import app

client = TestClient(app)


@pytest.fixture
def dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    built = tmp_path / "frontend" / "dist"
    (built / "assets").mkdir(parents=True)
    (built / "index.html").write_text('<html><div id="root"></div></html>', encoding="utf-8")
    (built / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    (tmp_path / "frontend" / "secret.txt").write_text("LUNA_API_KEY=sk-live", encoding="utf-8")
    monkeypatch.setattr(main, "FRONTEND_DIST", built)
    return built


def test_api_routes_still_win_over_the_spa_catch_all(dist: Path):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.parametrize("path", ["/api/definitely-not-a-real-endpoint", "/api", "/ws"])
def test_unknown_api_and_ws_paths_404_instead_of_returning_html(dist: Path, path: str):
    """A typo'd endpoint must fail as a 404, not hand back the HTML shell --
    otherwise the client gets a JSON-parse error instead of a clear 404."""
    resp = client.get(path)
    assert resp.status_code == 404
    assert "text/html" not in resp.headers.get("content-type", "")


def test_a_post_to_an_unknown_api_path_is_a_404_not_a_405(dist: Path):
    assert client.post("/api/definitely-not-a-real-endpoint", json={}).status_code == 404


def test_health_reports_whether_the_frontend_bundle_is_present(dist: Path, monkeypatch):
    assert client.get("/api/health").json()["frontend_bundled"] is True
    monkeypatch.setattr(main, "FRONTEND_DIST", dist / "missing")
    assert client.get("/api/health").json()["frontend_bundled"] is False


def test_root_serves_the_spa_shell(dist: Path):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert '<div id="root"></div>' in resp.text


def test_built_assets_are_served_as_files(dist: Path):
    assert client.get("/assets/app.js").text == "console.log('ok')"


def test_client_side_route_falls_back_to_the_shell(dist: Path):
    resp = client.get("/some/client/route")
    assert resp.status_code == 200
    assert '<div id="root"></div>' in resp.text


@pytest.mark.parametrize("path", ["/../secret.txt", "/%2e%2e/secret.txt", "/%00"])
def test_path_traversal_and_nul_bytes_fall_back_to_the_shell(dist: Path, path: str):
    resp = client.get(path)
    assert resp.status_code in (200, 404)
    assert "LUNA_API_KEY" not in resp.text
