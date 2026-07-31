"""The backend serves the built SPA from the same origin, so the whole game
runs on one port (which is what makes the Codespaces single-forwarded-port
setup work). These tests pin the routing rules that make that safe."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import main
from app.main import app

client = TestClient(app)


def _dist_built() -> bool:
    return main.FRONTEND_DIST.is_dir()


def test_api_routes_still_win_over_the_spa_catch_all():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_unknown_api_path_404s_instead_of_returning_html():
    """A typo'd endpoint must fail as a 404, not hand back the HTML shell --
    otherwise the client gets a JSON-parse error instead of a clear 404."""
    resp = client.get("/api/definitely-not-a-real-endpoint")
    assert resp.status_code == 404
    assert "text/html" not in resp.headers.get("content-type", "")


def test_health_reports_whether_the_frontend_bundle_is_present():
    body = client.get("/api/health").json()
    assert body["frontend_bundled"] == _dist_built()


def test_root_serves_the_spa_shell_when_built():
    if not _dist_built():
        import pytest

        pytest.skip("frontend not built; run `pnpm build` in frontend/")
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert '<div id="root"></div>' in resp.text


def test_client_side_route_falls_back_to_the_shell():
    if not _dist_built():
        import pytest

        pytest.skip("frontend not built; run `pnpm build` in frontend/")
    resp = client.get("/some/client/route")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_path_traversal_cannot_escape_the_dist_directory():
    if not _dist_built():
        import pytest

        pytest.skip("frontend not built; run `pnpm build` in frontend/")
    # Even if a client hand-crafts an escaping path, it must never read a
    # file outside the build output -- it falls back to the SPA shell.
    resp = client.get("/../backend/app/config.py")
    assert resp.status_code == 200
    assert "LUNA_API_KEY" not in resp.text
    assert "luna_api_key" not in resp.text
