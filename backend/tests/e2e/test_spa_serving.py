"""The backend serves the built SPA from the same origin, so the whole game
runs on one port (which is what makes the Codespaces single-forwarded-port
setup work). These tests pin the routing rules that make that safe."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.config as config
from app import main
from app.config import DEFAULT_ANTHROPIC_MODEL, Settings
from app.main import app

client = TestClient(app)

# Shaped like a real key so a leak through any formatting path would still match.
_SENTINEL_ANTHROPIC_KEY = "sk-ant-api03-SENTINEL-spa-serving-0123456789abcdef"


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


@pytest.mark.parametrize(
    ("password", "allow_open_access"),
    [("table-password", False), ("", True)],
    ids=["behind-password", "open-access-opt-in"],
)
def test_the_anthropic_api_key_never_reaches_health_or_the_spa(
    monkeypatch: pytest.MonkeyPatch, password: str, allow_open_access: bool
):
    """The key is in the server's settings for as long as it runs; nothing the
    browser can fetch -- the public health check, the HTML shell, any bundled
    asset, an error page -- may carry it."""
    monkeypatch.setattr(
        config,
        "_settings",
        Settings(
            werewolf_llm_provider="anthropic",
            werewolf_anthropic_api_key=_SENTINEL_ANTHROPIC_KEY,
            werewolf_access_password=password,
            werewolf_allow_open_access=allow_open_access,
        ),
    )
    keyed = TestClient(main.create_app())
    auth = ("werewolf", password) if password else None

    health = keyed.get("/api/health")
    assert health.status_code == 200
    assert health.json()["llm_provider"] == "anthropic"
    assert health.json()["llm_model"] == DEFAULT_ANTHROPIC_MODEL

    paths = ["/api/health", "/", "/index.html", "/some/client/route", "/api/nope", "/.env"]
    if _dist_built():
        paths += [
            "/" + asset.relative_to(main.FRONTEND_DIST).as_posix()
            for asset in main.FRONTEND_DIST.rglob("*")
            if asset.is_file()
        ]
    for path in paths:
        # Both as the owner and as a stranger without the password.
        for credentials in {auth, None}:
            resp = keyed.get(path, auth=credentials)
            assert _SENTINEL_ANTHROPIC_KEY not in resp.text, path
