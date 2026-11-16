from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes_game import router as game_router
from app.api.routes_ws import router as ws_router
from app.auth import PersonalAccessMiddleware
from app.config import Settings, get_settings

# backend/app/main.py -> backend/app -> backend -> <repo root>
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

_LOGGER = logging.getLogger(__name__)

# Every provider whose calls bill an API key. `luna` is dormant, but a server
# started with it bills its key exactly as `anthropic` does.
BILLED_PROVIDERS = frozenset({"anthropic", "luna"})

OPEN_ACCESS_REFUSAL = (
    "WEREWOLF_LLM_PROVIDER が課金されるプロバイダ(anthropic・luna)で "
    "WEREWOLF_ACCESS_PASSWORD が空のため、起動を中止しました。"
    "このままでは、このサーバーにアクセスできる人なら誰でも、APIキーに課金されるゲームを作れます。"
    "WEREWOLF_ACCESS_PASSWORD を設定してください。自分だけが使うローカル環境に限り、"
    "WEREWOLF_ALLOW_OPEN_ACCESS=true で外せます。 / "
    "Refusing to start: a billed WEREWOLF_LLM_PROVIDER (anthropic, luna) with an "
    "empty WEREWOLF_ACCESS_PASSWORD would let anyone who can reach this server "
    "start games billed to the API key. Set WEREWOLF_ACCESS_PASSWORD, or, only on "
    "a local machine that nobody else can reach, set WEREWOLF_ALLOW_OPEN_ACCESS=true."
)


def _serves_billed_games_openly(settings: Settings) -> bool:
    return (
        settings.werewolf_llm_provider in BILLED_PROVIDERS
        and not settings.werewolf_access_password
    )


def _refuse_open_paid_server(settings: Settings) -> None:
    """A paid provider behind no password is an open tab on the owner's account.

    Checked at startup rather than per request: a server that starts and then
    refuses games looks healthy, and the first sign of the problem would be
    the bill. The opt-out exists for a laptop nobody else can reach. It covers
    every billed provider, not only the default one: the README's luna
    section never mentions a password, and a luna game bills its key all the
    same.
    """
    if _serves_billed_games_openly(settings) and not settings.werewolf_allow_open_access:
        raise RuntimeError(OPEN_ACCESS_REFUSAL)


def _configure_logging(settings: Settings) -> None:
    """Give the app's loggers somewhere to go when nothing else has.

    uvicorn configures only its own loggers, so a warning from
    `app.ai.provider.anthropic` (a retry, an unpriced model, a refused schema)
    reached the root logger, found no handler, and was printed bare by
    Python's last-resort handler or, below WARNING, dropped. Leaves an
    already-configured root (a test runner, `--log-config`) untouched.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    name = settings.werewolf_log_level.strip().upper()
    level = logging.getLevelNamesMapping().get(name)
    logging.basicConfig(
        level=level if level is not None else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if level is None:
        _LOGGER.warning("unknown WEREWOLF_LOG_LEVEL %r; using INFO", settings.werewolf_log_level)


def _llm_model(settings: Settings) -> str | None:
    """The model a real provider asks for; None for the offline doubles."""
    if settings.werewolf_llm_provider == "anthropic":
        return settings.werewolf_anthropic_model
    if settings.werewolf_llm_provider == "luna":
        return settings.luna_model
    return None


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings)
    _refuse_open_paid_server(settings)
    if _serves_billed_games_openly(settings):
        _LOGGER.warning(
            "WEREWOLF_ALLOW_OPEN_ACCESS=true: serving a paid provider (%s) with no access "
            "password; anyone who can reach this server can start billed games",
            settings.werewolf_llm_provider,
        )
    app = FastAPI(title="Are you werewolf API", version="0.1.0")
    if settings.werewolf_access_password:
        app.add_middleware(
            PersonalAccessMiddleware,
            password=settings.werewolf_access_password,
        )

    # Only needed when the frontend is served from a different origin than
    # this API. In the default single-service setup it is inert, because
    # the SPA below is served from this very app.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(game_router)
    app.include_router(ws_router)

    # Public even behind the access password (see `PersonalAccessMiddleware`),
    # so it names the provider and model and nothing else: never a key, a
    # base URL or a budget.
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "llm_provider": settings.werewolf_llm_provider,
            "llm_model": _llm_model(settings),
            "reasoning_engine": settings.werewolf_reasoning_engine,
            "frontend_bundled": FRONTEND_DIST.is_dir(),
        }

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA from this same app, so the whole game runs on one
    origin/port. Registered last so it never shadows the API or WebSocket
    routes. A missing build directory is not an error -- the API still works
    (that is the state during backend-only tests and before `pnpm build`)."""

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> FileResponse:
        if not FRONTEND_DIST.is_dir():
            raise HTTPException(
                status_code=404,
                detail="frontend build not found; run `pnpm build` in frontend/",
            )

        # Unknown /api and /ws paths must 404 as such rather than silently
        # returning the HTML shell, which would turn a typo'd endpoint into
        # a confusing JSON-parse error on the client.
        if full_path.startswith(("api/", "ws/")):
            raise HTTPException(status_code=404, detail="not found")

        candidate = (FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST.resolve()):
            return FileResponse(candidate)

        # Any other path is a client-side route: hand back the SPA shell.
        return FileResponse(FRONTEND_DIST / "index.html")


app = create_app()
