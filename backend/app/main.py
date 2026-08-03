from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routes_game import router as game_router
from app.api.routes_ws import router as ws_router
from app.auth import PersonalAccessMiddleware
from app.config import get_settings

# backend/app/main.py -> backend/app -> backend -> <repo root>
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app() -> FastAPI:
    settings = get_settings()
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

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "llm_provider": settings.werewolf_llm_provider,
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
