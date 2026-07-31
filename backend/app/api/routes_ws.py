from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.sessions.store import get_session_store

router = APIRouter(tags=["ws"])


@router.websocket("/ws/{session_id}/{player_id}")
async def game_ws(websocket: WebSocket, session_id: str, player_id: str) -> None:
    session = get_session_store().get(session_id)
    if session is None:
        await websocket.close(code=4404)
        return

    await session.ws_hub.connect(player_id, websocket)
    try:
        while True:
            # The client doesn't need to send anything; we just keep the
            # connection open and drain any keepalive pings it sends.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        session.ws_hub.disconnect(player_id, websocket)
