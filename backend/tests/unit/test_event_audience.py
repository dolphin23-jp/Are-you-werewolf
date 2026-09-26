"""Private-channel events must only be pushed to that channel's seats.

`/view` already filters wolf and freemason chat per viewer; the live push used
to broadcast the same messages to every WebSocket, so any seat with devtools
open could read who the wolves were and what they planned.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.api.ws_hub import SessionWSHub
from app.engine.events import GameEvent, GameEventType
from app.engine.roles import RoleName
from tests.conftest import make_controller


def _started_controller():
    controller = make_controller(seed=1)
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    return controller


def _seats(controller, role: RoleName) -> tuple[str, ...]:
    return tuple(p.player_id for p in controller.state.players_by_role(role))


def _capture(controller) -> list[GameEvent]:
    events: list[GameEvent] = []
    controller.events.subscribe(events.append)
    return events


def test_private_chat_is_addressed_to_channel_members_only():
    controller = _started_controller()
    wolves = _seats(controller, RoleName.WEREWOLF)
    masons = _seats(controller, RoleName.FREEMASON)
    events = _capture(controller)

    controller.chat(wolves[0], "今夜はp3を噛もう", channel="wolf")
    controller.chat(masons[0], "相方です", channel="freemason")
    controller.chat("p0", "おはよう")

    chats = [e for e in events if e.type is GameEventType.CHAT_MESSAGE]
    assert [e.recipient_ids for e in chats] == [wolves, masons, None]


def test_private_typing_is_addressed_to_channel_members_only():
    controller = _started_controller()
    wolves = _seats(controller, RoleName.WEREWOLF)
    events = _capture(controller)

    controller.set_typing(wolves[0], True, "wolf")
    controller.set_typing("p0", True)

    typing = [e for e in events if e.type is GameEventType.TYPING_CHANGED]
    assert [e.recipient_ids for e in typing] == [wolves, None]


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def accept(self) -> None:
        return None

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


def test_hub_delivers_addressed_events_only_to_their_recipients():
    async def scenario() -> dict[str, _FakeSocket]:
        hub = SessionWSHub()
        sockets = {pid: _FakeSocket() for pid in ("p0", "p5", "p9")}
        for pid, socket in sockets.items():
            await hub.connect(pid, socket)  # type: ignore[arg-type]
        hub.on_event(GameEvent(GameEventType.CHAT_MESSAGE, {"x": 1}, recipient_ids=("p5",)))
        hub.on_event(GameEvent(GameEventType.CHAT_MESSAGE, {"x": 2}))
        hub.on_event(GameEvent(GameEventType.CHAT_MESSAGE, {"x": 3}, recipient_ids=()))
        await asyncio.sleep(0)
        return sockets

    sockets = asyncio.run(scenario())

    assert [m["payload"]["x"] for m in sockets["p5"].sent] == [1, 2]
    assert [m["payload"]["x"] for m in sockets["p0"].sent] == [2]
    assert [m["payload"]["x"] for m in sockets["p9"].sent] == [2]


def test_public_message_ids_do_not_count_private_messages():
    """Gaps in the public numbering used to reveal how much the wolves said."""
    controller = _started_controller()
    wolves = _seats(controller, RoleName.WEREWOLF)
    masons = _seats(controller, RoleName.FREEMASON)

    first = controller.chat("p0", "おはよう")
    controller.chat(wolves[0], "今夜の相談", channel="wolf")
    controller.chat(wolves[1], "了解", channel="wolf")
    mason = controller.chat(masons[0], "相方です", channel="freemason")
    second = controller.chat("p0", "議論しよう")

    assert (first, second) == ("m1", "m2")
    assert mason == "f1"
    assert len({m.message_id for m in controller.state.chat_log}) == 5
