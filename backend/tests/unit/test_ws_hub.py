"""Who each pushed frame reaches.

The client only uses a frame as a cue to refetch its own filtered view, but
the frame itself carries the event payload. A wolf-channel message, or a
"typing in the wolf channel" indicator, pushed to every socket names the
wolves to anyone reading the WebSocket in devtools. These tests pin that a
private event reaches exactly the seats whose own view would show it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.api.ws_hub import SessionWSHub
from app.engine.events import GameEvent, GameEventType
from app.engine.game import GameController
from app.engine.roles import RoleName
from tests.conftest import make_controller


class _FakeSocket:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def accept(self) -> None:
        return None

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.frames.append(payload)


async def _connect_every_seat(
    controller: GameController,
) -> tuple[SessionWSHub, dict[str, _FakeSocket]]:
    hub = SessionWSHub()
    controller.events.subscribe(hub.on_event)
    sockets = {pid: _FakeSocket() for pid in controller.state.players}
    for pid, socket in sockets.items():
        await hub.connect(pid, socket)  # type: ignore[arg-type]
    return hub, sockets


async def _flush() -> None:
    """Let the hub's send tasks run."""
    for _ in range(3):
        await asyncio.sleep(0)


def _receivers(sockets: dict[str, _FakeSocket], kind: str, **match: object) -> set[str]:
    return {
        pid
        for pid, socket in sockets.items()
        if any(
            frame["type"] == kind
            and all(frame["payload"].get(key) == value for key, value in match.items())
            for frame in socket.frames
        )
    }


def _ids_with_role(controller: GameController, role: RoleName) -> set[str]:
    return {p.player_id for p in controller.state.players_by_role(role)}


async def test_wolf_channel_messages_reach_only_werewolves() -> None:
    controller = make_controller(seed=3)
    _, sockets = await _connect_every_seat(controller)
    wolves = _ids_with_role(controller, RoleName.WEREWOLF)
    author = sorted(wolves)[0]

    message_id = controller.chat(author, "今夜は占い師を噛もう", "wolf")
    await _flush()

    receivers = _receivers(sockets, "chat_message", message_id=message_id)
    assert receivers == wolves
    outsider = next(pid for pid in sockets if pid not in wolves)
    assert sockets[outsider].frames == []


async def test_freemason_channel_messages_reach_only_freemasons() -> None:
    controller = make_controller(seed=3)
    _, sockets = await _connect_every_seat(controller)
    masons = _ids_with_role(controller, RoleName.FREEMASON)

    message_id = controller.chat(sorted(masons)[0], "明日は私がCOする", "freemason")
    await _flush()

    assert _receivers(sockets, "chat_message", message_id=message_id) == masons


async def test_public_events_still_reach_every_seat() -> None:
    controller = make_controller(seed=3)
    _, sockets = await _connect_every_seat(controller)

    message_id = controller.chat("p0", "おはようございます")
    controller.set_typing("p1", True)
    controller.start_game()
    await _flush()

    everyone = set(sockets)
    assert _receivers(sockets, "chat_message", message_id=message_id) == everyone
    assert _receivers(sockets, "typing_changed", player_id="p1", channel="public") == everyone
    assert _receivers(sockets, "phase_changed", phase="night") == everyone


async def test_private_channel_typing_reaches_only_that_channel() -> None:
    """Typing in the wolf channel names a wolf as surely as the message does."""
    controller = make_controller(seed=3)
    _, sockets = await _connect_every_seat(controller)
    wolves = _ids_with_role(controller, RoleName.WEREWOLF)
    masons = _ids_with_role(controller, RoleName.FREEMASON)
    wolf, mason = sorted(wolves)[0], sorted(masons)[0]

    controller.set_typing(wolf, True, "wolf")
    controller.set_typing(wolf, False, "wolf")
    controller.set_typing(mason, True, "freemason")
    await _flush()

    assert _receivers(sockets, "typing_changed", player_id=wolf, typing=True) == wolves
    assert _receivers(sockets, "typing_changed", player_id=wolf, typing=False) == wolves
    assert _receivers(sockets, "typing_changed", player_id=mason) == masons


async def test_every_push_matches_what_the_receivers_own_view_shows() -> None:
    """The invariant behind the filtering: a seat is pushed a chat message iff
    its own `/view` shows that message. Dead members of a channel still read
    its log, so they still get its pushes."""
    controller = make_controller(seed=5)
    _, sockets = await _connect_every_seat(controller)
    wolves = sorted(_ids_with_role(controller, RoleName.WEREWOLF))
    masons = sorted(_ids_with_role(controller, RoleName.FREEMASON))
    sent = [
        controller.chat(wolves[0], "狼の相談", "wolf"),
        controller.chat(masons[0], "共有の相談", "freemason"),
        controller.chat("p0", "みんなへ"),
    ]
    # A dead wolf keeps reading the wolf channel in its view.
    controller.state.players[wolves[1]].alive = False
    sent.append(controller.chat(wolves[0], "あとは二人で", "wolf"))
    await _flush()

    for message_id in sent:
        shown_to = {
            pid
            for pid in sockets
            if any(
                item["message_id"] == message_id
                for item in [
                    *controller.get_player_view(pid)["public_chat"],
                    *controller.get_player_view(pid)["private_chat"],
                ]
            )
        }
        assert _receivers(sockets, "chat_message", message_id=message_id) == shown_to


async def test_a_single_recipient_result_still_reaches_only_its_seat() -> None:
    controller = make_controller(seed=3)
    _, sockets = await _connect_every_seat(controller)

    controller.events.publish(
        GameEvent(GameEventType.DIVINE_RESULT, {"target_id": "p2"}, recipient_id="p4")
    )
    await _flush()

    assert _receivers(sockets, "divine_result") == {"p4"}


async def test_an_empty_audience_means_nobody_not_everybody() -> None:
    controller = make_controller(seed=3)
    _, sockets = await _connect_every_seat(controller)

    controller.events.publish(GameEvent(GameEventType.CHAT_MESSAGE, {}, recipient_ids=()))
    controller.set_typing("p1", True, "not-a-channel")
    await _flush()

    assert all(socket.frames == [] for socket in sockets.values())


def test_event_audience() -> None:
    assert GameEvent(GameEventType.VICTORY).audience() is None
    assert GameEvent(GameEventType.DIVINE_RESULT, recipient_id="p1").audience() == {"p1"}
    assert GameEvent(GameEventType.CHAT_MESSAGE, recipient_ids=("p1", "p2")).audience() == {
        "p1",
        "p2",
    }
    assert GameEvent(GameEventType.CHAT_MESSAGE, recipient_ids=()).audience() == frozenset()
