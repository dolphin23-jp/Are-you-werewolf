"""A night always resolves with a bite once the AI alpha has had its turn.

The engine now refuses a night with a living wolf and no bite (the board
memo's 「死体なし＝護衛で防がれたか、襲撃先が妖狐」 rests on that rule). The
coordinator used to swallow a refused resolution, so a night whose bite went
nowhere would have stalled the game without a sign. These pin the code bite
that fills the gap and the event that counts a night that still cannot resolve.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.ai.coordinator import AICoordinator
from app.ai.metrics import MetricsCollector
from app.ai.provider.mock import MockProvider
from app.engine.phases import Phase
from app.engine.roles import RoleName
from tests.conftest import make_controller


def _night_one():  # type: ignore[no-untyped-def]
    controller = make_controller(seed=4)
    state = controller.state
    controller.start_game()
    controller.resolve_night()
    controller.start_discussion()
    controller.end_discussion()
    target = next(
        pid for pid in state.alive_ids() if state.players[pid].role is RoleName.VILLAGER
    )
    for voter in state.alive_ids():
        if voter != target:
            controller.vote(voter, target)
    controller.resolve_votes()
    controller.start_night()
    metrics = MetricsCollector()
    ai_ids = [pid for pid in state.players if pid != "p0"]
    coordinator = AICoordinator(state, ai_ids, MockProvider(seed=1, metrics=metrics), seed=1)
    return controller, state, coordinator, metrics


def test_an_ai_alpha_whose_own_bite_went_nowhere_still_bites_someone() -> None:
    controller, state, coordinator, metrics = _night_one()

    async def bite_nowhere(*_args: object, **_kwargs: object) -> None:
        return None

    coordinator._cast_attack = bite_nowhere  # type: ignore[method-assign]
    assert controller.alpha_wolf_id in coordinator._agents  # seed 4: an AI alpha

    asyncio.run(coordinator.run_night_phase(SimpleNamespace(controller=controller)))

    assert state.phase is not Phase.NIGHT
    bites = [record for record in state.attack_records if record.day == 1]
    assert len(bites) == 1
    assert state.players[bites[0].target_id].role is not RoleName.WEREWOLF
    assert metrics.events.get("night_code_resolved", 0) >= 1
    assert "night_not_resolved" not in metrics.events


def test_a_night_that_cannot_resolve_is_counted_not_silent() -> None:
    controller, state, coordinator, metrics = _night_one()

    async def bite_nowhere(*_args: object, **_kwargs: object) -> None:
        return None

    coordinator._cast_attack = bite_nowhere  # type: ignore[method-assign]
    coordinator._attack_anyone = lambda *_args: None  # type: ignore[method-assign]

    asyncio.run(coordinator.run_night_phase(SimpleNamespace(controller=controller)))

    assert state.phase is Phase.NIGHT
    assert metrics.events.get("night_not_resolved") == 1
