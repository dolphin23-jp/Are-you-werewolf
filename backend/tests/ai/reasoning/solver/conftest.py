"""A real dealt board, shared by the solver tests.

Built once per module: the deal is seeded and the solver never mutates it, so
rebuilding it per test would only slow the suite down.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.ai.reasoning.observations import ObservationSet
from app.engine.roles import RoleName
from tests.conftest import make_controller


@dataclass(frozen=True)
class Board:
    observations: ObservationSet

    def seats(self, role: RoleName) -> tuple[str, ...]:
        return tuple(
            pid for pid, dealt in self.observations.true_roles.items() if dealt == role
        )

    def one(self, role: RoleName) -> str:
        return self.seats(role)[0]

    @property
    def first_victim_id(self) -> str:
        victim = self.observations.first_victim_id
        assert victim is not None
        return victim


@pytest.fixture(scope="module")
def board() -> Board:
    controller = make_controller(seed=4)
    controller.start_game()  # deals the scripted Day-0 victim
    return Board(observations=ObservationSet.from_state(controller.state))
