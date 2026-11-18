"""The game is still running: a fact about the living, at every check it passed.

The engine checks victory after every execution and at every dawn (night 0
included) and ends the game the moment the living wolves are zero or stand level
with everyone else (`app/engine/victory.py`, `GameController._check_victory`).
So every check that let the game go on said, of the seats alive at that moment:

    at least one wolf, and fewer wolves than non-wolves.

Nothing else can happen between two checks -- the only deaths are executions
and night deaths, and each is followed by a check -- so the checks below are
exactly the ones the engine ran. The "at least one wolf" half is stated for the
latest check only: wolves never come back, so it implies the earlier ones.

Public and hard. Without it the solver happily allowed "no living wolf" under a
seer claimant who had given white to every living seat (run #14 seed 1, ダイキ),
or "all three wolves alive among six" under a medium claimant who called every
execution white (seed 2, ケント) -- worlds in which the game would already be
over. With it those suppositions are impossible, and "if レン is real, ツムギ is
the only living wolf" (seed 3) follows.

Once the victory check has ended the game, the last check failed: its
constraint is left out, and the game no longer says anything about the living.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import Perspective
from app.ai.reasoning.solver.backend import RoleCountAmong
from app.ai.reasoning.solver.builder import ConstraintBuilder
from app.ai.reasoning.solver.rules.base import BaseRuleModule
from app.engine.roles import ROLE_DEFINITIONS, RoleName


@dataclass(frozen=True)
class VictoryCheck:
    """One victory check the engine ran, and who was alive right after it.

    `day` and `after` name the moment: `after="execution"` is the check after
    day `day`'s execution, `after="night"` the one at the dawn after night
    `day` (night 0's includes the first victim).
    """

    day: int
    after: str
    alive: tuple[str, ...]

    @property
    def label(self) -> str:
        if self.after == "execution":
            return f"{self.day}日目の処刑後"
        return f"{self.day + 1}日目の朝" if self.day >= 0 else "開始時"


def victory_checks(observations: ObservationSet) -> tuple[VictoryCheck, ...]:
    """Every check the engine has run so far, oldest first, from public deaths.

    A death the public record cannot place (no execution, night or first-victim
    entry) leaves the history unknowable, so only the present is returned.
    """
    when: dict[str, tuple[int, int]] = {}
    if observations.first_victim_id is not None:
        when[observations.first_victim_id] = (0, 1)
    for execution in observations.executions:
        when[execution.player_id] = (execution.day, 0)
    for death in observations.night_deaths:
        when[death.player_id] = (death.night, 1)
    dead = [pid for pid in observations.player_ids if not observations.is_alive(pid)]
    current = tuple(pid for pid in observations.player_ids if observations.is_alive(pid))
    if any(pid not in when for pid in dead):
        last_day = observations.day
        return (VictoryCheck(day=last_day, after="now", alive=current),)
    moments = sorted({when[pid] for pid in dead})
    checks: list[VictoryCheck] = []
    for day, half in moments:
        alive = tuple(
            pid
            for pid in observations.player_ids
            if pid not in when or when[pid] > (day, half)
        )
        checks.append(
            VictoryCheck(day=day, after="execution" if half == 0 else "night", alive=alive)
        )
    if not checks or checks[-1].alive != current:
        checks.append(VictoryCheck(day=observations.day, after="now", alive=current))
    return tuple(checks)


def passed_victory_checks(observations: ObservationSet) -> tuple[VictoryCheck, ...]:
    """The checks that let the game go on: all of them, unless the last ended it."""
    checks = victory_checks(observations)
    return checks[:-1] if observations.victory_decided else checks


class GameRunningRuleModule(BaseRuleModule):
    module_id = "game_running"

    def add_hard_constraints(
        self,
        builder: ConstraintBuilder,
        perspective: Perspective,
        observations: ObservationSet,
    ) -> None:
        checks = passed_victory_checks(observations)
        wolf = RoleName.WEREWOLF
        total = ROLE_DEFINITIONS[wolf].count
        for index, check in enumerate(checks):
            latest = index == len(checks) - 1
            most = (len(check.alive) - 1) // 2
            least = 1 if latest else 0
            if not latest and most >= total:
                continue  # says no more than the composition does
            self._record(
                builder,
                f"{self.module_id}:{check.day}:{check.after}",
                RoleCountAmong(
                    player_ids=check.alive, role=wolf, at_least=least, at_most=most
                ),
                (
                    f"試合は続いているので、生存者{len(check.alive)}人のうち人狼は"
                    f"{least}人以上{most}人以下です（人狼0人か、人狼以外と同数以上なら"
                    "その時点で試合が終わっています）。"
                    if latest
                    else f"{check.label}に試合が終わらなかったので、そのときの生存者"
                    f"{len(check.alive)}人のうち人狼は{most}人以下でした。"
                ),
            )


__all__ = [
    "GameRunningRuleModule",
    "VictoryCheck",
    "passed_victory_checks",
    "victory_checks",
]
