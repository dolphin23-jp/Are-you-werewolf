"""Checking a published account against the calendar the whole table can see.

A verdict is not just a colour. It is a claim that on a particular night a
particular ability was used on a particular player, and the table already knows
who was alive that night and who was executed which day. When those two do not
line up -- two divines on one night, a medium result for a day nobody was
executed, a look taken at someone already buried -- something in the account is
wrong.

What that something *is* stays open on purpose. A real seer can misremember a
night, or lie about the timing to protect an unpublished result; a wolf can
invent a clean-looking schedule and get it right. So nothing here produces a
hard rule. Conflicts are reported as facts about the *account*, and the two
consumers decide what to do with them:

* the belief engine treats a conflict as ordinary soft evidence, weighted by
  how much the listener trusts the speaker;
* `AccurateTimeline` turns them into constraints only for a caller who has
  explicitly supposed that a claimant's timing is accurate -- which is exactly
  what a bluffer checking their own story wants to know.

Every check is deliberately conservative. Where a legal timeline and an illegal
one are indistinguishable from the public record -- a seer divines on the same
night they are attacked, and from outside those are simultaneous -- no conflict
is reported. Inventing a contradiction is worse than missing one.

Only public observations are read: published verdicts, announced executions,
announced deaths, the current day. There is no route from here to a true role.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.reasoning.observations import ObservationSet, PublicVerdict
from app.engine.roles import RoleName

SEER_RESULT = "seer"
MEDIUM_RESULT = "medium"


class ConflictKind:
    """Why an account does not fit the calendar."""

    DUPLICATE_NIGHT = "duplicate_night"
    AFTER_OWN_DEATH = "after_own_death"
    FUTURE_NIGHT = "future_night"
    NO_EXECUTION = "no_execution"
    WRONG_EXECUTION = "wrong_execution"
    TARGET_ALREADY_DEAD = "target_already_dead"


@dataclass(frozen=True)
class TimelineConflict:
    """One published result that does not fit the public calendar."""

    claimant_id: str
    claimed_role: RoleName | None
    kind: str
    night: int
    target_id: str
    explanation: str
    source_message_ids: tuple[str, ...] = ()

    @property
    def conflict_id(self) -> str:
        return f"timeline:{self.claimant_id}:{self.kind}:{self.night}:{self.target_id}"


# Every seat asks the same question of the same board, so the answer is
# computed once per board version rather than sixteen times. Bounded because a
# long game is still only a few dozen versions, and an unbounded cache keyed on
# game state is a leak with extra steps.
_CACHE_LIMIT = 64
_conflict_cache: dict[str, tuple[TimelineConflict, ...]] = {}


def _execution_day(observations: ObservationSet, player_id: str) -> int | None:
    return next((e.day for e in observations.executions if e.player_id == player_id), None)


def _night_death(observations: ObservationSet, player_id: str) -> int | None:
    return next((d.night for d in observations.night_deaths if d.player_id == player_id), None)


def can_act_on_night(observations: ObservationSet, player_id: str, night: int) -> bool:
    """Whether a submitted action is legal, using public timing only.

    Execution precedes that day's night; an overnight death is simultaneous
    with already submitted actions and therefore only prevents later nights.
    """
    execution = _execution_day(observations, player_id)
    death = _night_death(observations, player_id)
    return not (
        (execution is not None and execution <= night)
        or (death is not None and death < night)
    )


def can_be_targeted_on_night(observations: ObservationSet, player_id: str, night: int) -> bool:
    if night == 0 and player_id == observations.first_victim_id:
        return False
    return can_act_on_night(observations, player_id, night)


def could_publish_on_day(observations: ObservationSet, player_id: str, day: int) -> bool:
    execution = _execution_day(observations, player_id)
    death = _night_death(observations, player_id)
    return not ((execution is not None and execution < day) or (death is not None and death < day))


def disclosable_result_nights(
    observations: ObservationSet, player_id: str, as_of_day: int
) -> frozenset[int]:
    """Result nights whose owner lived to a following public discussion."""
    return frozenset(
        night
        for night in range(max(0, as_of_day))
        if can_act_on_night(observations, player_id, night)
        and could_publish_on_day(observations, player_id, night + 1)
    )


def find_timeline_conflicts(
    observations: ObservationSet,
) -> tuple[TimelineConflict, ...]:
    """Every published result that contradicts what the table can already see."""
    key = observations.board_version
    if key:
        cached = _conflict_cache.get(key)
        if cached is not None:
            return cached
    conflicts: list[TimelineConflict] = []
    for claimant_id in observations.player_ids:
        conflicts.extend(_conflicts_for(observations, claimant_id))
    result = tuple(conflicts)
    if key:
        if len(_conflict_cache) >= _CACHE_LIMIT:
            _conflict_cache.clear()
        _conflict_cache[key] = result
    return result


def conflicts_by_claimant(
    observations: ObservationSet,
) -> dict[str, tuple[TimelineConflict, ...]]:
    grouped: dict[str, list[TimelineConflict]] = {}
    for conflict in find_timeline_conflicts(observations):
        grouped.setdefault(conflict.claimant_id, []).append(conflict)
    return {claimant: tuple(items) for claimant, items in grouped.items()}


def _conflicts_for(
    observations: ObservationSet, claimant_id: str
) -> list[TimelineConflict]:
    verdicts = observations.verdicts_by(claimant_id)
    if not verdicts:
        return []
    role = observations.claimed_role_of(claimant_id)
    conflicts: list[TimelineConflict] = []
    seer_nights: dict[int, str] = {}
    for verdict in sorted(verdicts, key=lambda v: (v.day, v.target_id)):
        if verdict.result_type == SEER_RESULT:
            conflicts.extend(_seer_conflicts(observations, verdict, role, seer_nights))
        elif verdict.result_type == MEDIUM_RESULT:
            conflicts.extend(_medium_conflicts(observations, verdict, role))
    return conflicts


def _conflict(
    verdict: PublicVerdict, role: RoleName | None, kind: str, explanation: str
) -> TimelineConflict:
    return TimelineConflict(
        claimant_id=verdict.claimant_id,
        claimed_role=role,
        kind=kind,
        night=verdict.source_night,
        target_id=verdict.target_id,
        explanation=explanation,
        source_message_ids=(
            (verdict.source_message_id,) if verdict.source_message_id else ()
        ),
    )


def _seer_conflicts(
    observations: ObservationSet,
    verdict: PublicVerdict,
    role: RoleName | None,
    seer_nights: dict[int, str],
) -> list[TimelineConflict]:
    claimant = verdict.claimant_id
    night = verdict.source_night
    conflicts: list[TimelineConflict] = []

    previous = seer_nights.get(night)
    if previous is not None and previous != verdict.target_id:
        conflicts.append(
            _conflict(
                verdict,
                role,
                ConflictKind.DUPLICATE_NIGHT,
                f"{claimant}は{night}日目の夜について{previous}と{verdict.target_id}の"
                "2件の占い結果を出しています。占いは一晩に1人だけです。",
            )
        )
    seer_nights.setdefault(night, verdict.target_id)

    if night >= verdict.day:
        conflicts.append(
            _conflict(
                verdict,
                role,
                ConflictKind.FUTURE_NIGHT,
                f"{claimant}はまだ訪れていない{night}日目の夜の占い結果を主張しています。",
            )
        )

    own_death = observations.death_day_of(claimant)
    if not can_act_on_night(observations, claimant, night):
        conflicts.append(
            _conflict(
                verdict,
                role,
                ConflictKind.AFTER_OWN_DEATH,
                f"{claimant}は{own_death}日目に死亡していますが、"
                f"{night}日目の夜の占い結果を主張しています。",
            )
        )

    target_death = observations.death_day_of(verdict.target_id)
    if not can_be_targeted_on_night(observations, verdict.target_id, night):
        conflicts.append(
            _conflict(
                verdict,
                role,
                ConflictKind.TARGET_ALREADY_DEAD,
                f"{verdict.target_id}は{target_death}日目に死亡しており、"
                f"{night}日目の夜に占うことはできません。",
            )
        )
    return conflicts


def _medium_conflicts(
    observations: ObservationSet, verdict: PublicVerdict, role: RoleName | None
) -> list[TimelineConflict]:
    """A medium reads the player executed that day, so the day names the target.

    `source_night` here is the day whose execution the result is about: the
    medium learns during the night that follows the vote.
    """
    day = verdict.source_night
    executed = observations.executions_on(day)
    if not executed:
        return [
            _conflict(
                verdict,
                role,
                ConflictKind.NO_EXECUTION,
                f"{day}日目に処刑は行われていませんが、{verdict.claimant_id}は"
                "その日の霊媒結果を主張しています。",
            )
        ]
    if verdict.target_id not in executed:
        return [
            _conflict(
                verdict,
                role,
                ConflictKind.WRONG_EXECUTION,
                f"{day}日目に処刑されたのは{'/'.join(executed)}であり、"
                f"{verdict.claimant_id}が霊媒結果を主張している{verdict.target_id}"
                "ではありません。",
            )
        ]
    return []


__all__ = [
    "ConflictKind",
    "TimelineConflict",
    "can_act_on_night",
    "can_be_targeted_on_night",
    "could_publish_on_day",
    "disclosable_result_nights",
    "conflicts_by_claimant",
    "find_timeline_conflicts",
]
