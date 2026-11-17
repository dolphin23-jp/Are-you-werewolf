"""Night action resolution, in explicit, documented order.

Day-0 night (the scripted "first victim" night before Day 1 discussion):
  1. The Seer may optionally divine (result revealed at dawn).
  2. If the divined target is the Fox, it is curse-killed, exactly as on a
     regular night: same cause (`CURSED`), same night numbering (dated 0, found
     at dawn of Day 1), applied before the night's other death.
  3. The scripted first victim dies unconditionally (never Wolf/Fox). It stands
     where a regular night's attack stands, so it is applied after the curse.
No guard/attack/medium activity happens on Day 0.

The night-0 curse is a rule, not a detail: the table reasons from it ("the
seer's night-0 target did not die, so if that seer is real it is not the fox"),
the solver assumes it on every night, and an engine that skipped it on night 0
made a real seer who happened to divine the fox look fake to their own logic.
With no attack on night 0, a night-0 death other than the first victim can only
be the curse, which is why the public death cause still reads as an ordinary
night death and the deduction is left to the table.

Regular night (Day N >= 1):
  0. The wolves must have chosen a non-wolf to attack: while any wolf lives,
     every regular night has exactly one attack (GameController refuses both
     a wolf target and resolving without an attack). So a morning without a
     corpse means the attack hit the guard target or the fox, and a night
     corpse is never a wolf -- facts every seat reasons from.
  1. Medium result: if a player was executed at the end of Day N, every
     living Medium learns whether that player was a werewolf.
  2. Seer divination resolves; if the target is the Fox, a curse-kill is
     scheduled (the Fox dies from being divined).
  3. Hunter's guard target is recorded.
  4. The alpha Werewolf's attack target is recorded.
  5. The curse-kill (step 2) is applied first: Fox dies if divined.
  6. The attack (step 4) is applied: no-op if the target is the Fox
     (immune to attack) or the target matches the guard target; otherwise
     the target dies.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.roles import RoleName
from app.engine.state import (
    AttackRecord,
    DeathCause,
    DeathRecord,
    DivineRecord,
    GameState,
    GuardRecord,
    MediumRecord,
)


@dataclass
class NightResult:
    deaths: list[DeathRecord]
    divine_result: DivineRecord | None = None
    medium_results: list[MediumRecord] | None = None


class NightResolver:
    def resolve_day_zero(self, state: GameState, first_victim_id: str) -> NightResult:
        deaths: list[DeathRecord] = []
        divine_result = self._resolve_divine(state)
        # The same curse as every other night. The first victim cannot be
        # divined (the controller refuses it), so the order below changes no
        # outcome -- it only keeps the death records in regular-night order.
        curse_death = self._apply_curse(state, divine_result)
        if curse_death is not None:
            deaths.append(curse_death)

        victim = state.players[first_victim_id]
        victim.alive = False
        victim.death_cause = DeathCause.FIRST_VICTIM
        victim.death_day = state.day
        record = DeathRecord(
            player_id=first_victim_id, cause=DeathCause.FIRST_VICTIM, day=state.day
        )
        state.death_records.append(record)
        deaths.append(record)

        state.pending_divine = None
        state.pending_guard = None
        state.pending_attack = None
        return NightResult(deaths=deaths, divine_result=divine_result)

    def resolve_night(self, state: GameState) -> NightResult:
        _require_a_legal_attack(state)
        medium_results = self._distribute_medium_results(state)
        divine_result = self._resolve_divine(state)
        guard_target = self._resolve_guard(state)
        deaths: list[DeathRecord] = []

        if divine_result is not None and divine_result.is_werewolf is False:
            pass  # informational only; curse handled below via fox check

        curse_death = self._apply_curse(state, divine_result)
        if curse_death is not None:
            deaths.append(curse_death)

        attack_death = self._apply_attack(state, guard_target)
        if attack_death is not None:
            deaths.append(attack_death)
        _in_seat_order(state, deaths)

        state.pending_divine = None
        state.pending_guard = None
        state.pending_attack = None
        return NightResult(
            deaths=deaths, divine_result=divine_result, medium_results=medium_results
        )

    def _distribute_medium_results(self, state: GameState) -> list[MediumRecord]:
        executed_today = [
            d for d in state.death_records if d.cause == DeathCause.EXECUTED and d.day == state.day
        ]
        if not executed_today:
            return []
        executed = executed_today[-1]
        target = state.players[executed.player_id]
        is_wolf = target.role == RoleName.WEREWOLF
        results: list[MediumRecord] = []
        for medium in state.players_by_role(RoleName.MEDIUM):
            if not medium.alive:
                continue
            record = MediumRecord(
                medium_id=medium.player_id,
                target_id=executed.player_id,
                day=state.day,
                is_werewolf=is_wolf,
            )
            state.medium_records.append(record)
            results.append(record)
        return results

    def _resolve_divine(self, state: GameState) -> DivineRecord | None:
        if state.pending_divine is None:
            return None
        seer_id, target_id = state.pending_divine
        target = state.players[target_id]
        is_wolf = target.role == RoleName.WEREWOLF
        record = DivineRecord(
            seer_id=seer_id, target_id=target_id, day=state.day, is_werewolf=is_wolf
        )
        state.divine_records.append(record)
        return record

    def _resolve_guard(self, state: GameState) -> str | None:
        if state.pending_guard is None:
            return None
        hunter_id, target_id = state.pending_guard
        state.guard_records.append(
            GuardRecord(hunter_id=hunter_id, target_id=target_id, day=state.day)
        )
        return target_id

    def _apply_curse(
        self, state: GameState, divine_result: DivineRecord | None
    ) -> DeathRecord | None:
        if divine_result is None:
            return None
        target = state.players[divine_result.target_id]
        if target.role != RoleName.FOX or not target.alive:
            return None
        target.alive = False
        target.death_cause = DeathCause.CURSED
        target.death_day = state.day
        record = DeathRecord(player_id=target.player_id, cause=DeathCause.CURSED, day=state.day)
        state.death_records.append(record)
        return record

    def _apply_attack(self, state: GameState, guard_target: str | None) -> DeathRecord | None:
        if state.pending_attack is None:
            return None
        wolf_id, target_id = state.pending_attack
        target = state.players[target_id]

        succeeded = target.alive and target.role != RoleName.FOX and target_id != guard_target
        state.attack_records.append(
            AttackRecord(wolf_id=wolf_id, target_id=target_id, day=state.day, succeeded=succeeded)
        )
        if not succeeded:
            return None
        target.alive = False
        target.death_cause = DeathCause.ATTACKED
        target.death_day = state.day
        record = DeathRecord(player_id=target_id, cause=DeathCause.ATTACKED, day=state.day)
        state.death_records.append(record)
        return record


def _require_a_legal_attack(state: GameState) -> None:
    """Step 0, checked before anything on the night resolves.

    `GameController` refuses both cases with a GameError first; this is the
    resolver's own guard, so a caller that writes `pending_attack` by hand
    cannot resolve half a night and then fail.
    """
    if state.pending_attack is None:
        if any(p.alive and p.role == RoleName.WEREWOLF for p in state.players.values()):
            raise ValueError("the werewolves must attack every night from night 1")
        return
    _wolf_id, target_id = state.pending_attack
    if state.players[target_id].role == RoleName.WEREWOLF:
        raise ValueError("the werewolves cannot attack a werewolf")


def _in_seat_order(state: GameState, deaths: list[DeathRecord]) -> None:
    """Put a night's corpses in seat order, in the result and in the record.

    The curse is applied before the attack, so in resolution order the first of
    two corpses was always the fox. Every reader that lists the morning's dead
    from these lists -- the PLAYER_DIED events every client receives, the
    prompts, the replay -- would hand that away. Seat order carries nothing.
    The night's records are the last ones appended, so they are reordered in
    place there too.
    """
    if len(deaths) < 2:
        return
    seat = {player_id: index for index, player_id in enumerate(state.players)}
    tail = state.death_records[-len(deaths) :]
    if [id(record) for record in tail] != [id(record) for record in deaths]:
        raise RuntimeError("night deaths are not the last death records")
    deaths.sort(key=lambda record: seat[record.player_id])
    state.death_records[-len(deaths) :] = deaths

