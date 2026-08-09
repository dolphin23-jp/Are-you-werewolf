"""Mechanical legal-action masks for the training environment.

This module encodes game rules only. It deliberately does not encode advice
such as when to CO, who is suspicious, or which role should be protected.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.game import GameController
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.training.actions import ActionType, Topic

_PUBLIC_SPEECH_ACTIONS = (
    ActionType.PASS,
    ActionType.CLAIM,
    ActionType.REPORT,
    ActionType.EVALUATE,
    ActionType.DECLARE,
    ActionType.PROPOSE,
    ActionType.QUESTION,
    ActionType.REACT,
    ActionType.RETRACT,
    ActionType.CORRECT,
)


@dataclass(frozen=True)
class NightActionChoice:
    topic: Topic
    target_ids: tuple[str, ...]


@dataclass(frozen=True)
class LegalActionMask:
    action_types: tuple[ActionType, ...]
    vote_target_ids: tuple[str, ...] = ()
    night_choices: tuple[NightActionChoice, ...] = ()


def legal_action_mask(controller: GameController, player_id: str) -> LegalActionMask:
    state = controller.state
    player = state.players[player_id]
    if not player.alive or state.phase is Phase.GAME_OVER:
        return LegalActionMask(())

    if state.phase is Phase.DISCUSSION:
        return LegalActionMask(_PUBLIC_SPEECH_ACTIONS)

    if state.phase in (Phase.VOTING, Phase.RUNOFF):
        targets = tuple(state.votable_ids(player_id))
        return LegalActionMask(
            (ActionType.VOTE,) if targets else (),
            vote_target_ids=targets,
        )

    if state.phase is not Phase.NIGHT:
        return LegalActionMask(())

    alive_ids = state.alive_ids()
    choices: list[NightActionChoice] = []

    if player.role is RoleName.SEER:
        targets = tuple(
            target_id
            for target_id in alive_ids
            if target_id != player_id and target_id != state.first_victim_id
        )
        if targets:
            choices.append(NightActionChoice(Topic.DIVINE, targets))

    if state.day > 0 and player.role is RoleName.HUNTER:
        # Consecutive guard is legal. Do not filter the previous guard target.
        targets = tuple(target_id for target_id in alive_ids if target_id != player_id)
        if targets:
            choices.append(NightActionChoice(Topic.GUARD, targets))

    if (
        state.day > 0
        and player.role is RoleName.WEREWOLF
        and player_id == controller.alpha_wolf_id
    ):
        targets = tuple(
            target_id
            for target_id in alive_ids
            if state.players[target_id].role is not RoleName.WEREWOLF
        )
        if targets:
            choices.append(NightActionChoice(Topic.ATTACK, targets))

    return LegalActionMask(
        (ActionType.NIGHT_ACTION,) if choices else (),
        night_choices=tuple(choices),
    )
