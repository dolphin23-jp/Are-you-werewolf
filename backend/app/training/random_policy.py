"""Strategy-free random baseline for validating the self-play environment."""

from __future__ import annotations

import random

from app.engine.roles import RoleName
from app.training.actions import (
    ActionType,
    ResultValue,
    Scope,
    SemanticAction,
    SpeechBundle,
    Stance,
    TimingBucket,
    Topic,
)
from app.training.legal import LegalActionMask
from app.training.observation import PolicyObservation
from app.training.scheduler import SpeakIntent


class RandomPolicy:
    def __init__(self, seed: int | None = None, speak_probability: float = 0.25) -> None:
        self._rng = random.Random(seed)
        self._speak_probability = speak_probability

    def speak_intent(self, observation: PolicyObservation) -> SpeakIntent:
        if self._rng.random() >= self._speak_probability:
            return SpeakIntent(TimingBucket.HOLD, None)
        bundle = SpeechBundle((self._random_speech_atom(observation),))
        timing = self._rng.choice(
            [
                TimingBucket.IMMEDIATE,
                TimingBucket.EARLY,
                TimingBucket.NORMAL,
                TimingBucket.LATE,
            ]
        )
        return SpeakIntent(timing, bundle)

    def vote_target(self, mask: LegalActionMask) -> str:
        if not mask.vote_target_ids:
            raise ValueError("no legal vote target")
        return self._rng.choice(mask.vote_target_ids)

    def night_choice(self, mask: LegalActionMask) -> tuple[Topic, str]:
        if not mask.night_choices:
            raise ValueError("no legal night action")
        choice = self._rng.choice(mask.night_choices)
        return choice.topic, self._rng.choice(choice.target_ids)

    def _random_speech_atom(self, observation: PolicyObservation) -> SemanticAction:
        candidates = [
            ActionType.CLAIM,
            ActionType.REPORT,
            ActionType.EVALUATE,
            ActionType.DECLARE,
            ActionType.PROPOSE,
        ]
        action_type = self._rng.choice(candidates)
        player_ids = [player.player_id for player in observation.players]
        alive_ids = [player.player_id for player in observation.players if player.alive]
        other_alive = [player_id for player_id in alive_ids if player_id != observation.viewer_id]

        if action_type is ActionType.CLAIM:
            return SemanticAction(
                action_type,
                topic=Topic.ROLE,
                role=self._rng.choice(list(RoleName)),
            )

        if action_type is ActionType.REPORT:
            return SemanticAction(
                action_type,
                topic=self._rng.choice([Topic.SEER_RESULT, Topic.MEDIUM_RESULT]),
                target_id=self._rng.choice(player_ids),
                result=self._rng.choice([ResultValue.WHITE, ResultValue.BLACK]),
                referenced_day=max(0, observation.day - 1),
            )

        if action_type is ActionType.PROPOSE:
            target_id = self._rng.choice(alive_ids or player_ids)
            return SemanticAction(
                action_type,
                topic=Topic.EXECUTION,
                target_id=target_id,
                scope=Scope.SELF if target_id == observation.viewer_id else Scope.NONE,
            )

        target_id = self._rng.choice(other_alive or alive_ids or player_ids)
        if action_type is ActionType.EVALUATE:
            return SemanticAction(
                action_type,
                topic=Topic.WOLF,
                target_id=target_id,
                stance=self._rng.choice([Stance.SUSPECT, Stance.TRUST]),
            )
        return SemanticAction(action_type, topic=Topic.VOTE, target_id=target_id)
