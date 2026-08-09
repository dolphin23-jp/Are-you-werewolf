"""Strategy-free random baseline for validating the self-play environment."""

from __future__ import annotations

import random

from app.training.actions import (
    ActionType,
    SemanticAction,
    SpeechBundle,
    TimingBucket,
    Topic,
)
from app.training.legal import LegalActionMask
from app.training.observation import PolicyObservation
from app.training.parameters import SemanticParameterMask, semantic_parameter_mask
from app.training.scheduler import SpeakIntent


class RandomPolicy:
    def __init__(self, seed: int | None = None, speak_probability: float = 0.25) -> None:
        self._rng = random.Random(seed)
        self._speak_probability = speak_probability

    def speak_intent(self, observation: PolicyObservation) -> SpeakIntent:
        if self._rng.random() >= self._speak_probability:
            return SpeakIntent(TimingBucket.HOLD, None)
        atom = self._random_speech_atom(observation)
        if atom.action_type is ActionType.PASS:
            return SpeakIntent(TimingBucket.HOLD, None)
        bundle = SpeechBundle((atom,))
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
        self._rng.shuffle(candidates)
        for action_type in candidates:
            atom = self._try_random_atom(observation, action_type)
            if atom is not None:
                return atom
        return SemanticAction(ActionType.PASS)

    def _try_random_atom(
        self,
        observation: PolicyObservation,
        action_type: ActionType,
    ) -> SemanticAction | None:
        base = semantic_parameter_mask(observation, action_type)
        if not base.topics:
            return None
        topics = list(base.topics)
        self._rng.shuffle(topics)
        for topic in topics:
            mask = semantic_parameter_mask(observation, action_type, topic=topic)
            atom = self._atom_from_mask(action_type, topic, mask)
            if atom is not None:
                return atom
        return None

    def _atom_from_mask(
        self,
        action_type: ActionType,
        topic: Topic,
        mask: SemanticParameterMask,
    ) -> SemanticAction | None:
        if action_type is ActionType.CLAIM:
            if topic is Topic.ROLE and mask.roles:
                return SemanticAction(action_type, topic=topic, role=self._rng.choice(mask.roles))
            if topic is Topic.PARTNER and mask.target_ids:
                return SemanticAction(
                    action_type,
                    topic=topic,
                    target_id=self._rng.choice(mask.target_ids),
                )
            if topic is Topic.WOLF_COUNT and mask.quantities:
                return SemanticAction(
                    action_type,
                    topic=topic,
                    quantity=self._rng.choice(mask.quantities),
                )
            return None

        if action_type is ActionType.REPORT:
            if not mask.target_ids or not mask.referenced_days:
                return None
            result = self._rng.choice(mask.results) if mask.results else None
            return SemanticAction(
                action_type,
                topic=topic,
                target_id=self._rng.choice(mask.target_ids),
                result=result,
                referenced_day=self._rng.choice(mask.referenced_days),
            )

        if action_type is ActionType.EVALUATE:
            if not mask.target_ids or not mask.stances:
                return None
            return SemanticAction(
                action_type,
                topic=topic,
                target_id=self._rng.choice(mask.target_ids),
                stance=self._rng.choice(mask.stances),
            )

        if action_type is ActionType.DECLARE:
            if not mask.target_ids:
                return None
            return SemanticAction(
                action_type,
                topic=topic,
                target_id=self._rng.choice(mask.target_ids),
            )

        if action_type is ActionType.PROPOSE:
            if not mask.target_ids or not mask.scopes:
                return None
            return SemanticAction(
                action_type,
                topic=topic,
                target_id=self._rng.choice(mask.target_ids),
                scope=self._rng.choice(mask.scopes),
            )

        return None
