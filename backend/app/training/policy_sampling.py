"""Turn model logits into legal structured game decisions.

The sampler applies only mechanical legality and semantic well-formedness. It
never adds strategic doctrine: false claims, contradictory reports, unusual CO
timing, self-execution proposals, and other unconventional plays remain legal.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

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
from app.training.encoding import MAX_SEMANTIC_EVENTS
from app.training.legal import LegalActionMask
from app.training.observation import PolicyObservation
from app.training.parameters import SemanticParameterMask, semantic_parameter_mask
from app.training.policy_contract import PolicyLogits
from app.training.scheduler import SpeakIntent

EnumT = TypeVar("EnumT", bound=Enum)
_NIGHT_TOPICS = (Topic.DIVINE, Topic.GUARD, Topic.ATTACK)


@dataclass(frozen=True)
class HeadChoice:
    head: str
    index: int
    valid_indices: tuple[int, ...]
    log_prob: float


@dataclass(frozen=True)
class PolicySampleTrace:
    choices: tuple[HeadChoice, ...]
    value_estimate: float

    @property
    def log_prob(self) -> float:
        return sum(choice.log_prob for choice in self.choices)


@dataclass(frozen=True)
class SampledSpeech:
    intent: SpeakIntent
    trace: PolicySampleTrace


@dataclass(frozen=True)
class SampledVote:
    target_id: str
    trace: PolicySampleTrace


@dataclass(frozen=True)
class SampledNightAction:
    topic: Topic
    target_id: str
    trace: PolicySampleTrace


class MaskedPolicySampler:
    def __init__(
        self,
        seed: int | None = None,
        *,
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._rng = random.Random(seed)
        self._temperature = temperature

    def sample_speech(
        self,
        observation: PolicyObservation,
        logits: PolicyLogits,
    ) -> SampledSpeech:
        logits.validate()
        choices: list[HeadChoice] = []

        timing, choice = self._enum_choice(
            "timing",
            logits.timing,
            TimingBucket,
            tuple(TimingBucket),
        )
        choices.append(choice)
        if timing is TimingBucket.HOLD:
            return SampledSpeech(
                SpeakIntent(TimingBucket.HOLD, None),
                PolicySampleTrace(tuple(choices), logits.value),
            )

        action_types = _available_speech_actions(observation)
        action_type, choice = self._enum_choice(
            "action_type",
            logits.action_type,
            ActionType,
            action_types,
        )
        choices.append(choice)
        if action_type is ActionType.PASS:
            return SampledSpeech(
                SpeakIntent(TimingBucket.HOLD, None),
                PolicySampleTrace(tuple(choices), logits.value),
            )

        atom = self._sample_atom(observation, action_type, logits, choices)
        return SampledSpeech(
            SpeakIntent(timing, SpeechBundle((atom,))),
            PolicySampleTrace(tuple(choices), logits.value),
        )

    def sample_vote(
        self,
        observation: PolicyObservation,
        mask: LegalActionMask,
        logits: PolicyLogits,
    ) -> SampledVote:
        logits.validate()
        target_id, choice = self._seat_choice(
            "vote_target",
            logits.vote_target,
            observation,
            mask.vote_target_ids,
        )
        return SampledVote(
            target_id,
            PolicySampleTrace((choice,), logits.value),
        )

    def sample_night_action(
        self,
        observation: PolicyObservation,
        mask: LegalActionMask,
        logits: PolicyLogits,
    ) -> SampledNightAction:
        logits.validate()
        legal_topics = tuple(choice.topic for choice in mask.night_choices)
        topic, topic_choice = self._value_choice(
            "night_topic",
            logits.night_topic,
            _NIGHT_TOPICS,
            legal_topics,
        )
        legal_targets = next(
            choice.target_ids for choice in mask.night_choices if choice.topic is topic
        )
        target_id, target_choice = self._seat_choice(
            "night_target",
            logits.night_target,
            observation,
            legal_targets,
        )
        return SampledNightAction(
            topic,
            target_id,
            PolicySampleTrace((topic_choice, target_choice), logits.value),
        )

    def _sample_atom(
        self,
        observation: PolicyObservation,
        action_type: ActionType,
        logits: PolicyLogits,
        choices: list[HeadChoice],
    ) -> SemanticAction:
        if action_type is ActionType.REACT:
            mask = semantic_parameter_mask(observation, action_type)
            reference, choice = self._reference_choice(
                logits.reference_event,
                observation,
                mask.reference_event_ids,
            )
            choices.append(choice)
            stance, choice = self._enum_choice(
                "stance", logits.stance, Stance, mask.stances
            )
            choices.append(choice)
            return SemanticAction(
                action_type,
                stance=stance,
                reference_event_id=reference,
            )

        if action_type is ActionType.RETRACT:
            mask = semantic_parameter_mask(observation, action_type)
            reference, choice = self._reference_choice(
                logits.reference_event,
                observation,
                mask.reference_event_ids,
            )
            choices.append(choice)
            return SemanticAction(action_type, reference_event_id=reference)

        base = semantic_parameter_mask(observation, action_type)
        topic, choice = self._enum_choice("topic", logits.topic, Topic, base.topics)
        choices.append(choice)
        mask = semantic_parameter_mask(observation, action_type, topic=topic)

        if action_type is ActionType.CLAIM:
            return self._claim_atom(topic, mask, logits, choices, observation)
        if action_type is ActionType.REPORT:
            return self._report_atom(action_type, topic, mask, logits, choices, observation)
        if action_type is ActionType.EVALUATE:
            target_id, target_choice = self._seat_choice(
                "target", logits.target, observation, mask.target_ids
            )
            choices.append(target_choice)
            stance, stance_choice = self._enum_choice(
                "stance", logits.stance, Stance, mask.stances
            )
            choices.append(stance_choice)
            return SemanticAction(
                action_type,
                topic=topic,
                target_id=target_id,
                stance=stance,
            )
        if action_type is ActionType.DECLARE:
            target_id, target_choice = self._seat_choice(
                "target", logits.target, observation, mask.target_ids
            )
            choices.append(target_choice)
            return SemanticAction(action_type, topic=topic, target_id=target_id)
        if action_type is ActionType.PROPOSE:
            target_id, target_choice = self._seat_choice(
                "target", logits.target, observation, mask.target_ids
            )
            choices.append(target_choice)
            scope, scope_choice = self._enum_choice(
                "scope", logits.scope, Scope, mask.scopes
            )
            choices.append(scope_choice)
            return SemanticAction(
                action_type,
                topic=topic,
                target_id=target_id,
                scope=scope,
            )
        if action_type is ActionType.QUESTION:
            target_id, target_choice = self._seat_choice(
                "target", logits.target, observation, mask.target_ids
            )
            choices.append(target_choice)
            return SemanticAction(action_type, topic=topic, target_id=target_id)
        if action_type is ActionType.CORRECT:
            return self._correct_atom(topic, mask, logits, choices, observation)
        raise ValueError(f"unsupported sampled speech action {action_type}")

    def _claim_atom(
        self,
        topic: Topic,
        mask: SemanticParameterMask,
        logits: PolicyLogits,
        choices: list[HeadChoice],
        observation: PolicyObservation,
    ) -> SemanticAction:
        if topic is Topic.ROLE:
            role, choice = self._enum_choice("role", logits.role, RoleName, mask.roles)
            choices.append(choice)
            return SemanticAction(ActionType.CLAIM, topic=topic, role=role)
        if topic is Topic.PARTNER:
            target_id, choice = self._seat_choice(
                "target", logits.target, observation, mask.target_ids
            )
            choices.append(choice)
            return SemanticAction(ActionType.CLAIM, topic=topic, target_id=target_id)
        if topic is Topic.WOLF_COUNT:
            quantity, choice = self._integer_choice(
                "quantity", logits.quantity, mask.quantities
            )
            choices.append(choice)
            return SemanticAction(ActionType.CLAIM, topic=topic, quantity=quantity)
        raise ValueError(f"unsupported claim topic {topic}")

    def _report_atom(
        self,
        action_type: ActionType,
        topic: Topic,
        mask: SemanticParameterMask,
        logits: PolicyLogits,
        choices: list[HeadChoice],
        observation: PolicyObservation,
    ) -> SemanticAction:
        target_id, choice = self._seat_choice(
            "target", logits.target, observation, mask.target_ids
        )
        choices.append(choice)
        referenced_day, choice = self._integer_choice(
            "referenced_day", logits.referenced_day, mask.referenced_days
        )
        choices.append(choice)
        result: ResultValue | None = None
        if mask.results:
            result, choice = self._enum_choice(
                "result", logits.result, ResultValue, mask.results
            )
            choices.append(choice)
        return SemanticAction(
            action_type,
            topic=topic,
            target_id=target_id,
            result=result,
            referenced_day=referenced_day,
        )

    def _correct_atom(
        self,
        topic: Topic,
        mask: SemanticParameterMask,
        logits: PolicyLogits,
        choices: list[HeadChoice],
        observation: PolicyObservation,
    ) -> SemanticAction:
        target_id, choice = self._seat_choice(
            "target", logits.target, observation, mask.target_ids
        )
        choices.append(choice)
        result, choice = self._enum_choice(
            "result", logits.result, ResultValue, mask.results
        )
        choices.append(choice)
        referenced_day, choice = self._integer_choice(
            "referenced_day", logits.referenced_day, mask.referenced_days
        )
        choices.append(choice)
        reference, choice = self._reference_choice(
            logits.reference_event,
            observation,
            mask.reference_event_ids,
        )
        choices.append(choice)
        return SemanticAction(
            ActionType.CORRECT,
            topic=topic,
            target_id=target_id,
            result=result,
            referenced_day=referenced_day,
            reference_event_id=reference,
        )

    def _seat_choice(
        self,
        head: str,
        logits: tuple[float, ...],
        observation: PolicyObservation,
        valid_player_ids: tuple[str, ...],
    ) -> tuple[str, HeadChoice]:
        seats = tuple(player.player_id for player in observation.players)
        valid_indices = tuple(
            index for index, player_id in enumerate(seats) if player_id in valid_player_ids
        )
        index, choice = self._index_choice(head, logits, valid_indices)
        return seats[index], choice

    def _reference_choice(
        self,
        logits: tuple[float, ...],
        observation: PolicyObservation,
        valid_event_ids: tuple[str, ...],
    ) -> tuple[str, HeadChoice]:
        events = observation.semantic_events[-MAX_SEMANTIC_EVENTS:]
        valid = set(valid_event_ids)
        valid_indices = tuple(
            index for index, event in enumerate(events) if event.event_id in valid
        )
        index, choice = self._index_choice("reference_event", logits, valid_indices)
        return events[index].event_id, choice

    def _enum_choice(
        self,
        head: str,
        logits: tuple[float, ...],
        enum_type: type[EnumT],
        valid_values: tuple[EnumT, ...],
    ) -> tuple[EnumT, HeadChoice]:
        return self._value_choice(head, logits, tuple(enum_type), valid_values)

    def _value_choice(
        self,
        head: str,
        logits: tuple[float, ...],
        all_values: tuple[EnumT, ...],
        valid_values: tuple[EnumT, ...],
    ) -> tuple[EnumT, HeadChoice]:
        valid = set(valid_values)
        valid_indices = tuple(index for index, value in enumerate(all_values) if value in valid)
        index, choice = self._index_choice(head, logits, valid_indices)
        return all_values[index], choice

    def _integer_choice(
        self,
        head: str,
        logits: tuple[float, ...],
        valid_values: tuple[int, ...],
    ) -> tuple[int, HeadChoice]:
        index, choice = self._index_choice(head, logits, valid_values)
        return index, choice

    def _index_choice(
        self,
        head: str,
        logits: tuple[float, ...],
        valid_indices: tuple[int, ...],
    ) -> tuple[int, HeadChoice]:
        if not valid_indices:
            raise ValueError(f"{head} has no legal choices")
        if any(index < 0 or index >= len(logits) for index in valid_indices):
            raise ValueError(f"{head} legal index exceeds head width")
        scaled = tuple(logits[index] / self._temperature for index in valid_indices)
        peak = max(scaled)
        weights = tuple(math.exp(value - peak) for value in scaled)
        total = sum(weights)
        needle = self._rng.random() * total
        cumulative = 0.0
        selected_offset = len(weights) - 1
        for offset, weight in enumerate(weights):
            cumulative += weight
            if needle <= cumulative:
                selected_offset = offset
                break
        selected_index = valid_indices[selected_offset]
        probability = weights[selected_offset] / total
        return selected_index, HeadChoice(
            head=head,
            index=selected_index,
            valid_indices=valid_indices,
            log_prob=math.log(probability),
        )


def _available_speech_actions(observation: PolicyObservation) -> tuple[ActionType, ...]:
    candidates = (
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
    return tuple(
        action_type
        for action_type in candidates
        if action_type is ActionType.PASS
        or _speech_action_is_materializable(observation, action_type)
    )


def _speech_action_is_materializable(
    observation: PolicyObservation,
    action_type: ActionType,
) -> bool:
    base = semantic_parameter_mask(observation, action_type)
    if action_type is ActionType.REACT:
        return bool(base.reference_event_ids and base.stances)
    if action_type is ActionType.RETRACT:
        return bool(base.reference_event_ids)
    for topic in base.topics:
        mask = semantic_parameter_mask(observation, action_type, topic=topic)
        if action_type is ActionType.CLAIM:
            if topic is Topic.ROLE and mask.roles:
                return True
            if topic is Topic.PARTNER and mask.target_ids:
                return True
            if topic is Topic.WOLF_COUNT and mask.quantities:
                return True
        elif action_type is ActionType.REPORT:
            if mask.target_ids and mask.referenced_days:
                return True
        elif action_type is ActionType.EVALUATE:
            if mask.target_ids and mask.stances:
                return True
        elif action_type is ActionType.DECLARE:
            if mask.target_ids:
                return True
        elif action_type is ActionType.PROPOSE:
            if mask.target_ids and mask.scopes:
                return True
        elif action_type is ActionType.QUESTION:
            if mask.target_ids:
                return True
        elif action_type is ActionType.CORRECT:
            if (
                mask.target_ids
                and mask.results
                and mask.referenced_days
                and mask.reference_event_ids
            ):
                return True
    return False
