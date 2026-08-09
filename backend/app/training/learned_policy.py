"""Framework-independent runtime wrapper around a learned policy model."""

from __future__ import annotations

from dataclasses import dataclass

from app.training.encoding import EncodedPolicyObservation, ObservationEncoder
from app.training.legal import LegalActionMask
from app.training.observation import PolicyObservation
from app.training.policy_contract import LearnedPolicyModel
from app.training.policy_sampling import (
    MaskedPolicySampler,
    SampledNightAction,
    SampledSpeech,
    SampledVote,
)


@dataclass(frozen=True)
class SpeechPolicyStep:
    observation: EncodedPolicyObservation
    sampled: SampledSpeech


@dataclass(frozen=True)
class VotePolicyStep:
    observation: EncodedPolicyObservation
    sampled: SampledVote


@dataclass(frozen=True)
class NightPolicyStep:
    observation: EncodedPolicyObservation
    sampled: SampledNightAction


class LearnedStructuredPolicy:
    """Encode one seat's view, run the model, then sample only legal actions."""

    def __init__(
        self,
        model: LearnedPolicyModel,
        *,
        seed: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        self._model = model
        self._encoder = ObservationEncoder()
        self._sampler = MaskedPolicySampler(seed=seed, temperature=temperature)

    def speech_step(self, observation: PolicyObservation) -> SpeechPolicyStep:
        encoded = self._encoder.encode(observation)
        logits = self._model.forward(encoded)
        return SpeechPolicyStep(encoded, self._sampler.sample_speech(observation, logits))

    def vote_step(
        self,
        observation: PolicyObservation,
        mask: LegalActionMask,
    ) -> VotePolicyStep:
        encoded = self._encoder.encode(observation)
        logits = self._model.forward(encoded)
        return VotePolicyStep(encoded, self._sampler.sample_vote(observation, mask, logits))

    def night_step(
        self,
        observation: PolicyObservation,
        mask: LegalActionMask,
    ) -> NightPolicyStep:
        encoded = self._encoder.encode(observation)
        logits = self._model.forward(encoded)
        return NightPolicyStep(
            encoded,
            self._sampler.sample_night_action(observation, mask, logits),
        )
