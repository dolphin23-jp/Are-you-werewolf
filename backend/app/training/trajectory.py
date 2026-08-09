"""Framework-agnostic trajectory records for self-play training.

The trace stores exactly what a seat observed when it acted, plus the chosen
structured action. Terminal team rewards are attached only after the game ends.
No shaped strategic reward is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from app.training.actions import SpeechBundle, Topic
from app.training.encoding import EncodedPolicyObservation
from app.training.policy_sampling import PolicySampleTrace


class DecisionKind(StrEnum):
    SPEECH = "speech"
    VOTE = "vote"
    NIGHT = "night"


@dataclass(frozen=True)
class RecordedDecision:
    player_id: str
    kind: DecisionKind
    observation: EncodedPolicyObservation
    speech_bundle: SpeechBundle | None = None
    target_id: str | None = None
    night_topic: Topic | None = None
    policy_trace: PolicySampleTrace | None = None
    reward: float = 0.0

    def __post_init__(self) -> None:
        if self.kind is DecisionKind.SPEECH:
            if (
                self.speech_bundle is None
                or self.target_id is not None
                or self.night_topic is not None
            ):
                raise ValueError("speech decision requires only speech_bundle")
        elif self.kind is DecisionKind.VOTE:
            if (
                self.target_id is None
                or self.speech_bundle is not None
                or self.night_topic is not None
            ):
                raise ValueError("vote decision requires only target_id")
        elif self.kind is DecisionKind.NIGHT:
            if (
                self.target_id is None
                or self.night_topic is None
                or self.speech_bundle is not None
            ):
                raise ValueError("night decision requires topic and target_id")


@dataclass
class EpisodeTrajectory:
    episode_id: str
    decisions: list[RecordedDecision] = field(default_factory=list)
    terminal_rewards: dict[str, float] = field(default_factory=dict)
    finalized: bool = False

    def append(self, decision: RecordedDecision) -> None:
        if self.finalized:
            raise RuntimeError("cannot append to a finalized trajectory")
        self.decisions.append(decision)

    def finalize(self, rewards: dict[str, float]) -> None:
        if self.finalized:
            raise RuntimeError("trajectory already finalized")
        self.terminal_rewards = dict(rewards)
        self.decisions = [
            replace(decision, reward=rewards.get(decision.player_id, 0.0))
            for decision in self.decisions
        ]
        self.finalized = True

    def decisions_for(self, player_id: str) -> tuple[RecordedDecision, ...]:
        return tuple(
            decision for decision in self.decisions if decision.player_id == player_id
        )
