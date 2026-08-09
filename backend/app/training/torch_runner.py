"""Batched full-episode runner for shared or faction-specific Transformers.

Every seat still receives its own information-safe observation and owns an
independent sampler/RNG stream. Neural inference is grouped only between seats
that use the exact same Transformer instance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from app.engine.game import PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import Team
from app.training.encoding import EncodedPolicyObservation, ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.learned_policy import SpeechPolicyStep
from app.training.learned_runner import LearnedEpisodeResult
from app.training.legal import LegalActionMask, legal_action_mask
from app.training.observation import PolicyObservation
from app.training.policy_contract import PolicyLogits
from app.training.policy_sampling import MaskedPolicySampler, PolicySampleTrace
from app.training.torch_policy import TorchTransformerPolicy
from app.training.trajectory import DecisionKind, EpisodeTrajectory, RecordedDecision


@dataclass(frozen=True)
class _PreparedSeat:
    player_id: str
    observation: PolicyObservation
    encoded: EncodedPolicyObservation
    logits: PolicyLogits


class TorchBatchedEpisodeRunner:
    """Drive all 17 seats while batching inference by shared model instance."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        model: TorchTransformerPolicy,
        *,
        team_models: Mapping[Team, TorchTransformerPolicy] | None = None,
        max_loops: int = 200,
        max_discussion_ticks: int = 12,
        temperature: float = 1.0,
    ) -> None:
        self._player_specs = player_specs
        self._model = model
        self._team_models = dict(team_models or {})
        self._max_loops = max_loops
        self._max_discussion_ticks = max_discussion_ticks
        self._temperature = temperature
        self._encoder = ObservationEncoder()

    def run(self, seed: int) -> LearnedEpisodeResult:
        env = WerewolfTrainingEnv(self._player_specs, seed=seed)
        teams = {
            player_id: player.team
            for player_id, player in env.controller.state.players.items()
        }
        player_models = {
            player_id: self._team_models.get(player.team, self._model)
            for player_id, player in env.controller.state.players.items()
        }
        for model in _unique_models(player_models.values()):
            model.eval()
        samplers = {
            player_id: MaskedPolicySampler(
                seed=seed * 1000 + index,
                temperature=self._temperature,
            )
            for index, player_id in enumerate(env.controller.state.players)
        }
        trajectory = EpisodeTrajectory(f"torch-batched-{seed}")

        for _ in range(self._max_loops):
            state = env.controller.state
            if state.phase is Phase.GAME_OVER:
                rewards = env.rewards()
                trajectory.finalize(rewards)
                return LearnedEpisodeResult(
                    winner=state.winner,
                    is_draw=state.is_draw,
                    days=state.day,
                    semantic_event_count=len(env.semantic_events),
                    rewards=rewards,
                    teams=teams,
                    trajectory=trajectory,
                )

            if state.phase is Phase.NIGHT:
                self._run_night(env, player_models, samplers, trajectory)
                env.controller.resolve_night()
            elif state.phase is Phase.DAWN:
                env.controller.start_discussion()
                env.scheduler.reset()
            elif state.phase is Phase.DISCUSSION:
                self._run_discussion(env, player_models, samplers, trajectory)
                env.controller.end_discussion()
            elif state.phase in (Phase.VOTING, Phase.RUNOFF):
                self._run_voting(env, player_models, samplers, trajectory)
                env.controller.resolve_votes()
            elif state.phase is Phase.VOTE_RESULT:
                env.controller.start_night()
            else:
                raise RuntimeError(f"unexpected training phase {state.phase}")

        raise RuntimeError(f"episode exceeded {self._max_loops} phase loops")

    def _run_discussion(
        self,
        env: WerewolfTrainingEnv,
        player_models: dict[str, TorchTransformerPolicy],
        samplers: dict[str, MaskedPolicySampler],
        trajectory: EpisodeTrajectory,
    ) -> None:
        for _ in range(self._max_discussion_ticks):
            player_ids = tuple(env.controller.state.alive_ids())
            prepared = self._prepare(env, player_ids, player_models)
            steps: dict[str, SpeechPolicyStep] = {}
            for player_id in player_ids:
                seat = prepared[player_id]
                sampled = samplers[player_id].sample_speech(
                    seat.observation,
                    seat.logits,
                )
                steps[player_id] = SpeechPolicyStep(seat.encoded, sampled)

            intents = {
                player_id: step.sampled.intent for player_id, step in steps.items()
            }
            selected = env.select_next_speaker(intents)
            selected_player_id = selected.player_id if selected else None
            _record_discussion_cycle(steps, selected_player_id, trajectory)
            if selected is None:
                return
            bundle = steps[selected.player_id].sampled.intent.bundle
            if bundle is None:
                raise RuntimeError("scheduler selected a non-speaking learned intent")
            env.emit_speech(selected.player_id, bundle)

    def _run_voting(
        self,
        env: WerewolfTrainingEnv,
        player_models: dict[str, TorchTransformerPolicy],
        samplers: dict[str, MaskedPolicySampler],
        trajectory: EpisodeTrajectory,
    ) -> None:
        masks: dict[str, LegalActionMask] = {}
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if mask.vote_target_ids:
                masks[player_id] = mask
        if not masks:
            return
        player_ids = tuple(masks)
        prepared = self._prepare(env, player_ids, player_models)
        for player_id in player_ids:
            seat = prepared[player_id]
            sampled = samplers[player_id].sample_vote(
                seat.observation,
                masks[player_id],
                seat.logits,
            )
            env.vote(player_id, sampled.target_id)
            trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.VOTE,
                    observation=seat.encoded,
                    target_id=sampled.target_id,
                    policy_trace=sampled.trace,
                )
            )

    def _run_night(
        self,
        env: WerewolfTrainingEnv,
        player_models: dict[str, TorchTransformerPolicy],
        samplers: dict[str, MaskedPolicySampler],
        trajectory: EpisodeTrajectory,
    ) -> None:
        masks: dict[str, LegalActionMask] = {}
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if mask.night_choices:
                masks[player_id] = mask
        if not masks:
            return
        player_ids = tuple(masks)
        prepared = self._prepare(env, player_ids, player_models)
        for player_id in player_ids:
            seat = prepared[player_id]
            sampled = samplers[player_id].sample_night_action(
                seat.observation,
                masks[player_id],
                seat.logits,
            )
            env.night_action(player_id, sampled.topic, sampled.target_id)
            trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.NIGHT,
                    observation=seat.encoded,
                    target_id=sampled.target_id,
                    night_topic=sampled.topic,
                    policy_trace=sampled.trace,
                )
            )

    def _prepare(
        self,
        env: WerewolfTrainingEnv,
        player_ids: tuple[str, ...],
        player_models: dict[str, TorchTransformerPolicy],
    ) -> dict[str, _PreparedSeat]:
        source: dict[str, tuple[PolicyObservation, EncodedPolicyObservation]] = {}
        grouped: dict[int, tuple[TorchTransformerPolicy, list[str]]] = {}
        for player_id in player_ids:
            observation = env.observe(player_id)
            encoded = self._encoder.encode(observation)
            source[player_id] = (observation, encoded)
            model = player_models[player_id]
            key = id(model)
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = (model, [player_id])
            else:
                existing[1].append(player_id)

        prepared: dict[str, _PreparedSeat] = {}
        with torch.no_grad():
            for model, group_ids in grouped.values():
                output = model.forward_batch(
                    tuple(source[player_id][1] for player_id in group_ids)
                )
                logits_batch = model.policy_logits_batch(output)
                for player_id, logits in zip(group_ids, logits_batch, strict=True):
                    observation, encoded = source[player_id]
                    prepared[player_id] = _PreparedSeat(
                        player_id=player_id,
                        observation=observation,
                        encoded=encoded,
                        logits=logits,
                    )
        return prepared


def _unique_models(
    models: Iterable[TorchTransformerPolicy],
) -> tuple[TorchTransformerPolicy, ...]:
    unique: dict[int, TorchTransformerPolicy] = {}
    for model in models:
        unique[id(model)] = model
    return tuple(unique.values())


def _record_discussion_cycle(
    steps: dict[str, SpeechPolicyStep],
    selected_player_id: str | None,
    trajectory: EpisodeTrajectory,
) -> None:
    for player_id, step in steps.items():
        sampled = step.sampled
        bundle = sampled.intent.bundle
        if player_id == selected_player_id:
            if bundle is None:
                raise RuntimeError("selected speech step has no bundle")
            trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.SPEECH,
                    observation=step.observation,
                    speech_bundle=bundle,
                    policy_trace=sampled.trace,
                )
            )
            continue

        trace = sampled.trace
        if bundle is not None:
            if not trace.choices or trace.choices[0].head != "timing":
                raise RuntimeError("speech trace must begin with timing head")
            trace = PolicySampleTrace((trace.choices[0],), trace.value_estimate)
        trajectory.append(
            RecordedDecision(
                player_id=player_id,
                kind=DecisionKind.TIMING,
                observation=step.observation,
                policy_trace=trace,
            )
        )
