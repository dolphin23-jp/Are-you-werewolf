"""Batched full-episode runner for one shared PyTorch Transformer policy.

All seats still own independent samplers/RNG streams. Only neural inference is
batched: every seat sees exactly the same information-safe observation it would
receive through :class:`LearnedEpisodeRunner`.
"""

from __future__ import annotations

import torch

from app.engine.game import PlayerSpec
from app.engine.phases import Phase
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.learned_policy import SpeechPolicyStep
from app.training.learned_runner import LearnedEpisodeResult
from app.training.legal import legal_action_mask
from app.training.policy_sampling import MaskedPolicySampler, PolicySampleTrace
from app.training.torch_policy import TorchTransformerPolicy
from app.training.trajectory import DecisionKind, EpisodeTrajectory, RecordedDecision


class TorchBatchedEpisodeRunner:
    """Drive all 17 seats while batching shared-model inference by game step."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        model: TorchTransformerPolicy,
        *,
        max_loops: int = 200,
        max_discussion_ticks: int = 12,
        temperature: float = 1.0,
    ) -> None:
        self._player_specs = player_specs
        self._model = model
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
        samplers = {
            player_id: MaskedPolicySampler(
                seed=seed * 1000 + index,
                temperature=self._temperature,
            )
            for index, player_id in enumerate(env.controller.state.players)
        }
        trajectory = EpisodeTrajectory(f"torch-batched-{seed}")
        self._model.eval()

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
                self._run_night(env, samplers, trajectory)
                env.controller.resolve_night()
            elif state.phase is Phase.DAWN:
                env.controller.start_discussion()
                env.scheduler.reset()
            elif state.phase is Phase.DISCUSSION:
                self._run_discussion(env, samplers, trajectory)
                env.controller.end_discussion()
            elif state.phase in (Phase.VOTING, Phase.RUNOFF):
                self._run_voting(env, samplers, trajectory)
                env.controller.resolve_votes()
            elif state.phase is Phase.VOTE_RESULT:
                env.controller.start_night()
            else:
                raise RuntimeError(f"unexpected training phase {state.phase}")

        raise RuntimeError(f"episode exceeded {self._max_loops} phase loops")

    def _run_discussion(
        self,
        env: WerewolfTrainingEnv,
        samplers: dict[str, MaskedPolicySampler],
        trajectory: EpisodeTrajectory,
    ) -> None:
        for _ in range(self._max_discussion_ticks):
            player_ids = tuple(env.controller.state.alive_ids())
            observations = tuple(env.observe(player_id) for player_id in player_ids)
            encoded = tuple(self._encoder.encode(observation) for observation in observations)
            with torch.no_grad():
                output = self._model.forward_batch(encoded)
            steps: dict[str, SpeechPolicyStep] = {}
            for index, player_id in enumerate(player_ids):
                logits = self._model.policy_logits_at(output, index)
                sampled = samplers[player_id].sample_speech(observations[index], logits)
                steps[player_id] = SpeechPolicyStep(encoded[index], sampled)

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
        samplers: dict[str, MaskedPolicySampler],
        trajectory: EpisodeTrajectory,
    ) -> None:
        candidates = []
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if not mask.vote_target_ids:
                continue
            observation = env.observe(player_id)
            candidates.append(
                (player_id, observation, self._encoder.encode(observation), mask)
            )
        if not candidates:
            return
        with torch.no_grad():
            output = self._model.forward_batch(tuple(item[2] for item in candidates))
        for index, (player_id, observation, encoded, mask) in enumerate(candidates):
            logits = self._model.policy_logits_at(output, index)
            sampled = samplers[player_id].sample_vote(observation, mask, logits)
            env.vote(player_id, sampled.target_id)
            trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.VOTE,
                    observation=encoded,
                    target_id=sampled.target_id,
                    policy_trace=sampled.trace,
                )
            )

    def _run_night(
        self,
        env: WerewolfTrainingEnv,
        samplers: dict[str, MaskedPolicySampler],
        trajectory: EpisodeTrajectory,
    ) -> None:
        candidates = []
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if not mask.night_choices:
                continue
            observation = env.observe(player_id)
            candidates.append(
                (player_id, observation, self._encoder.encode(observation), mask)
            )
        if not candidates:
            return
        with torch.no_grad():
            output = self._model.forward_batch(tuple(item[2] for item in candidates))
        for index, (player_id, observation, encoded, mask) in enumerate(candidates):
            logits = self._model.policy_logits_at(output, index)
            sampled = samplers[player_id].sample_night_action(observation, mask, logits)
            env.night_action(player_id, sampled.topic, sampled.target_id)
            trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.NIGHT,
                    observation=encoded,
                    target_id=sampled.target_id,
                    night_topic=sampled.topic,
                    policy_trace=sampled.trace,
                )
            )


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
