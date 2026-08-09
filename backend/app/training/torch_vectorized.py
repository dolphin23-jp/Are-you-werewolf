"""Cross-game vectorized rollout for Transformer self-play.

Several independent ``WerewolfTrainingEnv`` instances advance together. At each
logical decision point, observations from all currently eligible seats across
all games are grouped by the exact Transformer instance that owns the seat, then
run in inference microbatches. Games remain fully independent; only neural
inference is shared.
"""

from __future__ import annotations

from collections.abc import Mapping
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


@dataclass
class TorchRolloutInferenceStats:
    """Batch-shape metrics that never enter a policy observation."""

    logical_batches: int = 0
    inference_calls: int = 0
    inference_observations: int = 0
    max_pending_requests: int = 0
    max_inference_batch: int = 0

    def record_pending(self, count: int) -> None:
        if count <= 0:
            return
        self.logical_batches += 1
        self.max_pending_requests = max(self.max_pending_requests, count)

    def record_inference(self, count: int) -> None:
        if count <= 0:
            raise ValueError("inference batch must contain observations")
        self.inference_calls += 1
        self.inference_observations += count
        self.max_inference_batch = max(self.max_inference_batch, count)

    @property
    def mean_inference_batch(self) -> float:
        if self.inference_calls == 0:
            return 0.0
        return self.inference_observations / self.inference_calls

    @property
    def microbatch_expansion(self) -> float:
        if self.logical_batches == 0:
            return 0.0
        return self.inference_calls / self.logical_batches


@dataclass
class _EpisodeSlot:
    index: int
    seed: int
    env: WerewolfTrainingEnv
    samplers: dict[str, MaskedPolicySampler]
    player_models: dict[str, TorchTransformerPolicy]
    trajectory: EpisodeTrajectory
    discussion_ticks: int = 0
    result: LearnedEpisodeResult | None = None


@dataclass(frozen=True)
class _InferenceRequest:
    slot_index: int
    player_id: str
    observation: PolicyObservation
    encoded: EncodedPolicyObservation
    model: TorchTransformerPolicy


@dataclass(frozen=True)
class _PreparedRequest:
    request: _InferenceRequest
    logits: PolicyLogits


class TorchVectorizedEpisodeCollector:
    """Collect multiple independent episodes while batching shared model calls."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        model: TorchTransformerPolicy,
        *,
        max_global_steps: int = 2000,
        max_discussion_ticks: int = 12,
        max_inference_batch_size: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        if max_inference_batch_size is not None and max_inference_batch_size <= 0:
            raise ValueError("max_inference_batch_size must be positive")
        self._player_specs = player_specs
        self._model = model
        self._max_global_steps = max_global_steps
        self._max_discussion_ticks = max_discussion_ticks
        self.max_inference_batch_size = max_inference_batch_size
        self._temperature = temperature
        self._encoder = ObservationEncoder()
        self.inference_stats = TorchRolloutInferenceStats()

    def collect(
        self,
        seeds: tuple[int, ...],
        *,
        team_models: tuple[Mapping[Team, TorchTransformerPolicy], ...] | None = None,
    ) -> tuple[LearnedEpisodeResult, ...]:
        """Collect games, optionally assigning a model per faction per game.

        ``team_models`` is aligned one-to-one with ``seeds``. Missing factions
        fall back to the collector's default model. This keeps the shared-policy
        self-play API unchanged while allowing historical/population games to
        batch seats across independent games only when they truly share the same
        immutable model instance.
        """

        if not seeds:
            raise ValueError("vectorized rollout requires at least one seed")
        if len(set(seeds)) != len(seeds):
            raise ValueError("vectorized rollout seeds must be unique")
        if team_models is not None and len(team_models) != len(seeds):
            raise ValueError("team_models must align one-to-one with seeds")

        slots = [
            self._make_slot(
                index,
                seed,
                team_models[index] if team_models is not None else None,
            )
            for index, seed in enumerate(seeds)
        ]
        for model in _unique_models(
            model
            for slot in slots
            for model in slot.player_models.values()
        ):
            model.eval()

        for _ in range(self._max_global_steps):
            unfinished = [slot for slot in slots if slot.result is None]
            if not unfinished:
                return tuple(_require_result(slot) for slot in slots)

            self._finalize_completed(unfinished)
            unfinished = [slot for slot in slots if slot.result is None]
            if not unfinished:
                return tuple(_require_result(slot) for slot in slots)

            self._advance_dawn_and_vote_result(unfinished)
            self._close_expired_discussions(unfinished)
            self._run_night_batch(unfinished)
            self._run_discussion_batch(unfinished)
            self._run_voting_batch(unfinished)

        pending = [slot.seed for slot in slots if slot.result is None]
        raise RuntimeError(
            f"vectorized rollout exceeded {self._max_global_steps} steps; pending={pending}"
        )

    def _make_slot(
        self,
        index: int,
        seed: int,
        team_models: Mapping[Team, TorchTransformerPolicy] | None,
    ) -> _EpisodeSlot:
        env = WerewolfTrainingEnv(self._player_specs, seed=seed)
        samplers = {
            player_id: MaskedPolicySampler(
                seed=seed * 1000 + seat_index,
                temperature=self._temperature,
            )
            for seat_index, player_id in enumerate(env.controller.state.players)
        }
        supplied_models = team_models or {}
        player_models = {
            player_id: supplied_models.get(player.team, self._model)
            for player_id, player in env.controller.state.players.items()
        }
        return _EpisodeSlot(
            index=index,
            seed=seed,
            env=env,
            samplers=samplers,
            player_models=player_models,
            trajectory=EpisodeTrajectory(f"torch-vectorized-{seed}"),
        )

    @staticmethod
    def _finalize_completed(slots: list[_EpisodeSlot]) -> None:
        for slot in slots:
            state = slot.env.controller.state
            if state.phase is not Phase.GAME_OVER:
                continue
            rewards = slot.env.rewards()
            slot.trajectory.finalize(rewards)
            teams = {
                player_id: player.team
                for player_id, player in state.players.items()
            }
            slot.result = LearnedEpisodeResult(
                winner=state.winner,
                is_draw=state.is_draw,
                days=state.day,
                semantic_event_count=len(slot.env.semantic_events),
                rewards=rewards,
                teams=teams,
                trajectory=slot.trajectory,
            )

    @staticmethod
    def _advance_dawn_and_vote_result(slots: list[_EpisodeSlot]) -> None:
        for slot in slots:
            state = slot.env.controller.state
            if state.phase is Phase.DAWN:
                slot.env.controller.start_discussion()
                slot.env.scheduler.reset()
                slot.discussion_ticks = 0
            elif state.phase is Phase.VOTE_RESULT:
                slot.env.controller.start_night()

    def _close_expired_discussions(self, slots: list[_EpisodeSlot]) -> None:
        for slot in slots:
            if (
                slot.env.controller.state.phase is Phase.DISCUSSION
                and slot.discussion_ticks >= self._max_discussion_ticks
            ):
                slot.env.controller.end_discussion()

    def _run_night_batch(self, slots: list[_EpisodeSlot]) -> None:
        requests: list[_InferenceRequest] = []
        masks: dict[tuple[int, str], LegalActionMask] = {}
        night_slots = [
            slot for slot in slots if slot.env.controller.state.phase is Phase.NIGHT
        ]
        for slot in night_slots:
            for player_id in slot.env.controller.state.alive_ids():
                mask = legal_action_mask(slot.env.controller, player_id)
                if not mask.night_choices:
                    continue
                masks[(slot.index, player_id)] = mask
                requests.append(self._request(slot, player_id))

        prepared = self._infer(requests)
        for item in prepared:
            slot = _slot_by_index(slots, item.request.slot_index)
            player_id = item.request.player_id
            sampled = slot.samplers[player_id].sample_night_action(
                item.request.observation,
                masks[(slot.index, player_id)],
                item.logits,
            )
            slot.env.night_action(player_id, sampled.topic, sampled.target_id)
            slot.trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.NIGHT,
                    observation=item.request.encoded,
                    target_id=sampled.target_id,
                    night_topic=sampled.topic,
                    policy_trace=sampled.trace,
                )
            )

        for slot in night_slots:
            slot.env.controller.resolve_night()

    def _run_discussion_batch(self, slots: list[_EpisodeSlot]) -> None:
        discussion_slots = [
            slot
            for slot in slots
            if slot.env.controller.state.phase is Phase.DISCUSSION
            and slot.discussion_ticks < self._max_discussion_ticks
        ]
        requests = [
            self._request(slot, player_id)
            for slot in discussion_slots
            for player_id in slot.env.controller.state.alive_ids()
        ]
        prepared = self._infer(requests)
        by_slot: dict[int, dict[str, SpeechPolicyStep]] = {
            slot.index: {} for slot in discussion_slots
        }
        for item in prepared:
            slot = _slot_by_index(slots, item.request.slot_index)
            player_id = item.request.player_id
            sampled = slot.samplers[player_id].sample_speech(
                item.request.observation,
                item.logits,
            )
            by_slot[slot.index][player_id] = SpeechPolicyStep(
                item.request.encoded,
                sampled,
            )

        for slot in discussion_slots:
            steps = by_slot[slot.index]
            intents = {
                player_id: step.sampled.intent for player_id, step in steps.items()
            }
            selected = slot.env.select_next_speaker(intents)
            selected_player_id = selected.player_id if selected else None
            _record_discussion_cycle(steps, selected_player_id, slot.trajectory)
            slot.discussion_ticks += 1
            if selected is None:
                slot.env.controller.end_discussion()
                continue
            bundle = steps[selected.player_id].sampled.intent.bundle
            if bundle is None:
                raise RuntimeError("scheduler selected a non-speaking learned intent")
            slot.env.emit_speech(selected.player_id, bundle)
            if slot.discussion_ticks >= self._max_discussion_ticks:
                slot.env.controller.end_discussion()

    def _run_voting_batch(self, slots: list[_EpisodeSlot]) -> None:
        requests: list[_InferenceRequest] = []
        masks: dict[tuple[int, str], LegalActionMask] = {}
        voting_slots = [
            slot
            for slot in slots
            if slot.env.controller.state.phase in (Phase.VOTING, Phase.RUNOFF)
        ]
        for slot in voting_slots:
            for player_id in slot.env.controller.state.alive_ids():
                mask = legal_action_mask(slot.env.controller, player_id)
                if not mask.vote_target_ids:
                    continue
                masks[(slot.index, player_id)] = mask
                requests.append(self._request(slot, player_id))

        prepared = self._infer(requests)
        for item in prepared:
            slot = _slot_by_index(slots, item.request.slot_index)
            player_id = item.request.player_id
            sampled = slot.samplers[player_id].sample_vote(
                item.request.observation,
                masks[(slot.index, player_id)],
                item.logits,
            )
            slot.env.vote(player_id, sampled.target_id)
            slot.trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.VOTE,
                    observation=item.request.encoded,
                    target_id=sampled.target_id,
                    policy_trace=sampled.trace,
                )
            )

        for slot in voting_slots:
            slot.env.controller.resolve_votes()

    def _request(self, slot: _EpisodeSlot, player_id: str) -> _InferenceRequest:
        observation = slot.env.observe(player_id)
        return _InferenceRequest(
            slot_index=slot.index,
            player_id=player_id,
            observation=observation,
            encoded=self._encoder.encode(observation),
            model=slot.player_models[player_id],
        )

    def _infer(
        self,
        requests: list[_InferenceRequest],
    ) -> tuple[_PreparedRequest, ...]:
        if not requests:
            return ()
        self.inference_stats.record_pending(len(requests))

        grouped: dict[
            int,
            tuple[TorchTransformerPolicy, list[tuple[int, _InferenceRequest]]],
        ] = {}
        for request_index, request in enumerate(requests):
            key = id(request.model)
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = (request.model, [(request_index, request)])
            else:
                existing[1].append((request_index, request))

        prepared: list[_PreparedRequest | None] = [None] * len(requests)
        for model, indexed_requests in grouped.values():
            limit = self.max_inference_batch_size or len(indexed_requests)
            for start in range(0, len(indexed_requests), limit):
                batch = indexed_requests[start : start + limit]
                self.inference_stats.record_inference(len(batch))
                with torch.no_grad():
                    output = model.forward_batch(
                        tuple(request.encoded for _, request in batch)
                    )
                logits_batch = model.policy_logits_batch(output)
                for (request_index, request), logits in zip(
                    batch,
                    logits_batch,
                    strict=True,
                ):
                    prepared[request_index] = _PreparedRequest(
                        request=request,
                        logits=logits,
                    )

        return tuple(_require_prepared(item) for item in prepared)


def _unique_models(
    models: object,
) -> tuple[TorchTransformerPolicy, ...]:
    unique: dict[int, TorchTransformerPolicy] = {}
    for model in models:  # type: ignore[union-attr]
        if not isinstance(model, TorchTransformerPolicy):
            raise TypeError("rollout model collection contains a non-Transformer")
        unique[id(model)] = model
    return tuple(unique.values())


def _slot_by_index(slots: list[_EpisodeSlot], index: int) -> _EpisodeSlot:
    for slot in slots:
        if slot.index == index:
            return slot
    raise RuntimeError(f"episode slot {index} is not active in this batch")


def _require_result(slot: _EpisodeSlot) -> LearnedEpisodeResult:
    if slot.result is None:
        raise RuntimeError(f"episode {slot.seed} is not finalized")
    return slot.result


def _require_prepared(item: _PreparedRequest | None) -> _PreparedRequest:
    if item is None:
        raise RuntimeError("mixed-model inference did not prepare every request")
    return item


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
