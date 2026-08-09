"""Cross-game vectorized rollout for shared-policy Transformer self-play.

Several independent ``WerewolfTrainingEnv`` instances advance together. At each
logical decision point, observations from all currently eligible seats across
all games are concatenated into one Transformer batch. Games remain fully
independent; only neural inference is shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from app.engine.game import PlayerSpec
from app.engine.phases import Phase
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
class _EpisodeSlot:
    index: int
    seed: int
    env: WerewolfTrainingEnv
    samplers: dict[str, MaskedPolicySampler]
    trajectory: EpisodeTrajectory
    discussion_ticks: int = 0
    result: LearnedEpisodeResult | None = None


@dataclass(frozen=True)
class _InferenceRequest:
    slot_index: int
    player_id: str
    observation: PolicyObservation
    encoded: EncodedPolicyObservation


@dataclass(frozen=True)
class _PreparedRequest:
    request: _InferenceRequest
    logits: PolicyLogits


class TorchVectorizedEpisodeCollector:
    """Collect multiple independent shared-model episodes in lockstep batches."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        model: TorchTransformerPolicy,
        *,
        max_global_steps: int = 2000,
        max_discussion_ticks: int = 12,
        temperature: float = 1.0,
    ) -> None:
        self._player_specs = player_specs
        self._model = model
        self._max_global_steps = max_global_steps
        self._max_discussion_ticks = max_discussion_ticks
        self._temperature = temperature
        self._encoder = ObservationEncoder()

    def collect(self, seeds: tuple[int, ...]) -> tuple[LearnedEpisodeResult, ...]:
        if not seeds:
            raise ValueError("vectorized rollout requires at least one seed")
        if len(set(seeds)) != len(seeds):
            raise ValueError("vectorized rollout seeds must be unique")
        slots = [self._make_slot(index, seed) for index, seed in enumerate(seeds)]
        self._model.eval()

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

    def _make_slot(self, index: int, seed: int) -> _EpisodeSlot:
        env = WerewolfTrainingEnv(self._player_specs, seed=seed)
        samplers = {
            player_id: MaskedPolicySampler(
                seed=seed * 1000 + seat_index,
                temperature=self._temperature,
            )
            for seat_index, player_id in enumerate(env.controller.state.players)
        }
        return _EpisodeSlot(
            index=index,
            seed=seed,
            env=env,
            samplers=samplers,
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
            slot = slots[item.request.slot_index]
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
            slot = slots[item.request.slot_index]
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
            slot = slots[item.request.slot_index]
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
        )

    def _infer(
        self,
        requests: list[_InferenceRequest],
    ) -> tuple[_PreparedRequest, ...]:
        if not requests:
            return ()
        with torch.no_grad():
            output = self._model.forward_batch(
                tuple(request.encoded for request in requests)
            )
        return tuple(
            _PreparedRequest(
                request=request,
                logits=self._model.policy_logits_at(output, index),
            )
            for index, request in enumerate(requests)
        )


def _require_result(slot: _EpisodeSlot) -> LearnedEpisodeResult:
    if slot.result is None:
        raise RuntimeError(f"episode {slot.seed} is not finalized")
    return slot.result


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
