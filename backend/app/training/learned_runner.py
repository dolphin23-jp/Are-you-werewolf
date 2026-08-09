"""Full-episode runner for models that implement the learned-policy contract."""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.game import PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import Team
from app.training.env import WerewolfTrainingEnv
from app.training.learned_policy import LearnedStructuredPolicy, SpeechPolicyStep
from app.training.legal import legal_action_mask
from app.training.policy_contract import LearnedPolicyModel
from app.training.policy_sampling import PolicySampleTrace
from app.training.trajectory import DecisionKind, EpisodeTrajectory, RecordedDecision


@dataclass(frozen=True)
class LearnedEpisodeResult:
    winner: Team | None
    is_draw: bool
    days: int
    semantic_event_count: int
    rewards: dict[str, float]
    trajectory: EpisodeTrajectory


class LearnedEpisodeRunner:
    """Drive all 17 seats through one shared role-conditioned model."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        model: LearnedPolicyModel,
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

    def run(self, seed: int) -> LearnedEpisodeResult:
        env = WerewolfTrainingEnv(self._player_specs, seed=seed)
        policies = {
            player_id: LearnedStructuredPolicy(
                self._model,
                seed=seed * 1000 + index,
                temperature=self._temperature,
            )
            for index, player_id in enumerate(env.controller.state.players)
        }
        trajectory = EpisodeTrajectory(f"learned-{seed}")

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
                    trajectory=trajectory,
                )

            if state.phase is Phase.NIGHT:
                self._run_night(env, policies, trajectory)
                env.controller.resolve_night()
            elif state.phase is Phase.DAWN:
                env.controller.start_discussion()
                env.scheduler.reset()
            elif state.phase is Phase.DISCUSSION:
                self._run_discussion(env, policies, trajectory)
                env.controller.end_discussion()
            elif state.phase in (Phase.VOTING, Phase.RUNOFF):
                self._run_voting(env, policies, trajectory)
                env.controller.resolve_votes()
            elif state.phase is Phase.VOTE_RESULT:
                env.controller.start_night()
            else:
                raise RuntimeError(f"unexpected training phase {state.phase}")

        raise RuntimeError(f"episode exceeded {self._max_loops} phase loops")

    def _run_discussion(
        self,
        env: WerewolfTrainingEnv,
        policies: dict[str, LearnedStructuredPolicy],
        trajectory: EpisodeTrajectory,
    ) -> None:
        for _ in range(self._max_discussion_ticks):
            steps = {
                player_id: policies[player_id].speech_step(env.observe(player_id))
                for player_id in env.controller.state.alive_ids()
            }
            intents = {
                player_id: step.sampled.intent for player_id, step in steps.items()
            }
            selected = env.select_next_speaker(intents)
            selected_player_id = selected.player_id if selected else None
            self._record_discussion_cycle(steps, selected_player_id, trajectory)
            if selected is None:
                return
            bundle = steps[selected.player_id].sampled.intent.bundle
            if bundle is None:
                raise RuntimeError("scheduler selected a non-speaking learned intent")
            env.emit_speech(selected.player_id, bundle)

    @staticmethod
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

    @staticmethod
    def _run_voting(
        env: WerewolfTrainingEnv,
        policies: dict[str, LearnedStructuredPolicy],
        trajectory: EpisodeTrajectory,
    ) -> None:
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if not mask.vote_target_ids:
                continue
            step = policies[player_id].vote_step(env.observe(player_id), mask)
            env.vote(player_id, step.sampled.target_id)
            trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.VOTE,
                    observation=step.observation,
                    target_id=step.sampled.target_id,
                    policy_trace=step.sampled.trace,
                )
            )

    @staticmethod
    def _run_night(
        env: WerewolfTrainingEnv,
        policies: dict[str, LearnedStructuredPolicy],
        trajectory: EpisodeTrajectory,
    ) -> None:
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if not mask.night_choices:
                continue
            step = policies[player_id].night_step(env.observe(player_id), mask)
            env.night_action(player_id, step.sampled.topic, step.sampled.target_id)
            trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.NIGHT,
                    observation=step.observation,
                    target_id=step.sampled.target_id,
                    night_topic=step.sampled.topic,
                    policy_trace=step.sampled.trace,
                )
            )
