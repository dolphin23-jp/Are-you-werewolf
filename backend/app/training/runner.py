"""Episode runners used before any neural policy is introduced."""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.game import PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import Team
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.legal import legal_action_mask
from app.training.random_policy import RandomPolicy
from app.training.trajectory import DecisionKind, EpisodeTrajectory, RecordedDecision


@dataclass(frozen=True)
class EpisodeResult:
    winner: Team | None
    is_draw: bool
    days: int
    semantic_event_count: int
    rewards: dict[str, float]
    trajectory: EpisodeTrajectory


class RandomEpisodeRunner:
    """Drive a whole game through the new training protocol without an LLM."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        *,
        max_loops: int = 200,
        max_discussion_ticks: int = 12,
    ) -> None:
        self._player_specs = player_specs
        self._max_loops = max_loops
        self._max_discussion_ticks = max_discussion_ticks
        self._encoder = ObservationEncoder()

    def run(self, seed: int) -> EpisodeResult:
        env = WerewolfTrainingEnv(self._player_specs, seed=seed)
        trajectory = EpisodeTrajectory(episode_id=f"random-{seed}")
        policies = {
            player_id: RandomPolicy(seed=seed * 1000 + index)
            for index, player_id in enumerate(env.controller.state.players)
        }

        for _ in range(self._max_loops):
            state = env.controller.state
            if state.phase is Phase.GAME_OVER:
                rewards = env.rewards()
                trajectory.finalize(rewards)
                return EpisodeResult(
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
        policies: dict[str, RandomPolicy],
        trajectory: EpisodeTrajectory,
    ) -> None:
        for _ in range(self._max_discussion_ticks):
            observations = {
                player_id: env.observe(player_id)
                for player_id in env.controller.state.alive_ids()
            }
            intents = {
                player_id: policies[player_id].speak_intent(observation)
                for player_id, observation in observations.items()
            }
            selected = env.select_next_speaker(intents)
            if selected is None:
                return
            bundle = intents[selected.player_id].bundle
            if bundle is None:
                raise RuntimeError("scheduler selected a HOLD intent")
            trajectory.append(
                RecordedDecision(
                    player_id=selected.player_id,
                    kind=DecisionKind.SPEECH,
                    observation=self._encoder.encode(observations[selected.player_id]),
                    speech_bundle=bundle,
                )
            )
            env.emit_speech(selected.player_id, bundle)

    def _run_voting(
        self,
        env: WerewolfTrainingEnv,
        policies: dict[str, RandomPolicy],
        trajectory: EpisodeTrajectory,
    ) -> None:
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if not mask.vote_target_ids:
                continue
            target_id = policies[player_id].vote_target(mask)
            trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.VOTE,
                    observation=self._encoder.encode(env.observe(player_id)),
                    target_id=target_id,
                )
            )
            env.vote(player_id, target_id)

    def _run_night(
        self,
        env: WerewolfTrainingEnv,
        policies: dict[str, RandomPolicy],
        trajectory: EpisodeTrajectory,
    ) -> None:
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if not mask.night_choices:
                continue
            topic, target_id = policies[player_id].night_choice(mask)
            trajectory.append(
                RecordedDecision(
                    player_id=player_id,
                    kind=DecisionKind.NIGHT,
                    observation=self._encoder.encode(env.observe(player_id)),
                    target_id=target_id,
                    night_topic=topic,
                )
            )
            env.night_action(player_id, topic, target_id)
