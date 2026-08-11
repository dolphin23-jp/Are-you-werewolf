"""Post-hoc strategy observation for frozen learned policies.

This module is intentionally downstream of self-play learning. It never writes to
population state, payoff tables, policy checkpoints, rewards, or action masks. It
replays immutable policies in ordinary games and records what each seat observed,
what it sampled, which speaker was selected, and what actions were actually
committed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from app.engine.game import PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import Team
from app.training.env import WerewolfTrainingEnv
from app.training.learned_policy import LearnedStructuredPolicy
from app.training.legal import legal_action_mask
from app.training.policy_contract import LearnedPolicyModel

_STRATEGY_OBSERVATION_VERSION = 1


class StrategyObservatoryRunner:
    """Run one fully observed post-hoc game from immutable faction policies."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        team_models: Mapping[Team, LearnedPolicyModel],
        policy_ids: Mapping[Team, str],
        *,
        max_loops: int = 200,
        max_discussion_ticks: int = 8,
        temperature: float = 1.0,
    ) -> None:
        if set(team_models) != set(Team):
            raise ValueError("team_models must contain exactly village, werewolf, and fox")
        if set(policy_ids) != set(Team):
            raise ValueError("policy_ids must contain exactly village, werewolf, and fox")
        if max_loops <= 0:
            raise ValueError("max_loops must be positive")
        if max_discussion_ticks < 0:
            raise ValueError("max_discussion_ticks cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._player_specs = player_specs
        self._team_models = dict(team_models)
        self._policy_ids = dict(policy_ids)
        self._max_loops = max_loops
        self._max_discussion_ticks = max_discussion_ticks
        self._temperature = temperature

    def run(self, seed: int) -> dict[str, Any]:
        env = WerewolfTrainingEnv(self._player_specs, seed=seed)
        players = {
            player_id: {
                "player_id": player_id,
                "role": player.role.value,
                "team": player.team.value,
                "policy_id": self._policy_ids[player.team],
            }
            for player_id, player in env.controller.state.players.items()
        }
        policies = {
            player_id: LearnedStructuredPolicy(
                self._team_models[player.team],
                seed=seed * 1000 + index,
                temperature=self._temperature,
            )
            for index, (player_id, player) in enumerate(
                env.controller.state.players.items()
            )
        }
        decisions: list[dict[str, Any]] = []

        for _ in range(self._max_loops):
            state = env.controller.state
            if state.phase is Phase.GAME_OVER:
                rewards = env.rewards()
                return {
                    "schema_version": _STRATEGY_OBSERVATION_VERSION,
                    "seed": seed,
                    "profile": {
                        team.value: self._policy_ids[team]
                        for team in Team
                    },
                    "winner": state.winner.value if state.winner is not None else None,
                    "is_draw": state.is_draw,
                    "days": state.day,
                    "first_victim_id": state.first_victim_id,
                    "players": list(players.values()),
                    "terminal_rewards": rewards,
                    "semantic_events": _json_value(env.semantic_events),
                    "decisions": decisions,
                }

            if state.phase is Phase.NIGHT:
                self._run_night(env, policies, players, decisions)
                env.controller.resolve_night()
            elif state.phase is Phase.DAWN:
                env.controller.start_discussion()
                env.scheduler.reset()
            elif state.phase is Phase.DISCUSSION:
                self._run_discussion(env, policies, players, decisions)
                env.controller.end_discussion()
            elif state.phase in (Phase.VOTING, Phase.RUNOFF):
                self._run_voting(env, policies, players, decisions)
                env.controller.resolve_votes()
            elif state.phase is Phase.VOTE_RESULT:
                env.controller.start_night()
            else:
                raise RuntimeError(f"unexpected observatory phase {state.phase}")

        raise RuntimeError(f"observatory episode exceeded {self._max_loops} phase loops")

    def _run_discussion(
        self,
        env: WerewolfTrainingEnv,
        policies: dict[str, LearnedStructuredPolicy],
        players: dict[str, dict[str, Any]],
        decisions: list[dict[str, Any]],
    ) -> None:
        for _ in range(self._max_discussion_ticks):
            player_ids = tuple(env.controller.state.alive_ids())
            observations = env.observe_many(player_ids)
            steps = {
                player_id: policies[player_id].speech_step(observations[player_id])
                for player_id in player_ids
            }
            intents = {
                player_id: step.sampled.intent for player_id, step in steps.items()
            }
            selected = env.select_next_speaker(intents)
            selected_player_id = selected.player_id if selected is not None else None

            for player_id in player_ids:
                sampled = steps[player_id].sampled
                intent = sampled.intent
                decisions.append(
                    {
                        "kind": "discussion_intent",
                        "player_id": player_id,
                        "role": players[player_id]["role"],
                        "team": players[player_id]["team"],
                        "policy_id": players[player_id]["policy_id"],
                        "observation": _json_value(observations[player_id]),
                        "timing": intent.timing.name.lower(),
                        "timing_index": int(intent.timing),
                        "selected": player_id == selected_player_id,
                        "sampled_bundle": _json_value(intent.bundle),
                        "policy_trace": _json_value(sampled.trace),
                    }
                )

            if selected is None:
                return
            bundle = steps[selected.player_id].sampled.intent.bundle
            if bundle is None:
                raise RuntimeError("scheduler selected a non-speaking observatory intent")
            env.emit_speech(selected.player_id, bundle)

    @staticmethod
    def _run_voting(
        env: WerewolfTrainingEnv,
        policies: dict[str, LearnedStructuredPolicy],
        players: dict[str, dict[str, Any]],
        decisions: list[dict[str, Any]],
    ) -> None:
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if not mask.vote_target_ids:
                continue
            observation = env.observe(player_id)
            step = policies[player_id].vote_step(observation, mask)
            env.vote(player_id, step.sampled.target_id)
            decisions.append(
                {
                    "kind": "vote",
                    "player_id": player_id,
                    "role": players[player_id]["role"],
                    "team": players[player_id]["team"],
                    "policy_id": players[player_id]["policy_id"],
                    "observation": _json_value(observation),
                    "target_id": step.sampled.target_id,
                    "policy_trace": _json_value(step.sampled.trace),
                }
            )

    @staticmethod
    def _run_night(
        env: WerewolfTrainingEnv,
        policies: dict[str, LearnedStructuredPolicy],
        players: dict[str, dict[str, Any]],
        decisions: list[dict[str, Any]],
    ) -> None:
        for player_id in env.controller.state.alive_ids():
            mask = legal_action_mask(env.controller, player_id)
            if not mask.night_choices:
                continue
            observation = env.observe(player_id)
            step = policies[player_id].night_step(observation, mask)
            env.night_action(player_id, step.sampled.topic, step.sampled.target_id)
            decisions.append(
                {
                    "kind": "night",
                    "player_id": player_id,
                    "role": players[player_id]["role"],
                    "team": players[player_id]["team"],
                    "policy_id": players[player_id]["policy_id"],
                    "observation": _json_value(observation),
                    "topic": step.sampled.topic.value,
                    "target_id": step.sampled.target_id,
                    "policy_trace": _json_value(step.sampled.trace),
                }
            )


def render_strategy_transcript(game: Mapping[str, Any]) -> str:
    """Render a compact human-readable view while JSON retains every intent."""

    profile = game["profile"]
    lines = [
        "===== STRATEGY OBSERVATION =====",
        f"seed={game['seed']}",
        (
            "profile="
            f"village={profile['village']} "
            f"werewolf={profile['werewolf']} fox={profile['fox']}"
        ),
        (
            f"winner={game['winner']} draw={str(game['is_draw']).lower()} "
            f"days={game['days']}"
        ),
        "",
        "===== ROLE ASSIGNMENT (POST-HOC ONLY) =====",
    ]
    for player in game["players"]:
        lines.append(
            f"{player['player_id']}: role={player['role']} team={player['team']} "
            f"policy={player['policy_id']}"
        )

    lines.extend(("", "===== EXECUTED TIMELINE ====="))
    for decision in game["decisions"]:
        observation = decision["observation"]
        day = observation["day"]
        phase = observation["phase"]
        player_id = decision["player_id"]
        role = decision["role"]
        policy_id = decision["policy_id"]
        kind = decision["kind"]
        if kind == "discussion_intent":
            if not decision["selected"]:
                continue
            tick = observation["discussion_tick"]
            bundle = _format_bundle(decision["sampled_bundle"])
            lines.append(
                f"day={day} tick={tick} SPEECH {player_id}({role},{policy_id}) "
                f"timing={decision['timing']} {bundle}"
            )
        elif kind == "vote":
            lines.append(
                f"day={day} phase={phase} VOTE {player_id}({role},{policy_id}) "
                f"-> {decision['target_id']}"
            )
        elif kind == "night":
            lines.append(
                f"day={day} phase={phase} NIGHT {player_id}({role},{policy_id}) "
                f"{decision['topic']} -> {decision['target_id']}"
            )

    lines.extend(("", "===== TIMING INTENTS ====="))
    for decision in game["decisions"]:
        if decision["kind"] != "discussion_intent":
            continue
        observation = decision["observation"]
        marker = " selected" if decision["selected"] else ""
        lines.append(
            f"day={observation['day']} tick={observation['discussion_tick']} "
            f"{decision['player_id']}({decision['policy_id']}) "
            f"timing={decision['timing']}{marker}"
        )
    lines.append("===== END STRATEGY OBSERVATION =====")
    return "\n".join(lines) + "\n"


def _format_bundle(bundle: Any) -> str:
    if not isinstance(bundle, dict):
        return ""
    atoms = bundle.get("atoms")
    if not isinstance(atoms, list):
        return ""
    rendered: list[str] = []
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        parts = [str(atom.get("action_type"))]
        for key in (
            "topic",
            "target_id",
            "secondary_target_id",
            "role",
            "result",
            "quantity",
            "referenced_day",
            "scope",
            "stance",
            "reference_event_id",
        ):
            value = atom.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        rendered.append("(" + " ".join(parts) + ")")
    return " ".join(rendered)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value
