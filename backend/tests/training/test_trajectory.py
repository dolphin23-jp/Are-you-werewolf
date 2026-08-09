from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.actions import ActionType, SemanticAction, SpeechBundle, Topic
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.trajectory import DecisionKind, EpisodeTrajectory, RecordedDecision


def _encoded_observation():
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    env = WerewolfTrainingEnv(
        specs,
        seed=31,
        forced_roles={"p0": RoleName.VILLAGER},
    )
    return ObservationEncoder().encode(env.observe("p0"))


def test_finalize_attaches_only_terminal_team_reward():
    trajectory = EpisodeTrajectory("episode-1")
    trajectory.append(
        RecordedDecision(
            player_id="p0",
            kind=DecisionKind.SPEECH,
            observation=_encoded_observation(),
            speech_bundle=SpeechBundle(
                (SemanticAction(ActionType.EVALUATE, topic=Topic.WOLF, target_id="p1"),)
            ),
        )
    )

    trajectory.finalize({"p0": 1.0, "p1": -1.0})

    assert trajectory.finalized is True
    assert trajectory.decisions[0].reward == 1.0
    assert trajectory.terminal_rewards["p1"] == -1.0


def test_invalid_mixed_decision_payload_is_rejected():
    try:
        RecordedDecision(
            player_id="p0",
            kind=DecisionKind.VOTE,
            observation=_encoded_observation(),
            target_id="p1",
            night_topic=Topic.ATTACK,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mixed vote/night payload should be rejected")
