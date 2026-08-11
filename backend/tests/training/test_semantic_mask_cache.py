from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.actions import ActionType, Topic
from app.training.env import WerewolfTrainingEnv
from app.training.parameters import (
    _build_semantic_parameter_mask,
    semantic_parameter_mask,
)


def _observations():
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    env = WerewolfTrainingEnv(
        specs,
        seed=911,
        forced_roles={"p0": RoleName.VILLAGER},
    )
    return env.observe("p0"), env.observe("p1")


def test_cached_semantic_masks_match_uncached_builder_for_all_action_topic_pairs():
    observations = _observations()

    for observation in observations:
        for action_type in ActionType:
            for topic in (None, *tuple(Topic)):
                cached = semantic_parameter_mask(
                    observation,
                    action_type,
                    topic=topic,
                )
                expected = _build_semantic_parameter_mask(
                    observation,
                    action_type,
                    topic=topic,
                )
                assert cached == expected


def test_topic_insensitive_queries_reuse_the_same_cached_mask():
    observation, _ = _observations()

    base = semantic_parameter_mask(observation, ActionType.EVALUATE)
    topic_specific = semantic_parameter_mask(
        observation,
        ActionType.EVALUATE,
        topic=Topic.WOLF,
    )

    assert topic_specific is base


def test_cache_is_reset_for_a_different_observation_object():
    first_observation, second_observation = _observations()

    first = semantic_parameter_mask(first_observation, ActionType.QUESTION)
    second = semantic_parameter_mask(second_observation, ActionType.QUESTION)

    assert first == _build_semantic_parameter_mask(
        first_observation,
        ActionType.QUESTION,
    )
    assert second == _build_semantic_parameter_mask(
        second_observation,
        ActionType.QUESTION,
    )
    assert second is not first
