from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.engine.state import DeathCause, DeathRecord
from app.training.actions import ActionType, Scope, Topic
from app.training.env import WerewolfTrainingEnv
from app.training.parameters import semantic_parameter_mask


def _env() -> WerewolfTrainingEnv:
    specs = [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0))
        for i in range(17)
    ]
    return WerewolfTrainingEnv(
        specs,
        seed=19,
        forced_roles={"p0": RoleName.FOX, "p1": RoleName.WEREWOLF},
    )


def test_claim_mask_supports_role_partner_and_lw_count_without_doctrine():
    observation = _env().observe("p0")

    base = semantic_parameter_mask(observation, ActionType.CLAIM)
    count = semantic_parameter_mask(observation, ActionType.CLAIM, topic=Topic.WOLF_COUNT)

    assert Topic.ROLE in base.topics
    assert Topic.PARTNER in base.topics
    assert Topic.WOLF_COUNT in base.topics
    assert count.quantities == (1, 2, 3)


def test_execution_proposal_can_target_self_for_pillar_primitive():
    observation = _env().observe("p0")
    mask = semantic_parameter_mask(
        observation,
        ActionType.PROPOSE,
        topic=Topic.EXECUTION,
    )

    assert "p0" in mask.target_ids
    assert Scope.SELF in mask.scopes


def test_medium_report_targets_are_derived_from_public_execution_only():
    env = _env()
    state = env.controller.state
    state.players["p7"].alive = False
    state.players["p7"].death_cause = DeathCause.EXECUTED
    state.players["p7"].death_day = 1
    state.death_records.append(DeathRecord("p7", DeathCause.EXECUTED, 1))
    state.players["p8"].alive = False
    state.players["p8"].death_cause = DeathCause.ATTACKED
    state.players["p8"].death_day = 1
    state.death_records.append(DeathRecord("p8", DeathCause.ATTACKED, 1))
    state.day = 2

    observation = env.observe("p0")
    mask = semantic_parameter_mask(
        observation,
        ActionType.REPORT,
        topic=Topic.MEDIUM_RESULT,
    )

    assert "p7" in mask.target_ids
    assert "p8" not in mask.target_ids


def test_public_claim_mask_does_not_depend_on_viewers_true_role():
    env = _env()
    fox = semantic_parameter_mask(env.observe("p0"), ActionType.CLAIM, topic=Topic.ROLE)
    wolf = semantic_parameter_mask(env.observe("p1"), ActionType.CLAIM, topic=Topic.ROLE)

    assert fox.roles == wolf.roles == tuple(RoleName)
