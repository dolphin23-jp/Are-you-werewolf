from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.actions import ActionType, TimingBucket, Topic
from app.training.env import WerewolfTrainingEnv
from app.training.legal import LegalActionMask
from app.training.policy_contract import PolicyHeadSizes, PolicyLogits
from app.training.policy_sampling import MaskedPolicySampler


def _env() -> WerewolfTrainingEnv:
    specs = [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0))
        for i in range(17)
    ]
    return WerewolfTrainingEnv(
        specs,
        seed=23,
        forced_roles={"p0": RoleName.VILLAGER},
    )


def _head(width: int, hot_index: int | None = None) -> tuple[float, ...]:
    values = [-20.0] * width
    if hot_index is not None:
        values[hot_index] = 20.0
    return tuple(values)


def _logits(
    *,
    timing: TimingBucket = TimingBucket.IMMEDIATE,
    action_type: ActionType = ActionType.CLAIM,
    topic: Topic = Topic.WOLF_COUNT,
    quantity: int = 1,
    target_index: int = 0,
) -> PolicyLogits:
    sizes = PolicyHeadSizes()
    return PolicyLogits(
        timing=_head(sizes.timing, tuple(TimingBucket).index(timing)),
        action_type=_head(sizes.action_type, tuple(ActionType).index(action_type)),
        topic=_head(sizes.topic, tuple(Topic).index(topic)),
        target=_head(sizes.target, target_index),
        secondary_target=_head(sizes.secondary_target, target_index),
        role=_head(sizes.role, 0),
        result=_head(sizes.result, 0),
        quantity=_head(sizes.quantity, quantity),
        referenced_day=_head(sizes.referenced_day, 0),
        scope=_head(sizes.scope, 0),
        stance=_head(sizes.stance, 0),
        reference_event=_head(sizes.reference_event, 0),
        vote_target=_head(sizes.vote_target, target_index),
        night_topic=_head(sizes.night_topic, 0),
        night_target=_head(sizes.night_target, target_index),
        value=0.25,
    )


def test_sampler_can_emit_lw_count_claim_and_records_factorized_trace():
    observation = _env().observe("p0")
    sampler = MaskedPolicySampler(seed=1)

    sampled = sampler.sample_speech(observation, _logits(quantity=1))

    assert sampled.intent.timing is TimingBucket.IMMEDIATE
    assert sampled.intent.bundle is not None
    atom = sampled.intent.bundle.atoms[0]
    assert atom.action_type is ActionType.CLAIM
    assert atom.topic is Topic.WOLF_COUNT
    assert atom.quantity == 1
    assert [choice.head for choice in sampled.trace.choices] == [
        "timing",
        "action_type",
        "topic",
        "quantity",
    ]
    assert sampled.trace.value_estimate == 0.25


def test_impossible_day_zero_report_is_masked_even_with_dominant_logit():
    observation = _env().observe("p0")
    sampler = MaskedPolicySampler(seed=2)
    sizes = PolicyHeadSizes()
    logits = _logits()
    action_values = list(logits.action_type)
    action_values[tuple(ActionType).index(ActionType.REPORT)] = 100.0
    action_values[tuple(ActionType).index(ActionType.CLAIM)] = 20.0
    logits = PolicyLogits(
        timing=logits.timing,
        action_type=tuple(action_values),
        topic=logits.topic,
        target=logits.target,
        secondary_target=logits.secondary_target,
        role=logits.role,
        result=logits.result,
        quantity=logits.quantity,
        referenced_day=logits.referenced_day,
        scope=logits.scope,
        stance=logits.stance,
        reference_event=logits.reference_event,
        vote_target=logits.vote_target,
        night_topic=logits.night_topic,
        night_target=logits.night_target,
        value=logits.value,
    )

    sampled = sampler.sample_speech(observation, logits)

    assert sampled.intent.bundle is not None
    assert sampled.intent.bundle.atoms[0].action_type is ActionType.CLAIM
    assert tuple(ActionType).index(ActionType.REPORT) not in sampled.trace.choices[1].valid_indices
    assert len(logits.action_type) == sizes.action_type


def test_vote_sampler_never_uses_non_masked_seat():
    observation = _env().observe("p0")
    sampler = MaskedPolicySampler(seed=3)
    logits = _logits(target_index=16)
    mask = LegalActionMask(
        action_types=(ActionType.VOTE,),
        vote_target_ids=("p4", "p5"),
    )

    sampled = sampler.sample_vote(observation, mask, logits)

    assert sampled.target_id in {"p4", "p5"}
    assert set(sampled.trace.choices[0].valid_indices) == {4, 5}
