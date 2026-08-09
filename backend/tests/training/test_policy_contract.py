from app.training.policy_contract import PolicyHeadSizes, PolicyLogits


def _valid_logits() -> PolicyLogits:
    sizes = PolicyHeadSizes()
    zeros = lambda width: (0.0,) * width
    return PolicyLogits(
        timing=zeros(sizes.timing),
        action_type=zeros(sizes.action_type),
        topic=zeros(sizes.topic),
        target=zeros(sizes.target),
        secondary_target=zeros(sizes.secondary_target),
        role=zeros(sizes.role),
        result=zeros(sizes.result),
        quantity=zeros(sizes.quantity),
        referenced_day=zeros(sizes.referenced_day),
        scope=zeros(sizes.scope),
        stance=zeros(sizes.stance),
        vote_target=zeros(sizes.vote_target),
        night_topic=zeros(sizes.night_topic),
        night_target=zeros(sizes.night_target),
        value=0.0,
    )


def test_policy_head_contract_validates_exact_widths():
    logits = _valid_logits()
    logits.validate()


def test_policy_head_contract_rejects_wrong_target_width():
    logits = _valid_logits()
    bad = PolicyLogits(
        timing=logits.timing,
        action_type=logits.action_type,
        topic=logits.topic,
        target=logits.target[:-1],
        secondary_target=logits.secondary_target,
        role=logits.role,
        result=logits.result,
        quantity=logits.quantity,
        referenced_day=logits.referenced_day,
        scope=logits.scope,
        stance=logits.stance,
        vote_target=logits.vote_target,
        night_topic=logits.night_topic,
        night_target=logits.night_target,
        value=logits.value,
    )

    try:
        bad.validate()
    except ValueError as exc:
        assert "target head" in str(exc)
    else:
        raise AssertionError("wrong output width should be rejected")
