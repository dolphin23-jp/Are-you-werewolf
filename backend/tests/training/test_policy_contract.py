from app.training.policy_contract import PolicyHeadSizes, PolicyLogits


def _zeros(width: int) -> tuple[float, ...]:
    return (0.0,) * width


def _valid_logits() -> PolicyLogits:
    sizes = PolicyHeadSizes()
    return PolicyLogits(
        timing=_zeros(sizes.timing),
        action_type=_zeros(sizes.action_type),
        topic=_zeros(sizes.topic),
        target=_zeros(sizes.target),
        secondary_target=_zeros(sizes.secondary_target),
        role=_zeros(sizes.role),
        result=_zeros(sizes.result),
        quantity=_zeros(sizes.quantity),
        referenced_day=_zeros(sizes.referenced_day),
        scope=_zeros(sizes.scope),
        stance=_zeros(sizes.stance),
        vote_target=_zeros(sizes.vote_target),
        night_topic=_zeros(sizes.night_topic),
        night_target=_zeros(sizes.night_target),
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
