from dataclasses import replace

import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig


def _observation():
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    env = WerewolfTrainingEnv(
        specs,
        seed=401,
        forced_roles={"p0": RoleName.VILLAGER},
    )
    return ObservationEncoder().encode(env.observe("p0"))


def _model():
    return TorchTransformerPolicy(
        TransformerPolicyConfig(
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.0,
        )
    )


def test_transformer_matches_framework_agnostic_policy_contract():
    torch.manual_seed(7)
    model = _model().eval()
    observation = _observation()

    logits = model.forward(observation)

    logits.validate()
    assert len(logits.target) == 17
    assert len(logits.vote_target) == 17
    assert len(logits.night_target) == 17
    assert len(logits.reference_event) == 128
    assert all(
        torch.isfinite(torch.tensor(values)).all()
        for values in (
            logits.timing,
            logits.action_type,
            logits.topic,
            logits.target,
            logits.reference_event,
        )
    )


def test_transformer_batch_forward_is_differentiable():
    torch.manual_seed(11)
    model = _model().train()
    observation = _observation()

    output = model.forward_batch((observation, observation))
    loss = (
        output.action_type.mean()
        + output.target.mean()
        + output.reference_event.mean()
        + output.value.mean()
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert output.action_type.shape[0] == 2
    assert output.target.shape == (2, 17)
    assert output.reference_event.shape == (2, 128)
    assert output.value.shape == (2,)
    assert gradients
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_padding_content_cannot_change_unmasked_policy_context():
    torch.manual_seed(13)
    model = _model().eval()
    observation = _observation()
    assert observation.semantic_mask[-1] == 0
    altered_tokens = list(observation.semantic_tokens)
    altered_tokens[-1] = (1,) * len(altered_tokens[-1])
    altered = replace(observation, semantic_tokens=tuple(altered_tokens))

    with torch.no_grad():
        original = model.forward_batch((observation,))
        changed = model.forward_batch((altered,))

    assert torch.allclose(original.action_type, changed.action_type, atol=1e-6)
    assert torch.allclose(original.target, changed.target, atol=1e-6)
    assert torch.allclose(original.value, changed.value, atol=1e-6)


def test_same_seed_produces_same_initial_policy():
    observation = _observation()
    torch.manual_seed(17)
    first = _model().eval()
    torch.manual_seed(17)
    second = _model().eval()

    first_logits = first.forward(observation)
    second_logits = second.forward(observation)

    assert first_logits == second_logits
