from pathlib import Path

import numpy as np
import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv

torch = pytest.importorskip("torch")
torch_checkpoint = pytest.importorskip("app.training.torch_checkpoint")
torch_policy = pytest.importorskip("app.training.torch_policy")
load_torch_policy = torch_checkpoint.load_torch_policy
save_torch_policy = torch_checkpoint.save_torch_policy
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig


def _observation():
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    env = WerewolfTrainingEnv(
        specs,
        seed=601,
        forced_roles={"p0": RoleName.VILLAGER},
    )
    return ObservationEncoder().encode(env.observe("p0"))


def _model():
    return TorchTransformerPolicy(
        TransformerPolicyConfig(
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
    ).eval()


def test_transformer_checkpoint_round_trip_preserves_logits(tmp_path: Path):
    torch.manual_seed(603)
    model = _model()
    observation = _observation()
    before = model.forward(observation)
    path = tmp_path / "policy.npz"

    save_torch_policy(model, path)
    restored = load_torch_policy(path).eval()
    after = restored.forward(observation)

    assert restored.config == model.config
    assert restored.sizes == model.sizes
    assert before == after
    for original, loaded in zip(
        model.parameters(),
        restored.parameters(),
        strict=True,
    ):
        assert torch.equal(original, loaded)


def test_transformer_checkpoint_contains_no_pickle_objects(tmp_path: Path):
    torch.manual_seed(605)
    path = tmp_path / "safe-policy.npz"
    save_torch_policy(_model(), path)

    with np.load(path, allow_pickle=False) as archive:
        assert "__metadata__" in archive.files
        assert archive["__metadata__"].dtype == np.uint8
        assert all(archive[key].dtype != object for key in archive.files)
