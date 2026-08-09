import pytest

from app.engine.game import PlayerSpec
from app.engine.roles import RoleName
from app.training.actions import ActionType
from app.training.encoding import ObservationEncoder
from app.training.env import WerewolfTrainingEnv
from app.training.legal import LegalActionMask
from app.training.policy_sampling import HeadChoice, MaskedPolicySampler, PolicySampleTrace
from app.training.trajectory import DecisionKind, EpisodeTrajectory, RecordedDecision

torch = pytest.importorskip("torch")
torch_policy = pytest.importorskip("app.training.torch_policy")
torch_trainer = pytest.importorskip("app.training.torch_trainer")
TorchTransformerPolicy = torch_policy.TorchTransformerPolicy
TransformerPolicyConfig = torch_policy.TransformerPolicyConfig
TorchPPOConfig = torch_trainer.TorchPPOConfig
TorchPPOTrainer = torch_trainer.TorchPPOTrainer
_trace_log_prob = torch_trainer._trace_log_prob
_trace_log_probs = torch_trainer._trace_log_probs
_trace_log_probs_and_entropy = torch_trainer._trace_log_probs_and_entropy


def _env() -> WerewolfTrainingEnv:
    specs = [PlayerSpec(player_id=f"p{i}", name=f"Player{i}") for i in range(17)]
    return WerewolfTrainingEnv(
        specs,
        seed=501,
        forced_roles={"p0": RoleName.VILLAGER},
    )


def _model():
    return TorchTransformerPolicy(
        TransformerPolicyConfig(
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
    )


def _parameters(model):
    return torch.cat(
        [parameter.detach().cpu().flatten() for parameter in model.parameters()]
    )


def _vote_trajectory(model, *, reward: float, episode_id: str) -> EpisodeTrajectory:
    env = _env()
    public_observation = env.observe("p0")
    encoded = ObservationEncoder().encode(public_observation)
    logits = model.forward(encoded)
    sampled = MaskedPolicySampler(seed=503).sample_vote(
        public_observation,
        LegalActionMask(
            action_types=(ActionType.VOTE,),
            vote_target_ids=("p1", "p2", "p3"),
        ),
        logits,
    )
    trajectory = EpisodeTrajectory(episode_id)
    trajectory.append(
        RecordedDecision(
            player_id="p0",
            kind=DecisionKind.VOTE,
            observation=encoded,
            target_id=sampled.target_id,
            policy_trace=sampled.trace,
        )
    )
    trajectory.finalize({"p0": reward})
    return trajectory


def _scalar_path_entropy(output, batch_index: int, trace: PolicySampleTrace):
    total = output.value.new_zeros(())
    for choice in trace.choices:
        logits = output.head(choice.head)[batch_index]
        valid_indices = torch.tensor(
            choice.valid_indices,
            dtype=torch.long,
            device=logits.device,
        )
        valid_logits = logits.index_select(0, valid_indices)
        log_probs = torch.log_softmax(valid_logits, dim=0)
        total = total - (log_probs.exp() * log_probs).sum()
    return total


def test_vectorized_trace_log_probs_match_scalar_reference_for_mixed_heads():
    torch.manual_seed(499)
    model = _model().train()
    encoded = ObservationEncoder().encode(_env().observe("p0"))
    output = model.forward_batch((encoded, encoded, encoded))
    traces = [
        PolicySampleTrace(
            (
                HeadChoice("timing", 1, (0, 1, 2, 3, 4), 0.0),
                HeadChoice("action_type", 2, (0, 2, 4), 0.0),
                HeadChoice("target", 3, (1, 3, 5), 0.0),
            ),
            0.0,
        ),
        PolicySampleTrace(
            (HeadChoice("vote_target", 2, (1, 2, 3), 0.0),),
            0.0,
        ),
        PolicySampleTrace(
            (
                HeadChoice("night_topic", 1, (0, 1, 2), 0.0),
                HeadChoice("night_target", 4, (4, 6, 8), 0.0),
            ),
            0.0,
        ),
    ]

    vectorized, path_entropy = _trace_log_probs_and_entropy(output, traces)
    scalar = torch.stack(
        [_trace_log_prob(output, index, trace) for index, trace in enumerate(traces)]
    )
    scalar_entropy = torch.stack(
        [
            _scalar_path_entropy(output, index, trace)
            for index, trace in enumerate(traces)
        ]
    )

    assert vectorized.shape == (3,)
    assert path_entropy.shape == (3,)
    assert torch.allclose(_trace_log_probs(output, traces), vectorized)
    assert torch.allclose(vectorized, scalar, atol=1e-6, rtol=1e-6)
    assert torch.allclose(path_entropy, scalar_entropy, atol=1e-6, rtol=1e-6)
    assert torch.all(path_entropy >= 0.0)


def test_vectorized_trace_log_prob_matches_rollout_probability():
    torch.manual_seed(501)
    model = _model().eval()
    trajectory = _vote_trajectory(model, reward=1.0, episode_id="on-policy")
    decision = trajectory.decisions[0]
    trace = decision.policy_trace
    assert trace is not None

    model.train()
    output = model.forward_batch((decision.observation,))
    current = _trace_log_probs(output, [trace])[0]

    assert float(current.detach().cpu()) == pytest.approx(trace.log_prob, abs=1e-6)


def test_torch_ppo_changes_transformer_parameters_from_terminal_reward():
    torch.manual_seed(505)
    model = _model()
    trajectory = _vote_trajectory(model, reward=1.0, episode_id="torch-ppo")
    before = _parameters(model).clone()

    stats = TorchPPOTrainer(
        model,
        TorchPPOConfig(
            learning_rate=1e-3,
            epochs=2,
            minibatch_size=1,
        ),
        seed=507,
    ).update([trajectory])
    after = _parameters(model)

    assert stats.decisions == 1
    assert stats.epochs == 2
    assert torch.isfinite(torch.tensor(stats.mean_policy_loss))
    assert torch.isfinite(torch.tensor(stats.mean_value_loss))
    assert torch.isfinite(torch.tensor(stats.mean_approx_kl))
    assert torch.isfinite(torch.tensor(stats.mean_path_entropy))
    assert stats.mean_approx_kl >= 0.0
    assert stats.mean_path_entropy > 0.0
    assert stats.rollout_value_explained_variance == 0.0
    assert stats.gradient_norm > 0.0
    assert not torch.allclose(before, after)


def test_torch_ppo_batches_multiple_traced_decisions():
    torch.manual_seed(509)
    model = _model()
    positive = _vote_trajectory(model, reward=1.0, episode_id="positive")
    negative = _vote_trajectory(model, reward=-1.0, episode_id="negative")

    stats = TorchPPOTrainer(
        model,
        TorchPPOConfig(
            learning_rate=1e-3,
            epochs=1,
            minibatch_size=2,
        ),
        seed=511,
    ).update([positive, negative])

    assert stats.decisions == 2
    assert stats.epochs == 1
    assert stats.mean_ratio > 0.0
    assert 0.0 <= stats.clip_fraction <= 1.0
    assert stats.mean_approx_kl >= 0.0
    assert stats.mean_path_entropy > 0.0
    assert torch.isfinite(torch.tensor(stats.rollout_value_explained_variance))
    assert stats.rollout_value_explained_variance <= 1.0
    assert stats.gradient_norm > 0.0


def test_torch_ppo_rejects_unfinalized_trajectory():
    torch.manual_seed(513)
    model = _model()
    trajectory = _vote_trajectory(model, reward=1.0, episode_id="unfinished")
    trajectory.finalized = False

    with pytest.raises(ValueError, match="requires finalized"):
        TorchPPOTrainer(model).update([trajectory])


def test_torch_ppo_rejects_dropout_until_trace_probability_tracks_it():
    model = TorchTransformerPolicy(
        TransformerPolicyConfig(
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.1,
        )
    )

    with pytest.raises(ValueError, match="requires dropout=0"):
        TorchPPOTrainer(model)
