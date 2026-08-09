"""Full self-play training loop for the PyTorch Transformer policy."""

from __future__ import annotations

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.self_play_train import SelfPlayBatchStats
from app.training.torch_policy import TorchTransformerPolicy, TransformerPolicyConfig
from app.training.torch_runner import TorchBatchedEpisodeRunner
from app.training.torch_trainer import TorchPPOConfig, TorchPPOTrainer


class TorchSelfPlayTrainingLoop:
    """Generate 17-seat on-policy episodes, then PPO-update one shared model."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        *,
        model: TorchTransformerPolicy | None = None,
        model_config: TransformerPolicyConfig | None = None,
        ppo_config: TorchPPOConfig | None = None,
        max_discussion_ticks: int = 8,
        temperature: float = 1.0,
        trainer_seed: int = 0,
    ) -> None:
        if model is not None and model_config is not None:
            raise ValueError("provide either model or model_config, not both")
        self.player_specs = player_specs
        self.model = model or TorchTransformerPolicy(model_config)
        self.optimizer = TorchPPOTrainer(
            self.model,
            ppo_config,
            seed=trainer_seed,
        )
        self.max_discussion_ticks = max_discussion_ticks
        self.temperature = temperature

    def train_batch(self, *, start_seed: int, episodes: int) -> SelfPlayBatchStats:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        results = []
        trajectories = []
        self.model.eval()
        for offset in range(episodes):
            result = TorchBatchedEpisodeRunner(
                self.player_specs,
                self.model,
                max_discussion_ticks=self.max_discussion_ticks,
                temperature=self.temperature,
            ).run(start_seed + offset)
            results.append(result)
            trajectories.append(result.trajectory)

        update = self.optimizer.update(trajectories)
        return SelfPlayBatchStats(
            episodes=episodes,
            village_wins=sum(result.winner is Team.VILLAGE for result in results),
            werewolf_wins=sum(result.winner is Team.WEREWOLF for result in results),
            fox_wins=sum(result.winner is Team.FOX for result in results),
            draws=sum(result.is_draw for result in results),
            mean_days=sum(result.days for result in results) / episodes,
            mean_decisions=(
                sum(len(result.trajectory.decisions) for result in results) / episodes
            ),
            update=update,
        )
