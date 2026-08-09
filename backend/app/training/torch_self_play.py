"""Full self-play training loop for the PyTorch Transformer policy."""

from __future__ import annotations

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.learned_runner import LearnedEpisodeResult
from app.training.self_play_train import SelfPlayBatchStats
from app.training.torch_policy import TorchTransformerPolicy, TransformerPolicyConfig
from app.training.torch_trainer import TorchPPOConfig, TorchPPOTrainer
from app.training.torch_vectorized import TorchVectorizedEpisodeCollector


class TorchSelfPlayTrainingLoop:
    """Generate batched 17-seat episodes, then PPO-update one shared model."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        *,
        model: TorchTransformerPolicy | None = None,
        model_config: TransformerPolicyConfig | None = None,
        ppo_config: TorchPPOConfig | None = None,
        max_discussion_ticks: int = 8,
        max_parallel_games: int = 8,
        temperature: float = 1.0,
        trainer_seed: int = 0,
    ) -> None:
        if model is not None and model_config is not None:
            raise ValueError("provide either model or model_config, not both")
        if max_parallel_games <= 0:
            raise ValueError("max_parallel_games must be positive")
        if temperature != 1.0:
            raise ValueError(
                "Torch PPO currently requires temperature=1 because traces do not store it"
            )
        self.player_specs = player_specs
        self.model = model or TorchTransformerPolicy(model_config)
        self.optimizer = TorchPPOTrainer(
            self.model,
            ppo_config,
            seed=trainer_seed,
        )
        self.max_discussion_ticks = max_discussion_ticks
        self.max_parallel_games = max_parallel_games
        self.temperature = temperature

    def train_batch(self, *, start_seed: int, episodes: int) -> SelfPlayBatchStats:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        seeds = tuple(start_seed + offset for offset in range(episodes))
        collector = TorchVectorizedEpisodeCollector(
            self.player_specs,
            self.model,
            max_discussion_ticks=self.max_discussion_ticks,
            temperature=self.temperature,
        )
        results: list[LearnedEpisodeResult] = []
        for start in range(0, len(seeds), self.max_parallel_games):
            results.extend(
                collector.collect(seeds[start : start + self.max_parallel_games])
            )
        trajectories = [result.trajectory for result in results]

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
