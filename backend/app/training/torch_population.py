"""Empirical three-faction payoff evaluation for Transformer populations."""

from __future__ import annotations

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable, ProfilePayoff
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_runner import TorchBatchedEpisodeRunner


def evaluate_torch_policy_profile(
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    table: PopulationPayoffTable,
    profile: PolicyProfile,
    *,
    seeds: tuple[int, ...],
    max_discussion_ticks: int = 8,
    temperature: float = 1.0,
) -> ProfilePayoff:
    """Run missing games for one immutable three-Transformer profile."""
    if not seeds:
        raise ValueError("profile evaluation requires at least one seed")
    village_model = pool.load(profile.village).eval()
    werewolf_model = pool.load(profile.werewolf).eval()
    fox_model = pool.load(profile.fox).eval()
    team_models = {
        Team.VILLAGE: village_model,
        Team.WEREWOLF: werewolf_model,
        Team.FOX: fox_model,
    }

    for seed in seeds:
        result = TorchBatchedEpisodeRunner(
            player_specs,
            village_model,
            team_models=team_models,
            max_discussion_ticks=max_discussion_ticks,
            temperature=temperature,
        ).run(seed)
        table.record_result(
            profile,
            winner=result.winner,
            is_draw=result.is_draw,
            days=result.days,
        )

    record = table.get(profile)
    if record is None:
        raise RuntimeError("profile evaluation did not create a payoff record")
    return record
