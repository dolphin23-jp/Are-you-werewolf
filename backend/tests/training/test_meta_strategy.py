from pathlib import Path

from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy, solve_logit_response_mixture
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable


def test_logit_response_prefers_empirically_better_policy(tmp_path: Path):
    table = PopulationPayoffTable(tmp_path / "payoffs.json")
    weak = PolicyProfile("v0", "w0", "f0")
    strong = PolicyProfile("v1", "w0", "f0")

    for _ in range(8):
        table.record_result(weak, winner=Team.WEREWOLF, is_draw=False, days=3)
        table.record_result(strong, winner=Team.VILLAGE, is_draw=False, days=3)

    strategy = solve_logit_response_mixture(
        table,
        temperature=0.2,
        iterations=20,
        damping=0.5,
    )
    village = {item.policy_id: item.probability for item in strategy.village}

    assert village["v1"] > village["v0"]
    assert abs(sum(village.values()) - 1.0) < 1e-12
    assert strategy.werewolf[0].probability == 1.0
    assert strategy.fox[0].probability == 1.0


def test_meta_strategy_round_trip_normalizes_weights(tmp_path: Path):
    table = PopulationPayoffTable(tmp_path / "payoffs.json")
    profile = PolicyProfile("v0", "w0", "f0")
    table.record_result(profile, winner=Team.FOX, is_draw=False, days=4)
    strategy = solve_logit_response_mixture(table, iterations=2)
    path = tmp_path / "strategy.json"

    strategy.save(path)
    restored = PopulationMetaStrategy.load(path)

    assert restored == strategy
    assert restored.village[0].probability == 1.0
    assert restored.werewolf[0].probability == 1.0
    assert restored.fox[0].probability == 1.0
