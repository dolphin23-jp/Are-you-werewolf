from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.evaluation import evaluate_faction
from app.training.uniform_model import UniformPolicyModel


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def test_faction_evaluation_accounts_for_every_game():
    stats = evaluate_faction(
        _specs(),
        UniformPolicyModel(),
        UniformPolicyModel(),
        team=Team.VILLAGE,
        seeds=(71,),
        max_discussion_ticks=3,
    )

    assert stats.games == 1
    assert stats.wins + stats.losses + stats.draws == 1
    assert 0.0 <= stats.win_rate <= 1.0
    assert stats.mean_days >= 0
