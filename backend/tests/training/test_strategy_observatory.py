import json

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.strategy_observatory import (
    StrategyObservatoryRunner,
    render_strategy_transcript,
)
from app.training.uniform_model import UniformPolicyModel


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=False)
        for i in range(17)
    ]


def test_strategy_observatory_records_intents_actions_and_terminal_result():
    model = UniformPolicyModel()
    policy_ids = {
        Team.VILLAGE: "village-policy",
        Team.WEREWOLF: "werewolf-policy",
        Team.FOX: "fox-policy",
    }
    runner = StrategyObservatoryRunner(
        _specs(),
        {team: model for team in Team},
        policy_ids,
        max_discussion_ticks=4,
    )

    game = runner.run(seed=41)

    assert game["schema_version"] == 1
    assert game["is_draw"] or game["winner"] in {team.value for team in Team}
    assert len(game["players"]) == 17
    assert game["decisions"]
    assert any(
        decision["kind"] == "discussion_intent"
        for decision in game["decisions"]
    )
    assert any(decision["kind"] == "vote" for decision in game["decisions"])
    assert all("observation" in decision for decision in game["decisions"])
    assert all(
        player["policy_id"] == policy_ids[Team(player["team"])]
        for player in game["players"]
    )
    selected = [
        decision
        for decision in game["decisions"]
        if decision["kind"] == "discussion_intent" and decision["selected"]
    ]
    assert all(decision["sampled_bundle"] is not None for decision in selected)

    serialized = json.dumps(game, ensure_ascii=False)
    assert "discussion_intent" in serialized
    transcript = render_strategy_transcript(game)
    assert "ROLE ASSIGNMENT (POST-HOC ONLY)" in transcript
    assert "TIMING INTENTS" in transcript
