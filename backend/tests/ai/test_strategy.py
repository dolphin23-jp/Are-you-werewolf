from app.ai.strategy import StrategyAnalyzer
from app.engine.roles import RoleName
from tests.conftest import make_controller


def test_gray_list_excludes_co_and_named_players():
    controller = make_controller(seed=2)
    ids = controller.state.alive_ids()
    controller.state.day = 1
    controller.co(ids[0], RoleName.SEER.value)
    analysis = StrategyAnalyzer().analyze(controller.state)
    assert ids[0] not in analysis.gray_player_ids
    assert ids[1] in analysis.gray_player_ids


def test_rope_count_decreases_as_wolves_survive_relatively_more():
    baseline = StrategyAnalyzer()._rope_count(3, 17)
    fewer_non_wolves = StrategyAnalyzer()._rope_count(3, 8)
    assert fewer_non_wolves < baseline
    assert baseline >= 0
    assert fewer_non_wolves >= 0
