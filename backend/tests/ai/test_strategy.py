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


def test_rope_count_is_the_misses_left_after_a_night_attack_each_day():
    # 16 alive hold 7 executions (each day also loses one seat to the attack);
    # three of them must hit the three wolves, which leaves four to spare.
    assert StrategyAnalyzer()._rope_count(3, 16) == 4
    assert StrategyAnalyzer()._rope_count(1, 5) == 1
    assert StrategyAnalyzer()._rope_count(3, 6) == 0
