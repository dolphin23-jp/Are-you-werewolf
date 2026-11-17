"""Public lines the seed-1 review found missing: each one a fact the code
already held and the memo did not say (seed1-review-causes.md, R2 and §2.3).

Every line is pinned at the moment of run #13 seed 1 where its absence showed.
"""

from __future__ import annotations

from app.engine.roles import RoleName
from tests.ai.reasoning import game1_boards as g1
from tests.ai.reasoning import run13_seed1_boards as s1
from tests.ai.reasoning.solver import boards


def _public(day: int, at: str = "morning", seat: str = s1.REN) -> str:
    return s1.public_part(s1.memo(s1.board(day, at), seat))


def _line(public: str, prefix: str) -> str:
    return next(line for line in public.splitlines() if line.startswith(prefix))


# -- 3. the medium, execution by execution --


def test_after_the_medium_dies_every_execution_reads_unpublished_for_good():
    """4-B1 / 5-B4: アカリ was bitten on night 3. From day 4 nobody alive had
    claimed medium, and the memo said nothing of what that meant."""
    line = _line(_public(4), "- 処刑者の霊媒結果: ")

    assert line == (
        "- 処刑者の霊媒結果: 1日目処刑 ユイ=白(アカリ・2日目公開) / "
        "2日目処刑 アオイ=白(アカリ・3日目公開) / 3日目処刑 ダイキ=未公開。"
        "生存する霊媒COなし: 新たに霊媒COが出ない限り、処刑者の結果は公開されない"
    )


def test_while_a_medium_claimant_lives_the_result_is_only_not_out_yet():
    line = _line(_public(3), "- 処刑者の霊媒結果: ")

    assert line == (
        "- 処刑者の霊媒結果: 1日目処刑 ユイ=白(アカリ・2日目公開) / 2日目処刑 アオイ=未公開"
    )


# -- 4. every night's corpses, and what an empty morning means --

_QUIET = "死体なし＝襲撃が護衛で防がれたか、襲撃先が妖狐"


def test_every_night_has_its_own_line_and_an_empty_morning_is_read_for_the_table():
    """2-B1 / 3-B3 / 5-B5: the three quiet mornings were a "(死体なし)" inside
    the seer table's row headers, and nobody read them."""
    line = _line(_public(5), "- 夜の結果: ")

    assert line == (
        "- 夜の結果: 1日目夜 死体なし / 2日目夜 死体なし / 3日目夜 死体: アカリ / "
        f"4日目夜 死体なし（{_QUIET}）"
    )


def test_the_night_line_does_not_depend_on_a_seer_claimant_having_acted():
    """With no seer claim on the board there is no divination table, and the
    night's only record used to be in that table."""
    state = boards.deal({"p1": RoleName.WEREWOLF, "p5": RoleName.HUNTER}, day=2)
    boards.kill_first_victim(state, "p16")
    boards.execute(state, "p3", day=1)
    boards.guard(state, "p5", "p4", night=1)
    boards.attack(state, "p1", "p4", night=1, succeeded=False)
    public = s1.public_part(s1.memo(state, "p6"))

    assert f"- 夜の結果: 1日目夜 死体なし（{_QUIET}）" in public
    assert "占いCOの夜ごとの占い先" not in public


def test_the_reading_of_an_empty_morning_is_given_once_and_only_when_needed():
    assert _line(_public(4), "- 夜の結果: ").endswith(f"3日目夜 死体: アカリ（{_QUIET}）")
    # The first live game's night 1 left two corpses and no empty morning.
    two_corpses = s1.public_part(s1.memo(g1.day_two_morning(), g1.YUI))
    assert "- 夜の結果: 1日目夜 死体: ソウタ、ケント\n" in two_corpses
    # Day 1: no night has been resolved yet.
    assert "- 夜の結果: " not in _public(1)


def test_the_rule_line_says_the_wolves_bite_a_non_wolf_every_night():
    rule = _line(_public(1), "- 死体のルール: ")

    assert "人狼は1日目夜から毎晩、人狼以外の生存者を必ず1人襲撃する（襲撃なしの夜はない）" in rule
    assert "襲撃（1夜1人まで）" not in rule

