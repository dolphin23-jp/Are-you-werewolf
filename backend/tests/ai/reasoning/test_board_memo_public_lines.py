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
    return s1.shared_part(s1.memo(s1.board(day, at), seat), seat)


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


# -- 5. the grey, and the CO count with its zeros --


def test_the_grey_is_listed_apart_from_the_seats_a_claimant_has_judged():
    """2-B5: on day 2 the table argued over "the grey" with two definitions
    and no list. After m37 every seer claimant had published two nights."""
    line = _line(_public(2, "m37"), "- グレー")

    assert line == (
        "- グレー(生存・CO無し・どの占いCOの判定も無し): ハルト、レン、サクラ、ツムギ / "
        "判定を受けた非CO者: ソウタ(アオイの白)、ミオ(ダイキの白)、カイト(アオイの白)、"
        "リク(ケントの白)、シオン(ダイキの白)"
    )


def test_a_seat_judged_twice_shows_both_verdicts():
    line = _line(_public(5, "m75"), "- グレー")

    assert "カイト(アオイの白・ケントの黒)" in line


def test_the_co_count_says_the_zero_and_the_name_of_the_split():
    """1-B1: m4 called it 3-1 with no medium out; the memo never said 霊媒0."""
    expected = {
        "m2": "- CO数: 占い1(生存1)・霊媒0 → 内訳の呼び方は1-0",
        "m3": "- CO数: 占い2(生存2)・霊媒0 → 内訳の呼び方は2-0",
        "m4": "- CO数: 占い3(生存3)・霊媒0 → 内訳の呼び方は3-0",
    }
    for message, line in expected.items():
        public = _public(1, message)
        assert line in public.splitlines(), message
        assert "霊媒=まだなし" in _line(public, "- CO: ")

    after_akari = _public(2, "m31")
    assert "- CO数: 占い3(生存3)・霊媒1(生存1) → 内訳の呼び方は3-1" in after_akari.splitlines()
    assert "霊媒=まだなし" not in after_akari
    late = _public(5)
    assert "- CO数: 占い3(生存1)・霊媒1(生存0) → 内訳の呼び方は3-1" in late.splitlines()


def test_before_anybody_claims_there_is_no_count_to_give():
    public = _public(1)

    assert "- CO: まだ誰もCOしていない" in public.splitlines()
    assert "CO数" not in public


# -- 6. ballots, read both ways --


def test_ballots_are_tabled_by_target_without_an_arrow_to_misread():
    """4-B2: 「ケント←ハルト、カイト、ダイキ」 was read backwards. Each target now
    says how many ballots it got and from whom; (処刑) marks the day's
    execution wherever that seat appears."""
    lines = _public(4, "m61").splitlines()
    head = "- 投票（N日目: 投票先へ票数: 投票した人。(処刑)はその日に処刑された人）:"
    rows = lines[lines.index(head) + 1 : lines.index(head) + 5]

    assert rows == [
        "  - 1日目: ユイへ15票(処刑): マコト、アカリ、ハルト、ソウタ、ミオ、レン、サクラ、カイト、"
        "リク、ツムギ、ダイキ、ホノカ、シオン、アオイ、ケント / ソウタへ1票: ユイ(処刑)",
        "  - 2日目: レンへ3票: マコト、サクラ、シオン / アオイへ3票: ソウタ、カイト、ホノカ / "
        "ホノカへ2票: ハルト、ツムギ / シオンへ2票: ミオ、レン / "
        "ケントへ2票: ダイキ、アオイ(処刑) / ハルトへ1票: リク / ツムギへ1票: アカリ / "
        "ダイキへ1票: ケント",
        "  - 2日目決選: アオイへ10票(処刑): マコト、ソウタ、ミオ、レン、カイト、リク、ツムギ、"
        "ダイキ、ホノカ、ケント / レンへ5票: アカリ、ハルト、サクラ、シオン、アオイ(処刑)",
        "  - 3日目: ダイキへ11票(処刑): マコト、アカリ、ソウタ、ミオ、レン、サクラ、リク、ツムギ、"
        "ホノカ、シオン、ケント / ケントへ3票: ハルト、カイト、ダイキ(処刑)",
    ]
    assert "←" not in "\n".join(lines)


def test_every_living_voter_has_their_own_ballot_history():
    """4-B2 / 4-B3: m61 「ダイキ吊りの時はケントに入れてない」 -- his own row
    says otherwise, in one place."""
    lines = _public(4, "m61").splitlines()
    head = "- 生存者ごとの投票先:"
    rows = lines[lines.index(head) + 1 :]

    assert "  - ハルト: 1日目ユイ / 2日目ホノカ→決選レン / 3日目ケント" in rows
    assert "  - カイト: 1日目ユイ / 2日目アオイ→決選アオイ / 3日目ケント" in rows
    voters = [row[4:].split(":")[0] for row in rows if row.startswith("  - ")]
    # The living only, in seating order; the dead keep their rows above.
    assert voters[:4] == ["マコト", "ハルト", "ソウタ", "ミオ"]
    assert "ダイキ" not in voters and "アカリ" not in voters
