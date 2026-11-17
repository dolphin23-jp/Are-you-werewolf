"""What only one seat can conclude, said in that seat's section and nowhere
else (seed1-review-causes.md, §2.4 and 2-C1).

The design contract holds throughout: a hard conclusion is printed as one; a
conclusion that rests on a supposition names the supposition in its first
words; and nothing derived from a seat's private knowledge reaches the public
section or another seat.
"""

from __future__ import annotations

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import PlayerPrivatePerspective
from app.ai.reasoning.solver import build_solver, has_role
from app.engine.roles import RoleName
from tests.ai.reasoning import run13_seed1_boards as s1


def _memos(day: int, at: str) -> dict[str, str]:
    state = s1.board(day, at)
    return {seat: s1.memo(state, seat) for seat in state.alive_ids()}


def _others_see(memos: dict[str, str], seat: str, text: str) -> list[str]:
    """Seats other than `seat` whose memo contains `text`, plus 'public' when
    the shared part does."""
    leaks = [s1.name(other) for other, memo in memos.items() if other != seat and text in memo]
    if any(text in s1.public_part(memo) for memo in memos.values()):
        leaks.append("public")
    return leaks


# -- 7. the seer: not the fox either, and what follows if no villager bluffs --


def test_the_seer_is_told_its_survivors_are_neither_wolf_nor_fox():
    """2-B3: ケント's own solver had アオイ "not the fox" (divined on night 1
    and alive) and the memo had no way to say it."""
    private = s1.private_part(s1.memo(s1.board(2, "m32"), s1.KENT))

    assert "リク、アオイは人狼でも妖狐でもない" in private
    assert "人狼ではないと確定" not in private


def test_the_seer_reads_the_fake_claims_under_a_named_supposition():
    """「ダイキは人狼か妖狐」 is true only if no village seat bluffs a role. The
    line says so in its first words, and never sits with the certainties."""
    private = s1.private_part(s1.memo(s1.board(2, "m32"), s1.KENT))

    line = next(line for line in private.splitlines() if "村側の騙り" in line)
    assert line == "  - 村側の騙りがなければ: ダイキは人狼か妖狐、アオイは狂人"
    certain = next(line for line in private.splitlines() if "論理的に確定" in line)
    assert "狂人" not in certain


def test_nothing_the_seer_alone_concluded_reaches_another_seat():
    memos = _memos(2, "m32")

    for text in (
        "ダイキは人狼か妖狐",
        "アオイは狂人",
        "リク、アオイは人狼でも妖狐でもない",
        "アオイは人狼でも妖狐でもない",
    ):
        assert _others_see(memos, s1.KENT, text) == [], text


def test_the_fox_is_never_told_who_is_not_the_fox():
    for day, at in ((2, "m32"), (3, "m45"), (5, "m75")):
        private = s1.private_part(s1.memo(s1.board(day, at), s1.SOTA))
        assert "妖狐でもない" not in private, (day, at)
        assert "妖狐ではない" not in private, (day, at)


# -- 8. the wolves: a published colour they know is false --

_FALSE_WHITE = "アオイの『カイト=白』は事実と違う（カイトは人狼）"
_TRUTHFUL = (
    "  - 真の占い師が色を偽らないなら、アオイは真の占い師ではない。"
    "潜伏の占い師もいないなら、真の占い師はケント"
)


def test_every_wolf_is_told_that_aoi_whited_a_wolf():
    """2-B4: m34 アオイ published カイト white. All three wolves knew it was
    false; their memo had no way to say so, and the existing supposition
    `HonestResults(アオイ)` made the whole board impossible instead."""
    state = s1.board(2, "m34")

    for wolf in (s1.HARUTO, s1.DAIKI):
        lines = s1.private_part(s1.memo(state, wolf)).splitlines()
        assert f"- {_FALSE_WHITE}" in lines, s1.name(wolf)
        assert _TRUTHFUL in lines, s1.name(wolf)


def test_the_false_colour_is_a_fact_and_the_true_seer_a_supposition():
    lines = s1.private_part(s1.memo(s1.board(2, "m34"), s1.HARUTO)).splitlines()

    fact = lines.index(f"- {_FALSE_WHITE}")
    head = lines.index("- あなたの視点で、仮定つきで言えること（仮定が外れれば成り立たない）:")
    assert fact < head < lines.index(_TRUTHFUL)
    certain = next((line for line in lines if "論理的に確定" in line), "")
    assert "ケント" not in certain


def test_no_other_seat_learns_what_the_wolves_know_about_aoi():
    memos = _memos(2, "m34")

    for wolf in s1.WOLVES:
        memos.pop(wolf)
    for text in ("事実と違う", "色を偽らない", "真の占い師はケント"):
        assert _others_see(memos, "", text) == [], text


# -- 9. the hunter: the empty mornings that were its own guard --

_GUARD_READING = "襲撃先は護衛先か妖狐（護衛の成否は知らされない）"


def test_the_hunter_is_told_which_empty_mornings_followed_its_guard():
    """3-B2 / 5-N2: ミオ guarded ケント on every quiet night and never used it.
    She is not told the guard held -- the engine never tells her."""
    day_three = s1.private_part(s1.memo(s1.board(3), s1.MIO)).splitlines()
    assert f"- 死体なしの夜: 1日目夜(ケント)・2日目夜(ケント)。{_GUARD_READING}" in day_three

    day_five = s1.private_part(s1.memo(s1.board(5), s1.MIO)).splitlines()
    assert (
        f"- 死体なしの夜: 1日目夜(ケント)・2日目夜(ケント)・4日目夜(ケント)。{_GUARD_READING}"
    ) in day_five


def test_the_hunter_line_does_not_clear_the_seat_it_guarded():
    """The bite may have gone to the fox each time, so ケント as a wolf stays
    possible from ミオ's seat -- unnatural, not impossible."""
    state = s1.board(5)
    solver = build_solver(ObservationSet.from_state(state), PlayerPrivatePerspective(s1.MIO))
    private = s1.private_part(s1.memo(state, s1.MIO))

    assert solver.is_possible(has_role(s1.KENT, RoleName.WEREWOLF)) is True
    assert "ケントは人狼" not in private
    assert "ケント、" not in private.split("論理的に確定")[-1].split("\n")[0]


def test_only_the_hunter_sees_its_guards():
    memos = _memos(3, "morning")

    assert _others_see(memos, s1.MIO, "死体なしの夜") == []


# -- 10. the reader is 自分 in its own memo --


def test_a_seat_reads_itself_as_jibun_everywhere_but_the_header():
    """2-C1: the memo wrote the reader in the third person (「アオイ→ソウタ=白」
    in ソウタ's own memo), and seats spoke of themselves the same way."""
    text = s1.memo(s1.board(2, "m32"), s1.SOTA)

    assert text.splitlines()[0] == "【盤面メモ】2日目 / あなた: ソウタ"
    assert "アオイ→自分=白[1日目公開]" in text
    assert "自分へ1票: ユイ(処刑)" in text
    assert "ソウタ" not in text.partition("\n")[2]
    # Everyone else still reads ソウタ by name.
    assert "アオイ→ソウタ=白[1日目公開]" in s1.memo(s1.board(2, "m32"), s1.REN)


def test_the_foxs_curse_reasoning_keeps_its_conclusion_in_the_first_person():
    """The owner's §7 good behaviour: ソウタ (the fox, alive after アオイ's
    night-0 white on him) knew アオイ could not be the real seer."""
    private = s1.private_part(s1.memo(s1.board(1, "m14"), s1.SOTA))

    assert (
        "アオイは真の占い師ではありえない（アオイが真の占い師なら、公開どおり0日目夜に自分を"
        "占っています。自分はその夜に死んでいないので、その場合自分は妖狐ではありません。）"
    ) in private


def test_the_whited_wolf_reads_the_false_colour_about_itself():
    lines = s1.private_part(s1.memo(s1.board(2, "m34"), s1.KAITO)).splitlines()

    assert "- アオイの『自分=白』は事実と違う（自分は人狼）" in lines
    assert _TRUTHFUL in lines
