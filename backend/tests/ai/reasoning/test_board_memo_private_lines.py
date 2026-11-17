"""What only one seat can conclude, said in that seat's section and nowhere
else (seed1-review-causes.md, §2.4 and 2-C1).

The design contract holds throughout: a hard conclusion is printed as one; a
conclusion that rests on a supposition names the supposition in its first
words; and nothing derived from a seat's private knowledge reaches the public
section or another seat.
"""

from __future__ import annotations

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
