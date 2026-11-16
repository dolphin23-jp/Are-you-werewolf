"""The board memo: what a v3 seat is handed instead of a brief.

Three things have to hold or the memo is worse than the brief it replaces:
it speaks in names, it shows the table only what the table can see, and it
shows the seat exactly what the seat alone knows -- no more.
"""

from __future__ import annotations

import re

from app.ai.reasoning.board_memo import (
    MEMO_HEADING,
    PRIVATE_HEADING,
    namify,
    render_board_memo,
    render_roster,
)
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.perspectives import CommonPublicPerspective, PlayerPrivatePerspective
from app.ai.reasoning.solver import build_solver
from app.engine.roles import RoleName
from app.engine.speech_events import SpeechEventType
from app.engine.state import GameState, VoteRecord
from tests.ai.reasoning import game1_boards as g1
from tests.ai.reasoning.solver import boards

_ID_TOKEN = re.compile(r"(?<![0-9A-Za-z])p\d+(?![0-9])")


def _board():  # type: ignore[no-untyped-def]
    state = boards.deal(
        {
            "p1": RoleName.WEREWOLF,
            "p2": RoleName.WEREWOLF,
            "p4": RoleName.SEER,
            "p5": RoleName.MADMAN,
            "p6": RoleName.FOX,
            "p7": RoleName.FREEMASON,
            "p8": RoleName.FREEMASON,
            "p9": RoleName.MEDIUM,
        },
        day=2,
    )
    boards.kill_first_victim(state, "p16")
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.claim(state, "p5", RoleName.SEER, day=1)
    boards.claim(state, "p7", RoleName.FREEMASON, day=1)
    boards.claim(state, "p8", RoleName.FREEMASON, day=1)
    boards.claim(state, "p9", RoleName.MEDIUM, day=1)
    boards.divine(state, "p4", "p1", night=0)
    boards.verdict(state, "p4", "seer", "p1", True, day=1)
    boards.verdict(state, "p5", "seer", "p10", False, day=1)
    boards.divine(state, "p4", "p11", night=1)  # held back, not published
    state.vote_records.extend(
        VoteRecord(voter_id=voter, target_id="p1", day=1, round=1)
        for voter in ("p3", "p4", "p7", "p8", "p9")
    )
    state.vote_records.append(VoteRecord(voter_id="p1", target_id="p4", day=1, round=1))
    boards.execute(state, "p1", day=1)
    boards.attack(state, "p2", "p12", night=1, succeeded=True)
    boards.die_by_attack(state, "p12", night=1)
    return state


def _memo(state, player_id: str) -> str:  # type: ignore[no-untyped-def]
    observations = boards.observe(state)
    return render_board_memo(
        state,
        player_id,
        observations=observations,
        public_solver=build_solver(observations, CommonPublicPerspective()),
        seat_solver=build_solver(observations, PlayerPrivatePerspective(player_id)),
    )


def _public_part(memo: str) -> str:
    """Everything above the private section, minus the header naming the seat."""
    head, _, body = memo.split(PRIVATE_HEADING)[0].partition("\n")
    assert head.startswith(MEMO_HEADING)
    return body


def _private_part(memo: str) -> str:
    return memo.split(PRIVATE_HEADING)[1]


def test_the_memo_speaks_in_names_and_never_in_ids():
    memo = _memo(_board(), "p3")

    assert memo.startswith(MEMO_HEADING)
    assert "Player4" in memo
    assert not _ID_TOKEN.search(memo), memo


def test_the_public_part_is_identical_for_every_seat():
    state = _board()

    assert _public_part(_memo(state, "p3")) == _public_part(_memo(state, "p4"))
    assert _public_part(_memo(state, "p3")) == _public_part(_memo(state, "p2"))


def test_a_villager_is_told_only_its_own_card():
    private = _private_part(_memo(_board(), "p3"))

    assert "村人" in private
    assert "Player11" not in private  # the seer's unpublished look
    assert "仲間" not in private
    assert "占い結果" not in private


def test_the_seer_sees_its_results_and_which_are_still_unpublished():
    private = _private_part(_memo(_board(), "p4"))

    assert "占い師" in private
    assert "0日目夜: Player1=黒(人狼)（公開済み）" in private
    assert "1日目夜: Player11=白(人狼ではない)（未公開）" in private


def test_a_wolf_sees_its_partners_and_the_attack_history():
    private = _private_part(_memo(_board(), "p2"))

    assert "仲間の人狼: Player1(死亡)" in private
    assert "襲撃履歴: 1日目夜 Player12" in private
    # A wolf knows the whole roster; a list of "X is not a wolf" is noise.
    assert "論理的に確定" not in private


def test_two_seer_claims_are_a_contest_and_two_freemason_claims_are_not():
    public = _public_part(_memo(_board(), "p3"))

    assert "占い=Player4、Player5（占いは1人なので少なくとも1人は偽）" in public
    assert "共有=Player7、Player8" in public
    assert "共有は2人" not in public
    assert "少なくとも1人は偽）" in public  # only once, for the seers
    assert public.count("少なくとも") == 1


def test_a_named_partner_is_pending_until_the_pair_confirms_each_other():
    state = _board()
    boards._event(state, 1, "p7", SpeechEventType.PARTNER_CLAIM, target_id="p8")

    assert "Player7は相方をPlayer8と名指ししたが本人の確認待ち" in _public_part(_memo(state, "p3"))

    boards._event(state, 1, "p8", SpeechEventType.PARTNER_CLAIM, target_id="p7")

    assert "Player7とPlayer8が互いに相方と確認済み" in _public_part(_memo(state, "p3"))


def test_seer_verdicts_are_tabled_night_by_night_with_the_day_they_were_published():
    public = _public_part(_memo(_board(), "p3"))

    assert (
        "0日目夜（初日犠牲者: Player16）: Player4→Player1=黒[1日目公開] / "
        "Player5→Player10=白[1日目公開]"
    ) in public
    # Nobody has published night 1 yet, and the table says so rather than
    # leaving a gap the reader has to notice.
    assert "1日目夜（死体: Player12）: Player4→未公開 / Player5→未公開" in public


def test_votes_are_tabled_by_target_with_counts():
    public = _public_part(_memo(_board(), "p3"))

    expected = "1日目: Player1←Player3、Player4、Player7、Player8、Player9(5) / Player4←Player1(1)"
    assert expected in public


def test_public_logic_states_the_first_victim_and_a_double_corpse_night():
    state = _board()
    boards.divine(state, "p4", "p6", night=2)
    boards.die_by_curse(state, "p6", night=2)
    boards.attack(state, "p2", "p13", night=2, succeeded=True)
    boards.die_by_attack(state, "p13", night=2)
    state.day = 3

    public = _public_part(_memo(state, "p3"))

    assert "初日犠牲者のPlayer16は人狼でも妖狐でもない" in public
    assert "2日目夜は死体が2つ（Player6、Player13）" in public
    assert "1人は占われて死んだ妖狐" in public


def test_timeline_conflicts_are_rendered_with_names():
    state = _board()
    # A night-1 verdict about a seat executed on day 1, i.e. before that night.
    boards.verdict(state, "p5", "seer", "p1", False, day=2)

    public = _public_part(_memo(state, "p3"))

    assert "時系列が合わない" in public
    assert "Player1は1日目に死亡しており" in public
    assert not _ID_TOKEN.search(public)


def test_namify_rewrites_ids_without_touching_lookalikes():
    ledger = PublicFactLedger(_board())

    rewritten = namify("p1はp12へ投票した。p99は不明", ledger)

    assert rewritten == "Player1はPlayer12へ投票した。p99は不明"


def test_the_roster_is_the_one_place_ids_live():
    roster = render_roster(PublicFactLedger(_board()), "p3")

    assert roster.startswith("【名簿(JSONのplayer_id用)】あなた=Player3(p3)")
    assert "Player4=p4" in roster


# -- composition and rules: what the model must not have to count --


def test_the_memo_states_the_composition_and_the_rules_this_engine_plays_by():
    public = _public_part(_memo(_board(), "p3"))

    assert "- 配役(17人): 人狼3・狂人1・妖狐1・占い師1・霊媒師1・狩人1・共有者2・村人7" in public
    # Each rule below is checked against app/engine (night_resolver, game,
    # roles); a rule the engine does not play is worse than no rule at all.
    for rule in (
        "狂人・妖狐は白",
        "占い先が妖狐ならその夜に呪殺（0日目夜も）",
        "妖狐は襲撃では死なない",
        "1夜に2つなら片方は呪殺された妖狐",
        "0日目夜は襲撃がなく、初日犠牲者以外の死体は呪殺",
        "初日犠牲者は人狼でも妖狐でもない",
        "霊媒が知るのは処刑された人の結果だけ",
        "同じ人の連続護衛も可",
    ):
        assert rule in public, rule


def test_the_composition_is_read_from_the_role_table():
    public = _public_part(_memo(_board(), "p3"))
    line = next(line for line in public.splitlines() if line.startswith("- 配役"))

    counts = [int(match) for match in re.findall(r"(\d+)", line.split(":", 1)[1])]
    assert sum(counts) == 17
    assert len(counts) == len(RoleName)


# -- the first live game's mornings --


def _g1_memo(state: GameState, player_id: str) -> str:
    return _memo(state, player_id)


def test_day_two_says_the_fox_is_already_dead():
    public = _public_part(_g1_memo(g1.day_two_morning(), g1.YUI))

    assert "妖狐(1人)は死亡済み。生存者に妖狐はいない（ソウタかケントのどちらか）" in public


def test_day_two_says_daiki_cannot_be_the_real_seer_and_why():
    public = _public_part(_g1_memo(g1.day_two_morning(), g1.YUI))

    line = next(line for line in public.splitlines() if "ダイキは真の占い師ではありえない" in line)
    assert "1日目夜は死体2つ（ソウタ、ケント）" in line
    assert "シオン" in line
    # Only hard impossibilities: the other two claimants are not called fake.
    assert "アオイは真の占い師ではありえない" not in public
    assert "ケントは真の占い師ではありえない" not in public


def test_day_two_tables_each_claimants_nights_and_marks_what_was_never_published():
    public = _public_part(_g1_memo(g1.day_two_morning(), g1.YUI))

    assert (
        "0日目夜（初日犠牲者: ノゾミ）: ケント→リク=白[1日目公開] / "
        "ダイキ→ソウタ=白[1日目公開] / アオイ→ユイ=白[1日目公開]"
    ) in public
    assert (
        "1日目夜（死体: ソウタ、ケント）: ケント→未公開(1日目夜に死亡) / "
        "ダイキ→シオン=白[2日目公開] / アオイ→ソウタ=白[2日目公開]"
    ) in public
    assert "アカリ(霊媒CO): 1日目処刑 ハルト=黒[2日目公開]" in public


def test_day_four_keeps_the_dead_and_drops_nights_a_claimant_could_not_act():
    public = _public_part(_g1_memo(g1.day_four_morning(), g1.YUI))

    # ケント died on night 1 and ダイキ was hanged on day 3: no cell for nights
    # they could not divine. アオイ skipped night 2, and the table shows it.
    assert "2日目夜（死体なし）: ダイキ→カイト=白[3日目公開] / アオイ→未公開" in public
    assert "3日目夜（死体なし）: アオイ→ツムギ=黒[4日目公開]" in public
    assert "ケント→未公開(1日目夜に死亡)" in public
    assert "妖狐(1人)は死亡済み。生存者に妖狐はいない" in public
    assert "ダイキは真の占い師ではありえない" in public


def test_a_verdict_said_again_stays_on_the_night_and_day_it_was_first_published():
    state = g1.day_two_morning()
    boards.execute(state, g1.REN, day=2)
    boards.verdict(state, g1.DAIKI, "seer", g1.KAITO, False, day=3, referenced_day=2)
    boards.verdict(state, g1.DAIKI, "seer", g1.SHION, False, day=3)  # mentioned again
    state.day = 3

    public = _public_part(_g1_memo(state, g1.YUI))

    assert "ダイキ→シオン=白[2日目公開]" in public
    assert "2日目夜（死体なし）: ダイキ→カイト=白[3日目公開] / アオイ→未公開" in public
    assert "ダイキは真の占い師ではありえない" in public


def test_the_first_games_memo_is_names_only_and_the_same_for_every_seat():
    state = g1.day_two_morning()
    memos = {seat: _g1_memo(state, seat) for seat in (g1.YUI, g1.DAIKI, g1.AOI, g1.MIO, g1.SOTA)}

    publics = {_public_part(memo) for memo in memos.values()}
    assert len(publics) == 1
    for memo in memos.values():
        assert not _ID_TOKEN.search(memo), memo


def test_a_night_zero_curse_names_the_dead_fox():
    state = boards.deal({"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX})
    boards.kill_first_victim(state, "p16")
    boards.divine(state, "p4", "p6", night=0)
    boards.die_by_curse(state, "p6", night=0)
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p6", False, day=1, referenced_day=0)

    public = _public_part(_memo(state, "p3"))

    assert "0日目夜は襲撃がないので、初日犠牲者以外の死体のPlayer6は呪殺された妖狐" in public
    assert "妖狐(1人)は死亡済み。生存者に妖狐はいない（Player6）" in public
    row = "0日目夜（初日犠牲者: Player16 / 死体: Player6）: Player4→Player6=白[1日目公開]"
    assert row in public
    # Already said with its reason; the generic "dead and settled" line would
    # only repeat it.
    assert "Player6(死亡)は妖狐で確定" not in public


def test_corpses_are_listed_in_seat_order_never_in_the_order_they_died():
    """The engine records the curse before the bite. Listing corpses in that
    order would tell every seat which one was the fox."""
    state = _board()
    boards.die_by_curse(state, "p13", night=2)
    boards.die_by_attack(state, "p6", night=2)
    state.day = 3

    public = _public_part(_memo(state, "p3"))

    assert "2日目夜は死体が2つ（Player6、Player13）" in public
    assert "2日目夜（死体: Player6、Player13）" in public


def test_a_contradictory_board_settles_nothing_rather_than_everything():
    state = _board()
    # Two night-0 corpses: each would have to be the one fox. No world fits,
    # and a solver with no worlds must not be read as "nobody can be anything".
    boards.die_by_curse(state, "p6", night=0)
    boards.die_by_curse(state, "p13", night=0)

    facts = _public_part(_memo(state, "p3")).replace("公開情報だけで確定すること", "")

    assert "で確定" not in facts
    assert "と確定" not in facts
    assert "死亡済み" not in facts
    assert "真の占い師ではありえない" not in facts


def test_the_public_part_reads_nothing_but_public_facts():
    """Change only what no seat may see -- the deal of two unclaimed seats, the
    guards, an unpublished divination -- and the public part must not move."""
    plain = g1.day_four_morning()
    altered = g1.day_four_morning()
    altered.players[g1.MIO].role, altered.players[g1.RIKU].role = (
        altered.players[g1.RIKU].role,
        altered.players[g1.MIO].role,
    )
    altered.guard_records.clear()
    boards.divine(altered, g1.AOI, g1.SAKURA, night=2)

    assert _public_part(_g1_memo(plain, g1.YUI)) == _public_part(_g1_memo(altered, g1.YUI))


# -- the seat's own tail adds only what the seat alone can conclude --


def test_a_wolf_learns_which_corpse_was_the_fox_from_its_own_bite():
    private = _private_part(_g1_memo(g1.day_two_morning(), g1.KAITO))

    assert "ソウタ(死亡)は妖狐で確定" in private
    # Its own team is never news to it.
    assert "人狼ではない" not in private


def test_the_private_tail_does_not_repeat_the_public_logic():
    memo = _g1_memo(g1.day_two_morning(), g1.KAITO)

    assert "妖狐(1人)は死亡済み" in _public_part(memo)
    assert "死亡済み" not in _private_part(memo)
    assert "真の占い師ではありえない" not in _private_part(memo)


def test_a_villager_reads_nothing_private_from_the_curse():
    private = _private_part(_g1_memo(g1.day_two_morning(), g1.YUI))

    assert "妖狐で確定" not in private
