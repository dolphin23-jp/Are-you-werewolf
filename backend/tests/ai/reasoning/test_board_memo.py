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
from app.engine.state import VoteRecord
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


def test_published_verdicts_are_grouped_by_claimant_with_their_night():
    public = _public_part(_memo(_board(), "p3"))

    assert "Player4(占いCO): 0日目夜 Player1=黒" in public
    assert "Player5(占いCO): 0日目夜 Player10=白" in public


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
