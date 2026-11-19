"""The owner's review of run #16: the curse response, a claimant shown fake,
the freemason contest, and who led an execution.

Pinned at the real moments of run #16 (`run16_boards`): day 2 (two corpses,
the two fake seers publishing living night-1 targets) and days 4-6 (the
freemason pair with ホノカ bitten, the executions to look back on).

* X12-1/X12-3 -- "two corpses = one bite + the cursed fox" is a record every
  seat reads at once; "a claimant whose night-1 target is neither corpse
  cannot be the seer" is derived and reaches each seat by skill; a fake seer
  about to publish is shown what a real seer's night-1 account must say, by
  the same skill; the audit's reason is written in names.
* X25 -- a claimant this seat has been shown cannot be the seer settles
  nothing: its targets are grey again, in this seat's own grey line.
* X24-2 -- the confirmed pair stays on the record beside a third claim, and
  what each claim being real makes of the others is derived.
* X23-3 -- each execution records where the day's push began, the
  designation and the count.
"""

from __future__ import annotations

import re

from app.ai.coordinator import AICoordinator
from app.ai.provider.mock import MockProvider
from app.ai.reasoning.board_memo import PRIVATE_HEADING, render_board_memo
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import CommonPublicPerspective, PlayerPrivatePerspective
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.reasoning.skill import DerivationClock, Skill
from app.ai.reasoning.solver import build_solver, has_role, not_role
from app.ai.reasoning.solver.rules import plain_rule_modules
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, ChatMessage, GameState
from tests.ai.reasoning import run16_boards as r16
from tests.ai.reasoning.solver import boards

_SEER = RoleName.SEER
_MASON = RoleName.FREEMASON
_ID = re.compile(r"(?<![0-9A-Za-z])p\d+(?![0-9])")

_TWO_CORPSES = (
    "  - 1日目夜は死体が2つ（ソウタ、ケント）。襲撃は一晩1件、呪殺も1件なので、"
    "このうち1人は占われて死んだ妖狐"
)
_AOI_OUT = (
    "  - アオイは真の占い師ではありえない（1日目夜は死体2つ（ソウタ、ケント）で片方は呪殺なので、"
    "真の占い師のその夜の占い先は死体のどちらか。アオイが公開したその夜の占い先ユイは死んでいない）"
)
_GENUINE_NIGHT1 = (
    "- 占いCOの1日目夜の結果について: 1日目夜は死体が2つ（ソウタ、ケント）で"
    "片方は呪殺された妖狐なので、真の占い師の1日目夜の占い先はソウタかケントのどちらかで、結果は白。"
    "死体でない席を1日目夜の占い先と公開した占いCOは、真の占い師ではありえない"
)


def _public(text: str) -> str:
    return text.split(PRIVATE_HEADING)[0]


def _private(text: str) -> str:
    return text.split(PRIVATE_HEADING)[1]


def _line(text: str, head: str) -> str:
    (line,) = [row for row in text.splitlines() if row.startswith(head)]
    return line


def _say(state: GameState, author: str, content: str) -> None:
    state.chat_log.append(
        ChatMessage(
            message_id=f"t{len(state.chat_log)}",
            author_id=author,
            content=content,
            channel=ChatChannel.PUBLIC,
            day=state.day,
        )
    )


# -- X12: the curse, read by every seat and by the fakes --


def test_two_corpses_are_one_bite_and_the_cursed_fox_for_every_seat_at_once():
    """A record: a novice with a fresh clock reads it the morning it happens,
    whether or not カイト's false freemason CO is on the board."""
    for false_co in (True, False):
        state = r16.board(2, false_co=false_co)
        for seat in (r16.AOI, r16.DAIKI, r16.YUI):
            memo = r16.memo(state, seat, skill=Skill.NOVICE, clock=DerivationClock())
            assert _TWO_CORPSES in _public(memo).splitlines(), (seat, false_co)
            assert "妖狐(1人)は死亡済み。生存者に妖狐はいない（ソウタかケントのどちらか）" in memo


def test_a_living_night1_target_is_a_derived_refutation_shown_by_skill():
    """After m36 (アオイ: ユイ white): an expert reads it at once, the average
    seat the next day, the novice once the table has said it (m56)."""
    clock = DerivationClock()
    after_m36 = r16.board(2, "m36")
    assert _AOI_OUT in r16.memo(after_m36, r16.RIKU, skill=Skill.EXPERT, clock=clock)
    for skill in (Skill.AVERAGE, Skill.NOVICE):
        assert "アオイは真の占い師ではありえない" not in r16.memo(
            after_m36, r16.DAIKI, skill=skill, clock=clock
        ), skill

    # m38 (「真はいないと見ていいのか」) states nothing by the rule; m56 does
    # (「アオイさんを吊りたいです…。…真の占いではありえない」).
    assert "アオイは真の占い師ではありえない" not in r16.memo(
        r16.board(2, "m38"), r16.DAIKI, skill=Skill.NOVICE, clock=clock
    )
    assert _AOI_OUT in r16.memo(r16.board(2, "m56"), r16.DAIKI, skill=Skill.NOVICE, clock=clock)

    next_day = r16.board(2, "m36")
    next_day.day = 3
    assert _AOI_OUT in r16.memo(next_day, r16.DAIKI, skill=Skill.AVERAGE, clock=clock)


def test_a_fake_seer_is_shown_what_a_real_night1_account_must_say_by_skill():
    """アオイ, morning of day 2, before choosing her night-1 result. In the
    game she was 普通 and was not shown it; nothing forces the choice."""
    state = r16.board(2)

    expert = r16.memo(state, r16.AOI, skill=Skill.EXPERT, clock=DerivationClock())
    assert _GENUINE_NIGHT1 in _private(expert).splitlines()
    for skill in (Skill.AVERAGE, Skill.NOVICE):
        memo = r16.memo(state, r16.AOI, skill=skill, clock=DerivationClock())
        assert "占いCOの1日目夜の結果について" not in memo, skill
    # Said at the table before she speaks, it reaches every skill.
    clock = DerivationClock()
    r16.memo(state, r16.AOI, skill=Skill.EXPERT, clock=clock)  # stamped this morning
    _say(state, r16.AKARI, "真の占い師なら昨夜の占い先はソウタかケントのはず、それ以外はありえない")
    assert _GENUINE_NIGHT1 in r16.memo(state, r16.AOI, skill=Skill.NOVICE, clock=clock)


def test_the_planned_fake_seer_reads_it_before_its_claim_and_not_after():
    """ダイキ was the wolf told to fake the seer; he claimed at m37."""
    before = r16.board(2, "m36")
    planned = r16.memo(before, r16.DAIKI, planned_claim=_SEER)
    assert _GENUINE_NIGHT1 in _private(planned).splitlines()
    # Without the plan he is a latent wolf like ハルト: nothing to publish.
    assert "占いCOの1日目夜の結果について" not in r16.memo(before, r16.DAIKI)
    assert "占いCOの1日目夜の結果について" not in r16.memo(before, r16.HARUTO)
    # Once his night-1 account is out, the window has closed.
    after = r16.board(2, "m37")
    assert "占いCOの1日目夜の結果について" not in r16.memo(after, r16.DAIKI, planned_claim=_SEER)


def test_a_night0_curse_sets_the_real_seers_night0_target():
    """Hand-dealt: p0 the seer curses the fox p1 on night 0; the madman p5
    stands on a seer claim with nothing published yet."""
    state = boards.deal(
        {"p0": _SEER, "p1": RoleName.FOX, "p2": RoleName.WEREWOLF, "p5": RoleName.MADMAN}
    )
    boards.kill_first_victim(state, "p16")
    boards.divine(state, "p0", "p1", night=0)
    boards.die_by_curse(state, "p1", night=0)
    boards.claim(state, "p5", _SEER, day=1)
    observations = ObservationSet.from_state(state)

    def memo(seat: str) -> str:
        return render_board_memo(
            state,
            seat,
            observations=observations,
            public_solver=build_solver(observations, CommonPublicPerspective()),
            seat_solver=build_solver(observations, PlayerPrivatePerspective(seat)),
        )

    assert (
        "- 占いCOの0日目夜の結果について: 0日目夜は襲撃がなく、初日犠牲者以外の死体の"
        "Player1は呪殺された妖狐なので、真の占い師の0日目夜の占い先はPlayer1で、結果は白。"
        "ほかの席を0日目夜の占い先と公開した占いCOは、真の占い師ではありえない"
    ) in _private(memo("p5")).splitlines()
    assert "占いCOの0日目夜の結果について" not in memo("p0")  # the seer knows its own


def test_nobody_else_is_shown_the_fake_seer_line():
    state = r16.board(2)
    for seat in (r16.YUI, r16.MAKOTO, r16.MIO, r16.AKARI, r16.HARUTO):
        assert "占いCOの1日目夜の結果について" not in r16.memo(state, seat), seat


def test_the_runtime_passes_the_planned_claim_and_the_coordinator_sets_it():
    state = r16.board(2, "m36")
    seats = [pid for pid, player in state.players.items() if player.alive]
    runtime = ReasoningRuntime(state, seats, seed=1)
    runtime.skills[r16.DAIKI] = Skill.EXPERT
    assert "占いCOの1日目夜の結果について" not in runtime.board_memo(state, r16.DAIKI)
    runtime.planned_claims[r16.DAIKI] = _SEER
    assert _GENUINE_NIGHT1 in runtime.board_memo(state, r16.DAIKI)

    fresh = ReasoningRuntime(state, seats, seed=1)
    AICoordinator(state, seats, MockProvider(), seed=1, reasoning=fresh, model_decides=True)
    wolves = {pid for pid, player in state.players.items() if player.role is RoleName.WEREWOLF}
    assert fresh.planned_claims
    assert set(fresh.planned_claims) <= wolves | {r16.AOI}


def test_the_ally_vote_audit_reason_is_written_in_names():
    """Run #16's audit read 「1日目に占ったのはp14なのでp4は呪殺されておらず…」."""
    state = r16.board(3, "vote")
    seats = [pid for pid, player in state.players.items() if player.alive]
    runtime = ReasoningRuntime(state, seats, seed=1)

    marks = runtime.public_marks(state, r16.DAIKI)

    assert marks
    for mark in marks:
        assert not _ID.search(mark.explanation), mark.explanation
    impossible = [mark for mark in marks if mark.evidence_id.startswith("impossible_claim:")]
    assert "シオン" in impossible[0].explanation
    assert "ソウタ、ケントのいずれかが妖狐です" in impossible[0].explanation
    black = [mark for mark in runtime.public_marks(state, r16.SAKURA)]
    assert black and black[0].explanation == "ダイキの占い結果でサクラは黒"


def test_no_seat_id_reaches_any_memo_line():
    for day in range(2, 7):
        state = r16.board(day, "vote")
        for seat in (r16.MAKOTO, r16.HARUTO, r16.MIO):
            if not state.players[seat].alive:
                continue
            memo = r16.memo(state, seat)
            assert not _ID.search(memo), (day, seat, _ID.findall(memo))


# -- X25: the results of a claimant shown fake settle nothing --


def test_a_claimant_shown_fake_puts_its_targets_back_in_the_grey():
    """Day 4: アオイ's whites (ハルト, a wolf, and ユイ) and ダイキ's white on
    シオン and black on サクラ are grey again for a seat shown both
    refutations; ケント's white on ミオ still counts."""
    state = r16.board(4)
    expert = _line(r16.memo(state, r16.TSUMUGI, skill=Skill.EXPERT), "- グレー")

    assert expert == (
        "- グレー(生存・CO無し・真でありうる占いCOの判定も無し): ハルト、ユイ、サクラ、リク、"
        "自分、シオン / 判定を受けた非CO者: ミオ(ケントの白) / 真の占い師ではありえない"
        "占いCOの判定は何も決めない（判定された席もグレー）: ダイキ→シオン=白、サクラ=黒、"
        "アオイ→ハルト=白、ユイ=白"
    )


def test_a_seat_not_yet_shown_the_refutation_keeps_the_record_grey():
    state = r16.board(4)
    novice = _line(
        r16.memo(state, r16.TSUMUGI, skill=Skill.NOVICE, clock=DerivationClock()), "- グレー"
    )

    assert novice == (
        "- グレー(生存・CO無し・どの占いCOの判定も無し): リク、自分 / 判定を受けた非CO者: "
        "ハルト(アオイの白)、ユイ(アオイの白)、ミオ(ケントの白)、サクラ(ダイキの黒)、"
        "シオン(ダイキの白)"
    )


def test_the_grey_follows_what_the_seat_has_been_shown():
    """Day 2 after m36, an expert: アオイ's ユイ white is out, so ユイ and
    ハルト are grey; before m36 she had published nothing for night 1."""
    after = _line(r16.memo(r16.board(2, "m36"), r16.RIKU), "- グレー")
    before = _line(r16.memo(r16.board(2), r16.RIKU), "- グレー")

    assert after.startswith("- グレー(生存・CO無し・真でありうる占いCOの判定も無し): ハルト、ユイ")
    assert "アオイ→ハルト=白、ユイ=白" in after
    assert "判定を受けた非CO者: ハルト(アオイの白)" in before


# -- X24-2: the freemason pair beside a third claim --


def test_the_confirmed_pair_stays_on_the_record_beside_a_third_claim():
    """From m9 on, the count had replaced the pair on the CO line (X24-2)."""
    for day, honoka in ((2, "ホノカ"), (4, "ホノカ(死亡)")):
        line = _line(r16.memo(r16.board(day), r16.YUI), "- CO:")
        assert line.endswith(
            f"共有=マコト、{honoka}、カイト（共有は2人なので少なくとも1人は偽。"
            "マコトとホノカが互いに相方と確認済み。カイトは相方の名指しなし）"
        ), (day, line)


def test_without_the_false_co_the_pair_is_the_whole_record():
    memo = r16.memo(r16.board(4, false_co=False), r16.YUI)

    assert _line(memo, "- CO:").endswith(
        "共有=マコト、ホノカ(死亡)（マコトとホノカが互いに相方と確認済み。"
        "共有は2人とも名前が出ている）"
    )
    assert "が真の共有なら" not in memo


def test_a_third_claim_real_makes_the_bitten_pair_fake_and_not_a_wolf():
    """Day 4 morning: ホノカ bitten on night 3; the fox died on night 1."""
    state = r16.board(4)
    expert = _public(r16.memo(state, r16.YUI, skill=Skill.EXPERT))

    assert (
        "  - マコトとホノカ(死亡)が真の共有なら（互いに相方と確認済み）: カイトは真の共有ではない"
    ) in expert.splitlines()
    assert (
        "  - カイトが真の共有なら: マコトとホノカ(死亡)はどちらも真の共有ではない"
        "（マコトとホノカは互いに相方と名指ししているので、片方が真ならもう片方も真になる）。"
        "ホノカ(死亡)は人狼でも妖狐でもないので、狂人か村側の騙り"
    ) in expert.splitlines()
    # The record half is there for every seat at once.
    novice = _public(r16.memo(state, r16.YUI, skill=Skill.NOVICE, clock=DerivationClock()))
    assert "が真の共有なら" not in novice
    assert "夜に死んだアカリ、ソウタ、ホノカ、ケントは人狼ではない" in novice


def test_the_partner_rule_is_derived_and_ties_the_pair_together():
    observations = ObservationSet.from_state(r16.board(4))
    public = build_solver(observations, CommonPublicPerspective())
    plain = build_solver(observations, CommonPublicPerspective(), rules=plain_rule_modules())
    kaito = has_role(r16.KAITO, _MASON)

    for seat in (r16.MAKOTO, r16.HONOKA):
        assert public.implies(kaito, not_role(seat, _MASON))
        assert not plain.implies(kaito, not_role(seat, _MASON))
    assert public.implies(has_role(r16.HONOKA, _MASON), has_role(r16.MAKOTO, _MASON))
    # Nothing is settled outright: the table still has a contest.
    assert public.is_possible(kaito)
    assert public.is_possible(has_role(r16.MAKOTO, _MASON))


def test_the_real_freemason_reads_the_third_claim_as_fake_from_its_card():
    private = _private(r16.memo(r16.board(4), r16.MAKOTO))

    assert "- 共有の相方: ホノカ(死亡)（共有者は2人なので、ほかの共有COのカイトは偽）" in (
        private.splitlines()
    )
    # Its own card, not a conclusion to keep off the table.
    assert "カイトは真の共有者ではありえない" not in private


# -- X23-3: who led each execution --

_EXECUTIONS_DAY6 = [
    "- 処刑の経緯（その日の発言から語で拾った最初の疑い・最初の吊り先と、"
    "共有COの最後の指定。投票者は投票の行）:",
    "  - 1日目 レン（10票/16票）: 最初に疑い ハルト m20、"
    "吊り先として名前を出した発言なし、指定なし",
    "  - 2日目 アオイ（12票/13票）: 最初に吊り先 マコト m56、指定 マコト m58",
    "  - 3日目 ダイキ（9票/11票）: 最初に吊り先 サクラ m66、指定 マコト m79",
    "  - 4日目 カイト（7票/9票）: 最初に吊り先 マコト m87、指定 マコト m98",
    "  - 5日目 リク（7票/8票）: 最初に疑い ユイ m107、最初に吊り先 マコト m110、指定 マコト m114",
]


def test_each_execution_records_where_the_push_began():
    """レン, read white on day 2, was pushed first by ハルト (a wolf) with
    「今のところ気になる」 and named for a hang by nobody."""
    lines = r16.memo(r16.board(6, "vote"), r16.MIO).splitlines()
    start = lines.index(_EXECUTIONS_DAY6[0])

    assert lines[start : start + len(_EXECUTIONS_DAY6)] == _EXECUTIONS_DAY6


def test_a_hang_word_about_another_seat_does_not_name_this_one():
    """m100: 「リクさん、昨日カイトさんに入れた理由を、ダイキさん吊りの話とは別に…」
    puts 吊 on ダイキ; リク's first hang naming is マコト's m110."""
    lines = r16.memo(r16.board(6, "vote"), r16.MIO).splitlines()

    assert "シオン m100" not in "\n".join(lines)


def test_the_execution_record_is_the_same_for_every_seat_and_skill():
    state = r16.board(5)
    seen = set()
    for seat, skill in ((r16.MIO, Skill.EXPERT), (r16.TSUMUGI, Skill.NOVICE)):
        memo = r16.memo(state, seat, skill=skill, clock=DerivationClock())
        start = memo.index("- 処刑の経緯")
        seen.add(memo[start : memo.index("- 夜の結果")])
    assert len(seen) == 1


# -- day 5, as one board --


def test_day5_board_keeps_every_fix_together():
    """The day-5 morning memo of the hunter (ミオ, 普通 in the game)."""
    memo = r16.memo(r16.board(5), r16.MIO, skill=Skill.EXPERT)
    public = _public(memo)

    assert "マコトとホノカが互いに相方と確認済み。カイトは相方の名指しなし" in public
    assert "  - 4日目 カイト（7票/9票）: 最初に吊り先 マコト m87、指定 マコト m98" in public
    assert _line(public, "- グレー").startswith(
        "- グレー(生存・CO無し・真でありうる占いCOの判定も無し): "
        "ハルト、ユイ、サクラ、リク、ツムギ、シオン"
    )
    assert (
        "  - カイト(死亡)が真の共有なら: マコトとホノカ(死亡)はどちらも真の共有ではない"
    ) in public
    assert not _ID.search(memo)
