"""The ballot audit: every vote is seen, and a change of mind is read fairly.

The first Claude game showed two holes in it. A seat that changed its vote
after another seat argued for the new target was filed as an unexplained
change, because the classifier only knew runoffs, deaths, corrections and new
ability results. And only seats that had spoken that day had an audit at all,
so 22 of 69 ballots -- a werewolf's vote on its own partner among them -- were
never checked. These tests pin both, including on the board from that game.
"""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from app.ai.context import CHAT_VOTE_OUTPUT_INSTRUCTION
from app.ai.coordinator import AICoordinator
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    ChatDiscussionOutput,
    DiscussionOutput,
    MorningIntentOutput,
    NightActionOutput,
    PublicResultClaim,
    SummaryOutput,
    VoteOutput,
    WolfChatOutput,
)
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import ChatChannel, ChatMessage, GameState
from app.eval.reasoning_analyzer import ReasoningTranscriptAnalyzer, VoteChangeKind
from app.eval.transcript import GameTranscript, TranscriptRecorder
from tests.ai import first_claude_game as game1
from tests.ai.reasoning.solver import boards
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


class _PlanAndBallot:
    """Names `plan` as its candidate in every turn and votes for `vote`.

    `line` is what the turn says aloud, and `claim` -- (role, target, black) --
    a CO and one seer verdict the turn publishes with it.
    """

    def __init__(self, plan: str, vote: str | None = None, reason: str = "気が変わった。") -> None:
        self.plan = plan
        self.vote = vote or plan
        self.reason = reason
        self.line = "灰の発言を比べたい。"
        self.claim: tuple[str, str, bool] | None = None

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        if response_schema in (DiscussionOutput, ChatDiscussionOutput):
            output = DiscussionOutput(public_message=self.line)
            output.reasoning_memo.execution_target = self.plan
            if self.claim is not None:
                role, target, black = self.claim
                output.public_claim_role = role
                output.public_results = [
                    PublicResultClaim(result_type="seer", target_id=target, is_werewolf=black)
                ]
            if response_schema is ChatDiscussionOutput:
                return ChatDiscussionOutput.from_discussion_output(output)
            return output
        if response_schema is VoteOutput:
            return VoteOutput(vote_target=self.vote, reason=self.reason)
        if response_schema is NightActionOutput:
            return NightActionOutput(target=self.plan)
        if response_schema is WolfChatOutput:
            return WolfChatOutput(message="様子見。")
        return SummaryOutput(summary="要約")


def _v3_table():  # type: ignore[no-untyped-def]
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    runtime = ReasoningRuntime(state, AI_IDS, seed=4)
    speaker = "p1"
    plan, other = [pid for pid in state.alive_ids() if pid not in (speaker, "p0")][:2]
    provider = _PlanAndBallot(plan)
    # The seat says its candidate aloud: a change off a memo target nobody
    # heard is not judged by the gate (see `test_a_change_off_a_memo_target_...`).
    provider.line = f"{state.players[plan].name}が一番怪しい。"
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        state, AI_IDS, provider, seed=4, reasoning=runtime, model_decides=True, recorder=recorder
    )
    return controller, state, coordinator, provider, recorder, speaker, plan, other


def _name(state: GameState, player_id: str) -> str:
    return state.players[player_id].name


def _latest_audit(recorder: TranscriptRecorder, player_id: str, day: int):  # type: ignore[no-untyped-def]
    audit = recorder.latest_decision(player_id, day)
    assert audit is not None
    return audit


# -- a change of mind after someone argued for it --


def test_a_change_after_another_seat_argued_for_the_new_target_is_a_public_argument():
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    argued = controller.chat("p5", f"{_name(state, other)}の昨日の票、どう見ても浮いてる。")
    provider.vote = other
    provider.reason = "p5さんの指摘で変えます。"
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    audit = _latest_audit(recorder, speaker, state.day)
    assert audit.vote_target == other
    assert audit.target_change_classification == VoteChangeKind.PUBLIC_ARGUMENT
    # The reason the seat gave, and the message that moved it, both survive.
    assert argued in audit.target_change_reason
    assert "p5さんの指摘で変えます。" in audit.target_change_reason


def test_an_argument_about_the_old_target_also_explains_leaving_it():
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    argued = controller.chat("p5", f"{_name(state, plan)}の訂正は素直だったと思う。")
    provider.vote = other
    provider.reason = f"{_name(state, 'p5')}さんの話を聞いて変えます。"
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    audit = _latest_audit(recorder, speaker, state.day)
    assert audit.target_change_classification == VoteChangeKind.PUBLIC_ARGUMENT
    assert argued in audit.target_change_reason


def _change_after(line: str, reason: str = "気が変わった。", author: str = "p5"):  # type: ignore[no-untyped-def]
    """The seat names `plan`, `author` then posts `line`, and the seat votes `other`.

    `{plan}` and `{other}` in `line` and `reason` become those seats' names.
    """
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    names = {"plan": _name(state, plan), "other": _name(state, other)}
    message_id = controller.chat(author, line.format(**names))
    provider.vote = other
    provider.reason = reason.format(**names)
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    return _latest_audit(recorder, speaker, state.day), message_id, recorder


# Lines that name the old or the new target without arguing for the move. The
# seat's reason cites nobody, so none of them can be what changed its mind.
@pytest.mark.parametrize(
    "line",
    [
        # agreement with the seat's own plan
        "今日は{plan}さん吊りで決まりですね、異論なし。",
        # an argument against the target the seat then voted
        "{other}さんは白確定なので吊りません。",
        # greetings
        "{other}さん、おはようございます。",
        "{plan}さん、おはようございます。",
        # the day's designation, naming the seat's own plan
        "本日の吊り先は{plan}さんとします。",
    ],
    ids=["agreement", "against_new", "greeting_new", "greeting_old", "designation"],
)
def test_a_later_mention_the_reason_does_not_cite_leaves_the_change_unexplained(line: str):
    audit, _message_id, recorder = _change_after(line)

    assert audit.target_change_classification == VoteChangeKind.UNEXPLAINED
    assert audit.target_change_reason == "気が変わった。"
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unexplained_vote_change_count == 1


def test_a_reason_that_cites_the_later_speaker_by_name_is_a_public_argument():
    audit, message_id, _ = _change_after(
        "{other}さんの昨日の票、浮いてる。", reason="Player5さんの指摘で{other}さんにします。"
    )

    assert audit.target_change_classification == VoteChangeKind.PUBLIC_ARGUMENT
    assert message_id in audit.target_change_reason
    assert "Player5さんの指摘で" in audit.target_change_reason


def test_citing_a_seat_that_said_nothing_later_explains_nothing():
    audit, _message_id, _ = _change_after(
        "{other}さんの昨日の票、浮いてる。", reason="Player6さんの指摘で{other}さんにします。"
    )

    assert audit.target_change_classification == VoteChangeKind.UNEXPLAINED


def test_the_cited_name_is_matched_whole_so_player1_is_not_player11():
    """Player11 spoke later; a reason naming Player1 cites somebody else."""
    cited_wrong = _change_after(
        "{other}さんの昨日の票、浮いてる。",
        reason="Player1さんの指摘で{other}さんにします。",
        author="p11",
    )[0]
    cited_right = _change_after(
        "{other}さんの昨日の票、浮いてる。",
        reason="Player11さんの指摘で{other}さんにします。",
        author="p11",
    )[0]

    assert cited_wrong.target_change_classification == VoteChangeKind.UNEXPLAINED
    assert cited_right.target_change_classification == VoteChangeKind.PUBLIC_ARGUMENT


def test_an_argument_made_before_the_seats_own_turn_explains_nothing():
    """The seat had heard it and still named its plan; it cannot be what moved it."""
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    controller.chat("p5", f"{_name(state, other)}の昨日の票、どう見ても浮いてる。")
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    provider.vote = other
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    audit = _latest_audit(recorder, speaker, state.day)
    assert audit.target_change_classification == VoteChangeKind.UNEXPLAINED
    assert audit.target_change_reason == "気が変わった。"


def test_the_seats_own_later_words_are_not_someone_elses_argument():
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    # After the turn: a seat naming nobody relevant, then the voter itself
    # naming the new target. Nothing anyone else said comes after that.
    controller.chat("p5", "今日は灰を詰めたい。")
    controller.chat(speaker, f"{_name(state, other)}も気になってきた。")
    provider.vote = other
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    audit = _latest_audit(recorder, speaker, state.day)
    assert audit.target_change_classification == VoteChangeKind.UNEXPLAINED


def _after_a_closing_line(designated: str, stage: str = "final_designation"):  # type: ignore[no-untyped-def]
    """The seat names `plan`; a facilitator's closing turn then names `designated`;
    the seat votes `other` with a reason that cites nobody (run 16's カイト)."""
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    facilitator = next(
        pid
        for pid in state.alive_ids()
        if pid not in (speaker, plan, other, "p0")
        and state.players[pid].role is not RoleName.FREEMASON
    )
    target = {"plan": plan, "other": other}[designated]
    provider.plan = target
    provider.line = f"本指定は{_name(state, target)}さん。投票をお願いします。"
    asyncio.run(coordinator._speak(controller, state, facilitator, stage))
    provider.vote = other
    provider.reason = "進行に乗る。"
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    return _latest_audit(recorder, speaker, state.day), recorder


def test_following_the_final_designation_is_a_cause_the_live_audit_sees():
    """X10 (run 16, day 2): the live audit reads the designation as the offline one does."""
    audit, recorder = _after_a_closing_line("other")

    assert audit.target_change_classification == VoteChangeKind.FINAL_DESIGNATION
    assert "進行に乗る。" in audit.target_change_reason
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unexplained_vote_change_count == 0
    assert report.justified_vote_change_count == 1


def test_a_designation_of_the_seats_own_candidate_does_not_explain_leaving_it():
    audit, _recorder = _after_a_closing_line("plan")

    assert audit.target_change_classification == VoteChangeKind.UNEXPLAINED


def test_the_v3_change_record_calls_the_baseline_a_note_not_a_statement():
    """Under v3 the stated target is the private plan note; nobody said it aloud."""
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    provider.vote = other
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    change = next(u for u in recorder.transcript.utterances if u.kind == "vote_change")
    assert change.text.startswith(f"メモでは{state.players[plan].name}を第一候補としたが")
    assert f"最終判断で{state.players[other].name}へ変更した" in change.text
    assert "公開発言では" not in change.text


def test_the_vote_instruction_asks_a_seat_that_moves_to_name_who_moved_it():
    """The audit reads the cause from the reason, so the prompt has to ask for it."""
    instruction = CHAT_VOTE_OUTPUT_INSTRUCTION

    assert "卓で名指しした吊り先と違う相手に入れるなら" in instruction
    assert "誰の発言で変えたのか" in instruction
    assert "名前" in instruction
    # Plain wording: a request to write out the model's thinking can be refused.
    for word in ("思考", "推論", "thought", "reasoning", "考え"):
        assert word not in instruction


# -- every ballot is audited --


def test_a_seat_that_never_spoke_still_has_its_ballot_audited():
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    silent = "p7"
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, silent))

    audit = _latest_audit(recorder, silent, state.day)
    assert audit.vote_only is True
    assert audit.vote_target == plan
    assert audit.decision_target is None
    # Compared with nothing -- not "unchanged".
    assert audit.target_change_classification == VoteChangeKind.NO_BASIS
    assert audit.target_change_reason == "その日の手番なし"
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.vote_count == 1
    assert (report.no_basis_vote_count, report.compared_vote_count) == (1, 0)
    # No discussion decision was made, so none is counted.
    assert report.decision_count == 0


def test_a_change_off_a_memo_target_the_seat_never_said_is_not_judged():
    """The gate measures the public stance; the table heard none to leave."""
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    provider.line = "灰の発言を比べたい。"
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    provider.vote = other
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    audit = _latest_audit(recorder, speaker, state.day)
    assert audit.target_change_classification == VoteChangeKind.MEMO_TARGET_UNSPOKEN
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unexplained_vote_change_count == 0
    assert report.memo_target_unspoken_count == 1


def test_a_speaker_keeps_one_audit_for_its_turn_and_ballot():
    controller, state, coordinator, provider, recorder, speaker, plan, other = _v3_table()
    asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    state.phase = Phase.VOTING

    asyncio.run(coordinator._cast_vote(controller, state, speaker))

    audits = [a for a in recorder.transcript.decision_audits if a.player_id == speaker]
    assert len(audits) == 1
    assert audits[0].vote_only is False
    assert audits[0].vote_target == plan


# -- a werewolf voting its own partner --


def _wolf_pair_board() -> tuple[GameState, str, str]:
    state = boards.deal(
        {"p1": RoleName.WEREWOLF, "p2": RoleName.WEREWOLF, "p4": RoleName.SEER}, day=2
    )
    boards.kill_first_victim(state, "p16")
    state.phase = Phase.VOTING
    return state, "p1", "p2"


def _ally_vote(
    state: GameState, voter: str, ally: str
) -> tuple[TranscriptRecorder, AICoordinator]:
    ai_ids = [pid for pid in boards.PLAYER_IDS if pid != "p0"]
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        state,
        ai_ids,
        _PlanAndBallot(ally),
        seed=1,
        reasoning=ReasoningRuntime(state, ai_ids, seed=1),
        model_decides=True,
        recorder=recorder,
    )
    asyncio.run(coordinator._cast_vote(game1.BallotBox(state), state, voter))
    return recorder, coordinator


def _wolf_line(state: GameState, author: str, text: str, day: int) -> None:
    number = state.next_message_number
    state.next_message_number += 1
    state.chat_log.append(
        ChatMessage(f"m{number}", author, text, ChatChannel.WOLF, day=day)
    )


def test_an_ally_vote_with_no_plan_and_no_public_mark_is_unplanned():
    state, voter, ally = _wolf_pair_board()

    recorder, _ = _ally_vote(state, voter, ally)

    audit = _latest_audit(recorder, voter, state.day)
    assert audit.ally_vote is True
    assert audit.ally_vote_planned is False
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert report.unplanned_wolf_ally_vote_count == 1


def test_an_ally_vote_the_wolf_chat_agreed_on_is_planned():
    state, voter, ally = _wolf_pair_board()
    _wolf_line(state, ally, f"明日は{_name(state, ally)}を切って信用を取ってくれ。", day=1)

    recorder, _ = _ally_vote(state, voter, ally)

    audit = _latest_audit(recorder, voter, state.day)
    assert audit.ally_vote_planned is True
    plan = recorder.transcript.wolf_ally_vote_plans[-1]
    assert (plan.voter_id, plan.target_id, plan.basis) == (voter, ally, "wolf_chat")


@pytest.mark.parametrize(
    "line",
    ["{name}吊りは避けたい。", "{name}には投票しない。", "{name}、今夜はケント噛みでいこう。"],
)
def test_wolf_chat_that_names_the_partner_without_planning_its_ballot_is_no_plan(line: str):
    state, voter, ally = _wolf_pair_board()
    _wolf_line(state, voter, line.format(name=_name(state, ally)), day=1)

    recorder, _ = _ally_vote(state, voter, ally)

    assert _latest_audit(recorder, voter, state.day).ally_vote_planned is False


def test_an_older_nights_chat_is_not_todays_plan():
    state, voter, ally = _wolf_pair_board()
    state.day = 3
    _wolf_line(state, ally, f"明日は{_name(state, ally)}を切って信用を取ってくれ。", day=1)

    recorder, _ = _ally_vote(state, voter, ally)

    assert _latest_audit(recorder, voter, state.day).ally_vote_planned is False


def test_an_ally_the_table_already_holds_a_black_on_is_publicly_marked():
    state, voter, ally = _wolf_pair_board()
    boards.claim(state, "p4", RoleName.SEER, day=2)
    boards.verdict(state, "p4", "seer", ally, True, day=2, referenced_day=1)

    recorder, _ = _ally_vote(state, voter, ally)

    audit = _latest_audit(recorder, voter, state.day)
    assert audit.ally_vote_planned is True
    assert recorder.transcript.wolf_ally_vote_plans[-1].basis == "public_mark"


def test_first_game_day_3_kaito_voting_his_proven_fake_partner_daiki_is_planned():
    """カイト (wolf, silent that day) voted ダイキ (wolf, fake seer).

    ダイキ had claimed シオン as his night-1 look, シオン was alive, and night 1
    left two bodies -- one of them a cursed fox, so the real seer's look was
    one of the dead. No world has ダイキ as the real seer. Cutting a partner the
    whole table can prove fake is ordinary wolf play, not an unplanned betrayal.
    The ballot was not audited at all in the live game.
    """
    ids = game1.name_to_id()
    kaito, daiki = ids["カイト"], ids["ダイキ"]
    state = game1.state_at_vote(3)
    assert game1.ballots(3)[kaito] == daiki
    assert not [m for m in state.chat_log if m.author_id == kaito and m.day == 3]
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        state,
        game1.AI_IDS,
        _PlanAndBallot(daiki),
        seed=1,
        reasoning=ReasoningRuntime(state, game1.AI_IDS, seed=1),
        model_decides=True,
        recorder=recorder,
    )

    asyncio.run(coordinator._cast_vote(game1.BallotBox(state), state, kaito))

    audit = _latest_audit(recorder, kaito, 3)
    assert audit.vote_only is True
    assert (audit.ally_vote, audit.ally_vote_planned) == (True, True)
    plan = recorder.transcript.wolf_ally_vote_plans[-1]
    assert plan.basis == "public_mark"
    assert any(daiki in evidence for evidence in plan.public_evidence_ids)
    report = ReasoningTranscriptAnalyzer().analyze(recorder.transcript)
    assert (report.wolf_ally_vote_count, report.unplanned_wolf_ally_vote_count) == (1, 0)


# -- the first game, re-classified offline --


def _vote_reasons() -> dict[tuple[str, int], str]:
    """The reason each seat wrote with its ballot in the first game."""
    original = GameTranscript.from_dict(game1.load_transcript())
    return {(u.player_id, u.day): u.text for u in original.by_kind("vote")}


def _reclassify_first_game() -> Counter[tuple[int, str, str]]:
    """Every audited ballot of the first game, put through today's classifier.

    The audits carry what each seat had seen when it spoke (the published
    results, the corrections); the replayed board carries everything said
    before the vote. Together they are exactly what `_cast_vote` looked at,
    with the reason each seat wrote on its ballot.
    """
    original = GameTranscript.from_dict(game1.load_transcript())
    reasons = _vote_reasons()
    latest: dict[tuple[str, int], int] = {}
    for index, audit in enumerate(original.decision_audits):
        latest[(audit.player_id, audit.day)] = index
    kinds: Counter[tuple[int, str, str]] = Counter()
    for day in sorted({audit.day for audit in original.decision_audits}):
        state = game1.state_at_vote(day)
        recorder = TranscriptRecorder()
        recorder.transcript = GameTranscript.from_dict(game1.load_transcript())
        coordinator = AICoordinator(
            state, game1.AI_IDS, _PlanAndBallot("p1"), seed=1, recorder=recorder
        )
        for (player_id, audit_day), index in latest.items():
            audit = original.decision_audits[index]
            if audit_day != day or audit.vote_target is None:
                continue
            kind, _reason = coordinator._classify_vote_change(
                state,
                player_id,
                stated_target=audit.displayed_target,
                actual_target=audit.vote_target,
                candidates=state.votable_ids(player_id),
                reason=reasons[(player_id, day)],
            )
            kinds[(day, player_id, kind)] += 1
    return kinds


def test_the_first_games_ballots_reclassify_with_no_unexplained_change():
    kinds = _reclassify_first_game()
    ids = game1.name_to_id()

    by_kind = Counter(kind for (_day, _seat, kind) in kinds.elements())
    assert by_kind[VoteChangeKind.UNEXPLAINED] == 0
    assert sum(by_kind.values()) == 47
    # アオイ moved off レン to ミオ citing マコト's later point, but never said
    # レン at the table that day: a change off an unspoken memo target.
    assert kinds[(2, ids["アオイ"], VoteChangeKind.MEMO_TARGET_UNSPOKEN)] == 1
    # The two other changes were filed as new evidence live. Each reason names
    # a seat that argued about the old or the new target after the voter's
    # last line (ミオ: ツムギ's m50/m51; ツムギ: m61-m63), and an argument the
    # reason cites outranks a verdict published beside it.
    assert kinds[(3, ids["ミオ"], VoteChangeKind.PUBLIC_ARGUMENT)] == 1
    assert kinds[(4, ids["ツムギ"], VoteChangeKind.PUBLIC_ARGUMENT)] == 1
    # 17 ballots had nothing to compare with (no turn that day, or no memo
    # target) and used to be counted here as unchanged.
    assert by_kind[VoteChangeKind.NONE] == 27
    assert by_kind[VoteChangeKind.NO_BASIS] == 17


def test_aoi_s_day_2_change_left_a_memo_target_she_never_said():
    """「昼はレンさんと言いましたが」 -- but neither of her two lines that day names レン.

    The gate measures the stance the table heard, so the change is listed,
    not judged, whatever the reason cites.
    """
    ids = game1.name_to_id()
    aoi, ren, mio = ids["アオイ"], ids["レン"], ids["ミオ"]
    state = game1.state_at_vote(2)
    assert not [
        m for m in state.chat_log if m.day == 2 and m.author_id == aoi and "レン" in m.content
    ]
    recorder = TranscriptRecorder()
    recorder.transcript = GameTranscript.from_dict(game1.load_transcript())
    coordinator = AICoordinator(
        state, game1.AI_IDS, _PlanAndBallot("p1"), seed=1, recorder=recorder
    )

    for reason in (_vote_reasons()[(aoi, 2)], "気が変わった。"):
        kind, _why = coordinator._classify_vote_change(
            state,
            aoi,
            stated_target=ren,
            actual_target=mio,
            candidates=state.votable_ids(aoi),
            reason=reason,
        )
        assert kind == VoteChangeKind.MEMO_TARGET_UNSPOKEN


def test_had_aoi_said_ren_her_citing_makoto_would_make_it_persuasion():
    """The reason names マコト, who spoke after アオイ: that is what makes it persuasion.

    Her last line is made to name レン (the stance is then public). The same
    move with a reason that names nobody who spoke later is filed as
    unexplained, although マコト's later lines still name both レン and ミオ.
    """
    ids = game1.name_to_id()
    aoi, ren, mio = ids["アオイ"], ids["レン"], ids["ミオ"]
    reason = _vote_reasons()[(aoi, 2)]
    assert "マコトさんの指摘で変えます" in reason
    state = game1.state_at_vote(2)
    last_own = [m for m in state.chat_log if m.day == 2 and m.author_id == aoi][-1]
    last_own.content += "今日はレンさんに入れます。"
    recorder = TranscriptRecorder()
    recorder.transcript = GameTranscript.from_dict(game1.load_transcript())
    coordinator = AICoordinator(
        state, game1.AI_IDS, _PlanAndBallot("p1"), seed=1, recorder=recorder
    )

    def classify(text: str) -> tuple[str, str]:
        return coordinator._classify_vote_change(
            state,
            aoi,
            stated_target=ren,
            actual_target=mio,
            candidates=state.votable_ids(aoi),
            reason=text,
        )

    kind, why = classify(reason)
    assert kind == VoteChangeKind.PUBLIC_ARGUMENT
    makoto_lines = [
        m.message_id
        for m in state.chat_log
        if m.day == 2 and m.author_id == ids["マコト"] and m.channel is ChatChannel.PUBLIC
    ]
    assert any(message_id in why for message_id in makoto_lines)
    assert classify("気が変わった。")[0] == VoteChangeKind.UNEXPLAINED


def test_on_the_first_games_boards_no_move_with_a_vague_reason_counts_as_persuasion():
    """Every seat with a stated candidate, moved to every other legal candidate.

    Counting any later mention of either target as the cause filed 167 of
    these 257 hypothetical moves as persuaded, because the day's summary and
    designation name the candidate after nearly everyone has spoken. With a
    reason that cites nobody, none is.
    """
    original = GameTranscript.from_dict(game1.load_transcript())
    latest: dict[tuple[str, int], int] = {}
    for index, audit in enumerate(original.decision_audits):
        latest[(audit.player_id, audit.day)] = index
    kinds: Counter[str] = Counter()
    for day in sorted({audit.day for audit in original.decision_audits}):
        state = game1.state_at_vote(day)
        recorder = TranscriptRecorder()
        recorder.transcript = GameTranscript.from_dict(game1.load_transcript())
        coordinator = AICoordinator(
            state, game1.AI_IDS, _PlanAndBallot("p1"), seed=1, recorder=recorder
        )
        for (player_id, audit_day), index in latest.items():
            audit = original.decision_audits[index]
            if audit_day != day or audit.vote_target is None or audit.displayed_target is None:
                continue
            for moved_to in state.votable_ids(player_id):
                if moved_to == audit.displayed_target:
                    continue
                kind, _reason = coordinator._classify_vote_change(
                    state,
                    player_id,
                    stated_target=audit.displayed_target,
                    actual_target=moved_to,
                    candidates=state.votable_ids(player_id),
                    reason="気が変わった。",
                )
                kinds[kind] += 1

    assert sum(kinds.values()) == 257
    assert kinds[VoteChangeKind.PUBLIC_ARGUMENT] == 0
    # A result published after the seat spoke still explains a move -- but only
    # one about the old or the new target, or by one of them. Any new verdict
    # used to explain every move of the day: 57 of these, 43 of them on
    # verdicts that touched neither seat or were the voter's own.
    # (14 with the four seats that never said their candidate counted; their
    # 43 moves leave a memo target nobody heard, and are not judged.)
    assert kinds[VoteChangeKind.NEW_PUBLIC_EVIDENCE] == 13
    assert kinds[VoteChangeKind.MEMO_TARGET_UNSPOKEN] == 43
    # アカリ on day 3 named her candidate ダイキ only in a question to him
    # (「ダイキさん、この点はどう説明されますか？」): no stance on him the table
    # heard (`stance_excuse`), so her 10 moves are listed, not judged.
    assert kinds[VoteChangeKind.MEMO_TARGET_ONLY_REPORTED_OR_ASKED] == 10
    assert kinds[VoteChangeKind.UNEXPLAINED] == 191
