"""Readers of a published verdict's date agree with the projection on a restatement.

`ObservationSet` dates a plain restatement -- a verdict said again on a later
day without naming a night -- by the statement it repeats
(docs/real_game_log_reasoning_plan.md, principle 3; see
`test_restated_verdicts.py`). Three readers went on dating it by the standing
version, i.e. today:

* the belief engine's verdict evidence, whose explanation reaches supporting
  evidence, vote reasons and belief changes, said "published today, about last
  night" for a fresh engine;
* the code-written day summary listed it under 「本日公開された判定」, and
  dropped it from the day it was really published;
* the discussion scheduler counted a restated black as today's black and owed
  its target a second reply turn.

Each now reads the projection's date.
"""

from __future__ import annotations

from app.ai.coordinator import AICoordinator
from app.ai.provider.mock import MockProvider
from app.ai.reasoning.belief import BeliefEngine
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import PlayerPrivatePerspective
from app.ai.reasoning.solver import build_solver
from app.ai.reasoning.summaries import render_public_fact_summary
from app.ai.schemas import DiscussionOutput
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.speech_events import SpeechEventType
from app.engine.state import GameState
from app.sessions.models import DiscussionRoundState
from tests.ai.reasoning.solver import boards
from tests.conftest import make_controller

SEER, BLACKED, WHITED = "p4", "p9", "p10"


def _board_with_restatement(*, restated_night: int | None = None) -> GameState:
    """p4 (seer CO) blacks p9 on day 2 for night 1 and says it again on day 3.

    Day 3 also brings p4's real result for night 2, a white on p10, so the
    day has one new verdict and one restated one.
    """
    state = boards.deal({SEER: RoleName.SEER, BLACKED: RoleName.WEREWOLF}, day=3)
    boards.claim(state, SEER, RoleName.SEER, day=1)
    _verdict(state, BLACKED, True, day=2, referenced_day=1, message_id="m20")
    _verdict(state, WHITED, False, day=3, referenced_day=2, message_id="m40")
    _verdict(state, BLACKED, True, day=3, referenced_day=restated_night, message_id="m41")
    return state


def _verdict(
    state: GameState,
    target_id: str,
    black: bool,
    *,
    day: int,
    referenced_day: int | None,
    message_id: str,
) -> None:
    boards._event(
        state,
        day,
        SEER,
        SpeechEventType.ABILITY_RESULT,
        role=RoleName.SEER,
        target_id=target_id,
        result_is_werewolf=black,
        referenced_day=referenced_day,
        source_message_id=message_id,
    )


# -- belief engine --


def _fresh_engine(state: GameState, player_id: str = "p2") -> BeliefEngine:
    engine = BeliefEngine(player_id, PlayerPrivatePerspective(player_id))
    observations = ObservationSet.from_state(state)
    engine.observe(
        PublicFactLedger(state),
        build_solver(observations, PlayerPrivatePerspective(player_id)),
        observations,
    )
    return engine


def _verdict_evidence(engine: BeliefEngine, target_id: str):  # type: ignore[no-untyped-def]
    return next(
        record
        for record in engine.evidence
        if record.evidence_id.startswith(f"verdict:{SEER}:seer:{target_id}:")
    )


def test_a_fresh_engine_dates_a_restated_black_by_the_statement_it_repeats():
    record = _verdict_evidence(_fresh_engine(_board_with_restatement()), BLACKED)

    assert record.explanation == (
        f"2日目、{SEER}の占い判定で{BLACKED}は黒と公開された（対象の夜: 1日目）。"
    )
    # The source is the original statement, not today's mention of it.
    assert record.source_event_ids == ("m20",)


def test_a_fresh_engine_still_dates_todays_new_verdict_today():
    record = _verdict_evidence(_fresh_engine(_board_with_restatement()), WHITED)

    assert record.explanation == (
        f"3日目、{SEER}の占い判定で{WHITED}は白と公開された（対象の夜: 2日目）。"
    )
    assert record.source_event_ids == ("m40",)


def test_a_fresh_engine_matches_one_that_read_the_original_on_its_own_day():
    state = _board_with_restatement()
    day_two = boards.deal({SEER: RoleName.SEER, BLACKED: RoleName.WEREWOLF}, day=2)
    boards.claim(day_two, SEER, RoleName.SEER, day=1)
    _verdict(day_two, BLACKED, True, day=2, referenced_day=1, message_id="m20")
    engine = _fresh_engine(day_two)
    engine.observe(PublicFactLedger(state), None, ObservationSet.from_state(state))

    assert _verdict_evidence(engine, BLACKED).explanation == _verdict_evidence(
        _fresh_engine(state), BLACKED
    ).explanation


def test_a_restatement_that_names_a_night_of_its_own_is_dated_today():
    # A new account of the timing is a new statement, as in the projection.
    record = _verdict_evidence(_fresh_engine(_board_with_restatement(restated_night=2)), BLACKED)

    assert record.explanation.startswith("3日目、")
    assert "（対象の夜: 2日目）" in record.explanation


# -- the code-written day summary --


def _published_line(summary: str) -> str:
    return next(
        (line for line in summary.splitlines() if line.startswith("- 本日公開された判定")), ""
    )


def test_the_day_summary_does_not_list_a_restatement_as_published_today():
    line = _published_line(render_public_fact_summary(PublicFactLedger(_board_with_restatement())))

    assert f"Player{WHITED[1:]}({WHITED})=白" in line
    assert f"({BLACKED})" not in line


def test_the_day_summary_keeps_a_restated_verdict_on_the_day_it_was_published():
    line = _published_line(
        render_public_fact_summary(PublicFactLedger(_board_with_restatement()), day=2)
    )

    assert f"Player{BLACKED[1:]}({BLACKED})=黒" in line


def test_the_day_summary_lists_a_restatement_with_a_new_night_today():
    line = _published_line(
        render_public_fact_summary(PublicFactLedger(_board_with_restatement(restated_night=2)))
    )

    assert f"Player{BLACKED[1:]}({BLACKED})=黒" in line


# -- owed replies to today's blacks --


def _output(target: str | None, text: str = "見解") -> DiscussionOutput:
    output = DiscussionOutput(public_message=text)
    output.reasoning_memo.execution_target = target
    return output


def _day_three_table(*, restated_night: int | None) -> tuple[GameState, AICoordinator]:
    """p1 blacked p2 on day 2 and p2 answered it; on day 3 p1 says the black again."""
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    coordinator = AICoordinator(state, ["p1", "p2", "p3", "p4"], MockProvider(seed=1), seed=1)
    controller.co("p1", "seer")
    black = controller.chat("p1", "占いCO。p2は黒。")
    controller.public_result("p1", "seer", "p2", True, source_message_id=black, referenced_day=1)
    controller.chat("p2", "その黒は偽。対抗で占いCOする。")
    state.day = 3
    again = controller.chat("p1", "昨日も言ったがp2は黒。")
    controller.public_result(
        "p1", "seer", "p2", True, source_message_id=again, referenced_day=restated_night
    )
    controller.chat("p3", "p2の反論を待ちたい。")
    return state, coordinator


def _round(state: GameState) -> DiscussionRoundState:
    return DiscussionRoundState(
        day=state.day,
        order=["p1", "p3"],
        cursor=2,
        outputs=[("p1", _output("p2")), ("p3", _output("p4"))],
        speech_counts={"p1": 1, "p3": 1},
        major_targets_ready=True,
        max_total=10,
    )


def test_a_black_only_restated_today_owes_its_target_no_second_reply():
    state, coordinator = _day_three_table(restated_night=None)
    round_state = _round(state)

    speaker, stage = coordinator._next_discussion_speaker(state, round_state)

    assert (speaker, stage) != ("p2", "rebuttal_or_reassessment")
    assert stage == "consensus_summary"
    assert round_state.max_total == 10


def test_a_black_restated_with_a_new_night_is_a_new_black_and_owes_a_reply():
    state, coordinator = _day_three_table(restated_night=2)
    round_state = _round(state)

    speaker, stage = coordinator._next_discussion_speaker(state, round_state)

    assert (speaker, stage) == ("p2", "rebuttal_or_reassessment")
    assert "p2" in round_state.major_targets


# -- a restatement that names the same night again --
#
# 「昨日も言ったが、1日目の夜はp9黒」 repeats the statement; only a different
# night is a new account of the timing.


def test_a_restatement_naming_the_same_night_is_dated_by_the_statement_it_repeats():
    state = _board_with_restatement(restated_night=1)

    record = _verdict_evidence(_fresh_engine(state), BLACKED)
    assert record.explanation == (
        f"2日目、{SEER}の占い判定で{BLACKED}は黒と公開された（対象の夜: 1日目）。"
    )
    assert record.source_event_ids == ("m20",)
    line = _published_line(render_public_fact_summary(PublicFactLedger(state)))
    assert f"({BLACKED})" not in line
    assert f"Player{WHITED[1:]}({WHITED})=白" in line


def test_a_black_restated_for_the_same_night_owes_its_target_no_second_reply():
    state, coordinator = _day_three_table(restated_night=1)
    round_state = _round(state)

    speaker, stage = coordinator._next_discussion_speaker(state, round_state)

    assert (speaker, stage) != ("p2", "rebuttal_or_reassessment")
    assert round_state.max_total == 10
