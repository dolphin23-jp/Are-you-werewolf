"""The public ledger must expose every public fact and no private one."""

from __future__ import annotations

import json

from app.ai.reasoning import PublicFactLedger, mentions_player
from app.engine.roles import RoleName
from app.engine.state import PublicDeathCause
from tests.ai.reasoning.fixtures import (
    cast_vote,
    declare_co,
    execute,
    kill_at_night,
    make_first_victim,
    make_state,
    publish_result,
)


def test_the_ledger_reports_public_deaths_without_their_real_cause():
    state = make_state(day=2)
    make_first_victim(state, "p0")
    execute(state, "p5", day=1)
    kill_at_night(state, "p7", day=1)
    ledger = PublicFactLedger(state)

    assert ledger.player("p5").death_cause == PublicDeathCause.EXECUTED
    # Attacked and cursed are indistinguishable to the table; the ledger must
    # not be the place that gives the difference away.
    assert ledger.player("p7").death_cause == PublicDeathCause.NIGHT
    assert ledger.player("p0").death_cause == PublicDeathCause.FIRST_VICTIM
    assert set(ledger.dead_ids()) == {"p0", "p5", "p7"}
    assert "p5" not in ledger.alive_ids()


def test_no_private_role_or_night_action_reaches_the_public_dict():
    state = make_state(
        day=2,
        roles={"p3": RoleName.WEREWOLF, "p4": RoleName.SEER, "p9": RoleName.FOX},
    )
    state.pending_divine = ("p4", "p3")
    state.pending_attack = ("p3", "p6")
    kill_at_night(state, "p6", day=1)
    execute(state, "p2", day=1)

    dumped = json.dumps(PublicFactLedger(state).to_public_dict(), ensure_ascii=False)

    # Nobody COed here, so any role name in the dump could only have come from
    # the hidden assignment. Likewise the real death cause and the night actions.
    for secret in ("werewolf", "seer", "fox", "attacked", "pending"):
        assert secret not in dumped
    assert "p6" in dumped  # the death itself is public


def test_executions_are_distinguished_from_night_deaths():
    state = make_state(day=3)
    execute(state, "p2", day=1)
    kill_at_night(state, "p8", day=2)
    ledger = PublicFactLedger(state)

    assert ledger.executed_ids() == ("p2",)
    assert ledger.execution_day("p2") == 1
    assert ledger.was_executed("p8") is False
    assert ledger.night_death_ids(2) == ("p8",)


def test_the_latest_public_co_is_reported_and_absent_cos_stay_none():
    state = make_state()
    declare_co(state, "p3", RoleName.SEER, day=1)
    ledger = PublicFactLedger(state)

    assert ledger.claimed_role_of("p3") == RoleName.SEER
    assert ledger.claimed_role_of("p4") is None


def test_the_actual_vote_target_is_read_back_per_day_and_round():
    state = make_state(day=2)
    cast_vote(state, "p0", "p4", day=1, round_number=1)
    cast_vote(state, "p0", "p6", day=2, round_number=1)
    cast_vote(state, "p0", "p9", day=2, round_number=2)
    ledger = PublicFactLedger(state)

    # The regression this pins: remembering p0's vote from the wrong day or the
    # wrong runoff round, and then arguing from a history that never happened.
    assert ledger.vote_of("p0", 1, 1).target_id == "p4"
    assert ledger.vote_of("p0", 2, 1).target_id == "p6"
    assert ledger.vote_of("p0", 2, 2).target_id == "p9"
    assert ledger.vote_of("p0", 2).target_id == "p9"
    assert ledger.vote_of("p1", 2) is None


def test_p1_and_p11_are_never_confused():
    state = make_state()
    ledger = PublicFactLedger(state)
    publish_result(state, "p3", "seer", "p11", True)

    assert mentions_player("Player11(p11)が黒でした", "p11", "Player11") is True
    assert mentions_player("Player11(p11)が黒でした", "p1", "Player1") is False
    assert mentions_player("Player1(p1)が黒でした", "p11", "Player11") is False
    assert ledger.mentioned_player_ids("Player11(p11)を疑います") == ("p11",)
    assert ledger.find_result("p3", "seer", "p1") is None
    assert ledger.find_result("p3", "seer", "p11").is_werewolf is True


def test_the_ledger_reads_the_live_state_rather_than_copying_it():
    state = make_state()
    ledger = PublicFactLedger(state)
    assert ledger.is_alive("p4") is True

    execute(state, "p4", day=1)

    # A snapshot would still be reporting p4 alive here, which is exactly how a
    # dead player ends up back on the execution block the next morning.
    assert ledger.is_alive("p4") is False
    assert ledger.was_executed("p4") is True
