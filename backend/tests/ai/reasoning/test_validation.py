"""State-consistency rules: what AI output is allowed to assert about the board."""

from __future__ import annotations

from app.ai.reasoning import (
    PublicFactLedger,
    detect_vote_plan_mismatch,
    resolve_target,
    validate_discussion_output,
    validate_public_result_claim,
    validate_reasoning_memo,
)
from app.ai.schemas import DiscussionOutput, PublicResultClaim, ReasoningMemo
from app.engine.roles import RoleName
from tests.ai.reasoning.fixtures import (
    cast_vote,
    declare_co,
    execute,
    kill_at_night,
    make_first_victim,
    make_state,
    publish_result,
)

# -- execution candidates --


def test_yesterdays_executed_player_is_removed_from_todays_execution_target():
    state = make_state(day=2)
    execute(state, "p4", day=1)
    memo = ReasoningMemo(execution_target="p4", suspects=["p4", "p7"])

    result = validate_reasoning_memo(memo, PublicFactLedger(state), owner_id="p3")

    assert result.memo.execution_target == "p7"
    assert "memo_execution_target_dead" in [issue.code for issue in result.issues]


def test_a_repaired_execution_target_comes_from_the_players_own_prior_belief():
    state = make_state(day=2)
    execute(state, "p4", day=1)
    memo = ReasoningMemo(execution_target="p4", suspects=["p8"])

    result = validate_reasoning_memo(
        memo, PublicFactLedger(state), owner_id="p3", previous_execution_target="p6"
    )

    # Deterministic and explainable: yesterday's conclusion, not a random seat
    # and not the first suspect that happens to be listed.
    assert result.memo.execution_target == "p6"


def test_an_unrepairable_execution_target_becomes_none_rather_than_a_guess():
    state = make_state(day=2)
    execute(state, "p4", day=1)

    result = validate_reasoning_memo(
        ReasoningMemo(execution_target="p4"), PublicFactLedger(state), owner_id="p3"
    )

    assert result.memo.execution_target is None


def test_a_deliberate_withdrawal_is_not_refilled():
    state = make_state(day=2)

    result = validate_reasoning_memo(
        ReasoningMemo(execution_target=None, suspects=["p7"]),
        PublicFactLedger(state),
        owner_id="p3",
        previous_execution_target="p6",
    )

    assert result.memo.execution_target is None
    assert result.ok


def test_a_player_cannot_name_themselves_as_the_execution_target():
    state = make_state(day=2)
    memo = ReasoningMemo(execution_target="p3", suspects=["p3", "p9"], trusted_seer="p3")

    result = validate_reasoning_memo(memo, PublicFactLedger(state), owner_id="p3")

    assert result.memo.execution_target == "p9"
    assert result.memo.suspects == ["p9"]
    assert result.memo.trusted_seer is None
    assert "memo_execution_target_self" in [issue.code for issue in result.issues]


def test_unknown_and_duplicated_player_ids_are_dropped_from_belief_lists():
    state = make_state(day=1)
    memo = ReasoningMemo(
        suspects=["p2", "p2", "p99", "Player4"], fox_candidates=["p11", ""], trusted=["p5"]
    )

    result = validate_reasoning_memo(memo, PublicFactLedger(state), owner_id="p3")

    assert result.memo.suspects == ["p2"]
    assert result.memo.fox_candidates == ["p11"]
    assert result.memo.trusted == ["p5"]


def test_only_survivors_remain_candidates_on_the_final_day():
    # Three seats left: everyone else is already dead, so a stale target from an
    # earlier day must not survive into the last vote.
    state = make_state(day=5)
    for index in range(3, 12):
        execute(state, f"p{index}", day=index - 2)
    ledger = PublicFactLedger(state)
    memo = ReasoningMemo(execution_target="p5", suspects=["p7", "p2"])

    result = validate_reasoning_memo(memo, ledger, owner_id="p0")

    assert set(ledger.alive_ids()) == {"p0", "p1", "p2"}
    assert result.memo.execution_target == "p2"


def test_a_night_victim_is_excluded_from_the_current_ability_target():
    state = make_state(day=2)
    kill_at_night(state, "p6", day=1)
    ledger = PublicFactLedger(state)
    candidates = [pid for pid in ledger.alive_ids() if pid != "p4"]

    resolution = resolve_target("p6", candidates=candidates, preferred=["p9"], actor_id="p4")

    assert "p6" not in candidates
    assert resolution.target == "p9"
    assert resolution.issues[0].code == "target_invalid"


def test_an_invalid_target_never_resolves_to_a_random_seat():
    candidates = ["p2", "p5", "p9"]

    first = resolve_target("p99", candidates=candidates, preferred=["p77"])
    second = resolve_target("p99", candidates=candidates, preferred=["p77"])

    assert first.target == second.target == "p2"


# -- published results --


def test_a_medium_result_about_someone_never_executed_is_rejected():
    state = make_state(day=2)
    declare_co(state, "p3", RoleName.MEDIUM)
    execute(state, "p4", day=1)

    rejected = validate_public_result_claim(
        PublicResultClaim(result_type="medium", target_id="p7", is_werewolf=True),
        PublicFactLedger(state),
        claimant_id="p3",
    )
    accepted = validate_public_result_claim(
        PublicResultClaim(result_type="medium", target_id="p4", is_werewolf=True),
        PublicFactLedger(state),
        claimant_id="p3",
    )

    assert rejected.claim is None
    assert rejected.issues[0].code == "result_medium_target_not_executed"
    assert accepted.claim is not None


def test_seer_and_medium_results_are_validated_separately():
    state = make_state(day=2)
    declare_co(state, "p3", RoleName.SEER)
    make_first_victim(state, "p0")

    # A living, un-executed player is a fine seer target and an impossible
    # medium one; collapsing the two is how a medium reports on the living.
    seer_ok = validate_public_result_claim(
        PublicResultClaim(result_type="seer", target_id="p7", is_werewolf=False),
        PublicFactLedger(state),
        claimant_id="p3",
    )
    wrong_role = validate_public_result_claim(
        PublicResultClaim(result_type="medium", target_id="p7", is_werewolf=False),
        PublicFactLedger(state),
        claimant_id="p3",
    )
    first_victim = validate_public_result_claim(
        PublicResultClaim(result_type="seer", target_id="p0", is_werewolf=True),
        PublicFactLedger(state),
        claimant_id="p3",
    )

    assert seer_ok.claim is not None
    assert wrong_role.claim is None
    assert wrong_role.issues[0].code == "result_role_mismatch"
    assert first_victim.claim is None
    assert first_victim.issues[0].code == "result_seer_target_first_victim"


def test_a_published_white_result_cannot_flip_to_black():
    state = make_state(day=3)
    declare_co(state, "p3", RoleName.MEDIUM)
    execute(state, "p4", day=1)
    publish_result(state, "p3", "medium", "p4", is_werewolf=False, day=2)

    validation = validate_public_result_claim(
        PublicResultClaim(result_type="medium", target_id="p4", is_werewolf=True),
        PublicFactLedger(state),
        claimant_id="p3",
    )

    assert validation.claim.is_werewolf is False
    assert validation.issues[0].code == "result_polarity_conflict"


def test_a_player_cannot_publish_a_result_about_themselves():
    state = make_state(day=2)
    declare_co(state, "p3", RoleName.SEER)

    validation = validate_public_result_claim(
        PublicResultClaim(result_type="seer", target_id="p3", is_werewolf=False),
        PublicFactLedger(state),
        claimant_id="p3",
    )

    assert validation.claim is None
    assert validation.issues[0].code == "result_target_self"


# -- whole discussion turn --


def test_a_whole_turn_is_reconciled_before_anything_downstream_reads_it():
    state = make_state(day=2)
    declare_co(state, "p3", RoleName.MEDIUM)
    execute(state, "p4", day=1)
    publish_result(state, "p3", "medium", "p4", is_werewolf=False, day=2)
    output = DiscussionOutput(
        public_message="霊媒結果を報告します。",
        reasoning_memo=ReasoningMemo(execution_target="p4", suspects=["p9"]),
        alternative_execution_target="p4",
        public_results=[
            PublicResultClaim(result_type="medium", target_id="p4", is_werewolf=True),
            PublicResultClaim(result_type="medium", target_id="p8", is_werewolf=True),
        ],
    )

    checked, issues = validate_discussion_output(
        output, PublicFactLedger(state), speaker_id="p3"
    )

    assert checked.reasoning_memo.execution_target == "p9"
    assert checked.alternative_execution_target is None
    assert [(r.target_id, r.is_werewolf) for r in checked.public_results] == [("p4", False)]
    assert {issue.code for issue in issues} == {
        "memo_execution_target_dead",
        "memo_alternative_target_dead",
        "result_polarity_conflict",
        "result_medium_target_not_executed",
    }


def test_an_observer_seat_is_never_repaired_into_an_execution_target():
    state = make_state(day=1)

    result = validate_reasoning_memo(
        ReasoningMemo(execution_target="p99", suspects=["p2", "p5"]),
        PublicFactLedger(state),
        owner_id="p3",
        excluded_target_ids={"p2"},
    )

    assert result.memo.execution_target == "p5"


# -- stated plan vs actual ballot --


def test_a_vote_that_contradicts_the_stated_plan_is_detected():
    state = make_state(day=2)
    cast_vote(state, "p3", "p9", day=2, round_number=1)

    mismatch = detect_vote_plan_mismatch(
        PublicFactLedger(state), voter_id="p3", stated_target="p7", day=2, round_number=1
    )

    assert mismatch.stated_target == "p7"
    assert mismatch.actual_target == "p9"
    assert mismatch.stated_target_votable is True


def test_a_vote_matching_the_stated_plan_is_not_flagged():
    state = make_state(day=2)
    cast_vote(state, "p3", "p7", day=2, round_number=1)
    ledger = PublicFactLedger(state)

    assert (
        detect_vote_plan_mismatch(
            ledger, voter_id="p3", stated_target="p7", day=2, round_number=1
        )
        is None
    )
    # No stated plan at all is silence, not incoherence.
    assert (
        detect_vote_plan_mismatch(
            ledger, voter_id="p3", stated_target=None, day=2, round_number=1
        )
        is None
    )


def test_a_runoff_that_removed_the_stated_target_is_marked_as_such():
    state = make_state(day=2)
    state.runoff_candidates = ["p5", "p9"]
    cast_vote(state, "p3", "p9", day=2, round_number=2)

    mismatch = detect_vote_plan_mismatch(
        PublicFactLedger(state), voter_id="p3", stated_target="p7", day=2, round_number=2
    )

    assert mismatch.stated_target_votable is False
