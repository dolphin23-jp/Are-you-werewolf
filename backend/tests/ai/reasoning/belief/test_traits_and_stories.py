"""Personality that changes decisions, and bluffers with two heads.

Two AIs phrasing things differently and then voting identically are one AI with
two voices. These tests are about the difference actually reaching the ballot --
while never letting it reach a conclusion the rules have already settled.
"""

from __future__ import annotations

from app.ai.reasoning.belief import (
    TRAIT_PROFILES,
    BeliefEngine,
    CognitiveTraits,
    CorrectionKind,
    EvidenceRecord,
    FactCorrection,
    HypothesisRank,
    RoleCertainty,
    StoryStatus,
    assign_traits,
    deception_state_for,
    private_solver,
    rank_hypotheses,
    refresh_story,
    story_solver,
    summarise,
    vote_fact_id,
)
from app.ai.reasoning.belief.state import RankedHypothesis
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import CommonPublicPerspective
from app.ai.reasoning.solver import Certainty, build_solver, has_role
from app.engine.roles import RoleName
from app.engine.state import VoteRecord
from tests.ai.reasoning.solver import boards


def _contested_board(day: int = 2):  # type: ignore[no-untyped-def]
    """A published black on p9, and the table's votes piling onto p11 instead."""
    state = boards.deal(
        {"p1": RoleName.WEREWOLF, "p4": RoleName.SEER, "p6": RoleName.FOX}, day=day
    )
    boards.claim(state, "p4", RoleName.SEER, day=1)
    boards.verdict(state, "p4", "seer", "p9", True, day=1)
    for voter in ("p0", "p2", "p3", "p5", "p7", "p8"):
        state.vote_records.append(
            VoteRecord(voter_id=voter, target_id="p11", day=1, round=1)
        )
    return state


# -- traits change decisions, not just wording --


def test_the_same_board_leads_two_personalities_to_different_targets():
    ledger = PublicFactLedger(_contested_board())
    evidence_driven = BeliefEngine(
        "p13", CommonPublicPerspective(), TRAIT_PROFILES["evidence_driven"]
    )
    conformist = BeliefEngine(
        "p13", CommonPublicPerspective(), TRAIT_PROFILES["conformist"]
    )

    evidence_driven.observe(ledger)
    conformist.observe(ledger)

    # One follows the published black, the other follows the room.
    assert evidence_driven.state.current_execution_target == "p9"
    assert conformist.state.current_execution_target == "p11"


def test_a_table_of_personalities_does_not_converge_on_one_candidate():
    ledger = PublicFactLedger(_contested_board())

    targets = set()
    for name, traits in TRAIT_PROFILES.items():
        engine = BeliefEngine(f"seat_{name}", CommonPublicPerspective(), traits)
        engine.observe(ledger)
        targets.add(engine.state.current_execution_target)

    assert len(targets) > 1


def test_a_sceptic_is_barely_moved_by_the_majority_but_never_opposes_it():
    ledger = PublicFactLedger(_contested_board())
    conformist = BeliefEngine("p13", CommonPublicPerspective(), TRAIT_PROFILES["conformist"])
    sceptic = BeliefEngine("p13", CommonPublicPerspective(), TRAIT_PROFILES["sceptic"])

    conformist.observe(ledger)
    sceptic.observe(ledger)

    # Contrarianism damps the crowd; it does not invert it into negative weight.
    assert (
        conformist.state.public_suspicion_scores["p11"]
        > sceptic.state.public_suspicion_scores["p11"]
    )
    assert sceptic.state.public_suspicion_scores["p11"] >= 0.0


def test_a_stubborn_player_needs_a_bigger_reason_to_switch():
    ledger = PublicFactLedger(_contested_board())
    nudge = EvidenceRecord(
        evidence_id="nudge:p12",
        subject_id="p12",
        category="misremembered_vote",
        source_event_ids=("nudge",),
        weight=1.6,
        explanation="やや強い違和感。",
    )

    flexible = BeliefEngine("p13", CommonPublicPerspective(), TRAIT_PROFILES["impulsive"])
    stubborn = BeliefEngine("p13", CommonPublicPerspective(), TRAIT_PROFILES["stubborn"])
    for engine in (flexible, stubborn):
        engine.observe(ledger)
    settled = {engine: engine.state.current_execution_target for engine in (flexible, stubborn)}
    for engine in (flexible, stubborn):
        engine.add_evidence(nudge)
        engine.recompute(ledger)

    # The same new lead flips the impulsive player and not the stubborn one.
    assert flexible.state.current_execution_target != settled[flexible]
    assert stubborn.state.current_execution_target == settled[stubborn]


def test_evidence_driven_players_swing_further_on_a_correction():
    state = _contested_board()
    state.vote_records.append(VoteRecord(voter_id="p0", target_id="p3", day=1, round=1))
    ledger = PublicFactLedger(state)
    correction = FactCorrection(
        kind=CorrectionKind.VOTE_TARGET,
        source_player_id="p0",
        subject_id="p0",
        day=1,
        asserted="p3",
        denied="p12",
    )
    misreading = EvidenceRecord(
        evidence_id="recalled:p0",
        subject_id="p0",
        category="misremembered_vote",
        source_event_ids=(vote_fact_id("p0", 1, 1, "p12"),),
        weight=2.0,
        explanation="p0はp12へ投票していた。",
    )

    swings = {}
    for name in ("evidence_driven", "conformist"):
        engine = BeliefEngine("p13", CommonPublicPerspective(), TRAIT_PROFILES[name])
        engine.observe(ledger)
        engine.add_evidence(misreading)
        engine.recompute(ledger)
        before = engine.state.public_suspicion_scores["p0"]
        engine.apply_correction(correction, ledger)
        swings[name] = before - engine.state.public_suspicion_scores["p0"]

    assert swings["evidence_driven"] > swings["conformist"] > 0


def test_traits_cannot_reorder_what_the_rules_have_settled():
    state = _contested_board()
    boards.kill_first_victim(state, "p3")
    ledger = PublicFactLedger(state)
    solver = build_solver(ObservationSet.from_state(state), CommonPublicPerspective())
    hunch = EvidenceRecord(
        evidence_id="hunch:p3",
        subject_id="p3",
        category="misremembered_vote",
        source_event_ids=("hunch",),
        weight=50.0,
        explanation="強い勘。",
    )

    for traits in TRAIT_PROFILES.values():
        engine = BeliefEngine("p13", CommonPublicPerspective(), traits)
        engine.add_evidence(hunch)
        engine.observe(ledger, solver, ObservationSet.from_state(state))
        # Traits scale the soft hunch all they like; the rules already settled
        # that p3 is not a wolf, and no profile can put them on the block.
        assert engine.state.certainty_of("p3") is RoleCertainty.EXCLUDED
        assert engine.state.current_execution_target != "p3"


def test_trait_assignment_is_reproducible_for_a_seed():
    seats = [f"p{i}" for i in range(17)]

    first = assign_traits(seats, seed=7)
    second = assign_traits(seats, seed=7)
    other = assign_traits(seats, seed=8)

    assert first == second
    assert len({id(profile) for profile in first.values()}) >= 1
    assert first != other


# -- ranking in words, not percentages --


def test_hypotheses_are_banded_rather_than_scored_to_three_decimals():
    hypotheses = [
        RankedHypothesis("wolf:p9", "p9が人狼", 2.4, Certainty.POSSIBLE),
        RankedHypothesis("wolf:p11", "p11が人狼", 2.1, Certainty.POSSIBLE),
        RankedHypothesis("wolf:p2", "p2が人狼", 0.5, Certainty.POSSIBLE),
        RankedHypothesis("wolf:p7", "p7が人狼", 0.0, Certainty.POSSIBLE),
        RankedHypothesis("wolf:p3", "p3が人狼", 9.9, Certainty.IMPOSSIBLE),
    ]

    views = rank_hypotheses(hypotheses, CognitiveTraits())
    by_id = {view.hypothesis.hypothesis_id: view.rank for view in views}

    assert by_id["wolf:p9"] is HypothesisRank.MAIN
    assert by_id["wolf:p11"] is HypothesisRank.STRONG_ALTERNATIVE
    assert by_id["wolf:p2"] is HypothesisRank.THIN
    assert by_id["wolf:p7"] is HypothesisRank.LOGICAL_ONLY
    # Excluded by the rules, however much soft weight it carries.
    assert by_id["wolf:p3"] is HypothesisRank.IMPOSSIBLE
    assert "本線: p9が人狼" in summarise(views)
    assert "%" not in summarise(views)


def test_a_sceptic_keeps_more_alternatives_under_serious_review():
    hypotheses = [
        RankedHypothesis("wolf:p9", "p9が人狼", 2.4, Certainty.POSSIBLE),
        RankedHypothesis("wolf:p11", "p11が人狼", 1.5, Certainty.POSSIBLE),
    ]

    plain = rank_hypotheses(hypotheses, TRAIT_PROFILES["conformist"])
    sceptic = rank_hypotheses(hypotheses, TRAIT_PROFILES["sceptic"])

    assert plain[1].rank is HypothesisRank.THIN
    assert sceptic[1].rank is HypothesisRank.STRONG_ALTERNATIVE


def test_a_settled_hypothesis_is_always_the_main_line():
    hypotheses = [
        RankedHypothesis("wolf:p9", "p9が人狼", 0.1, Certainty.CERTAIN),
        RankedHypothesis("wolf:p11", "p11が人狼", 5.0, Certainty.POSSIBLE),
    ]

    views = {
        view.hypothesis.hypothesis_id: view.rank
        for view in rank_hypotheses(hypotheses, CognitiveTraits())
    }

    assert views["wolf:p9"] is HypothesisRank.MAIN


# -- the bluffer's two heads --


def test_a_wolfs_story_and_their_real_knowledge_are_separate_states():
    state = boards.deal(
        {"p1": RoleName.WEREWOLF, "p2": RoleName.WEREWOLF, "p3": RoleName.WEREWOLF}
    )
    boards.claim(state, "p1", RoleName.SEER)
    observations = ObservationSet.from_state(state)
    deception = deception_state_for("p1", observations, claimed_role=RoleName.SEER)

    story = story_solver(deception, observations)
    private = private_solver(deception, observations)

    assert deception.is_bluffing is True
    assert deception.ally_ids == ("p2", "p3")
    # The story knows only the claimed card; the private view knows the team.
    assert story.is_forced(has_role("p1", RoleName.SEER)) is True
    assert story.is_forced(has_role("p2", RoleName.WEREWOLF)) is False
    assert private.is_forced(has_role("p2", RoleName.WEREWOLF)) is True
    assert (
        deception.public_story.perspective_id
        != deception.private_perspective.perspective_id
    )


def test_a_madman_is_not_handed_the_wolves():
    state = boards.deal({"p1": RoleName.WEREWOLF, "p8": RoleName.MADMAN})
    boards.claim(state, "p8", RoleName.SEER)
    observations = ObservationSet.from_state(state)
    deception = deception_state_for("p8", observations, claimed_role=RoleName.SEER)

    private = private_solver(deception, observations)

    assert deception.ally_ids == ()
    assert private.certain_role("p8") is RoleName.MADMAN
    assert private.is_forced(has_role("p1", RoleName.WEREWOLF)) is False


def test_a_madmans_invented_black_can_land_on_a_real_wolf():
    state = boards.deal({"p1": RoleName.WEREWOLF, "p8": RoleName.MADMAN})
    boards.claim(state, "p8", RoleName.SEER)
    boards.verdict(state, "p8", "seer", "p1", True)
    observations = ObservationSet.from_state(state)
    deception = deception_state_for("p8", observations, claimed_role=RoleName.SEER)

    refreshed = refresh_story(deception, observations)

    # They guessed. The guess was right. Nothing about that is contradictory.
    assert refreshed.status is StoryStatus.INTACT
    assert story_solver(refreshed, observations).is_forced(
        has_role("p1", RoleName.WEREWOLF)
    )


def test_betrayal_is_priced_rather_than_forbidden():
    state = boards.deal(
        {
            "p1": RoleName.WEREWOLF,
            "p2": RoleName.WEREWOLF,
            "p3": RoleName.WEREWOLF,
        }
    )
    boards.claim(state, "p1", RoleName.SEER)
    boards.verdict(state, "p1", "seer", "p2", True)
    observations = ObservationSet.from_state(state)
    deception = deception_state_for("p1", observations, claimed_role=RoleName.SEER)

    refreshed = refresh_story(deception, observations)

    # Selling out a teammate stays logically available -- it is occasionally the
    # strongest play -- but it costs something a ranker has to overcome.
    assert refreshed.status is StoryStatus.INTACT
    assert deception.betrayal_cost("p2") > 0
    assert deception.betrayal_cost("p9") == 0


def test_a_story_that_outruns_the_wolf_count_collapses_with_its_reasons():
    state = boards.deal({"p1": RoleName.MADMAN}, day=4)
    boards.claim(state, "p1", RoleName.SEER, day=1)
    for day, target in enumerate(("p6", "p9", "p12", "p15"), start=1):
        boards.verdict(state, "p1", "seer", target, True, day=day)
    observations = ObservationSet.from_state(state)
    deception = deception_state_for("p1", observations, claimed_role=RoleName.SEER)

    refreshed = refresh_story(deception, observations)

    assert refreshed.status is StoryStatus.COLLAPSED
    assert refreshed.has_collapsed is True
    assert "role_count:werewolf" in refreshed.collapse_constraint_ids
    assert any("人狼はちょうど3人です。" in text for text in refreshed.collapse_reasons)


def test_a_seat_with_no_story_has_nothing_to_collapse():
    state = boards.deal({"p1": RoleName.WEREWOLF})
    observations = ObservationSet.from_state(state)

    deception = deception_state_for("p1", observations)

    assert deception.is_bluffing is False
    assert refresh_story(deception, observations).status is StoryStatus.INTACT
