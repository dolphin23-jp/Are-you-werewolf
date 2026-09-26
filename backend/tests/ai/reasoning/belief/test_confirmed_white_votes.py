"""Voting out a 確定白 is legal, and the table remembers who did it.

A wolf bloc plus the fox can carry a scattered gray day onto the freemason pair
with four or five ballots. Real tables rarely see it, and not because it is
forbidden: the ballots are public, and the voters are the next day's suspects.
These tests hold both halves -- the evidence against such a voter, and the
non-village seats pricing that in before they vote.
"""

from __future__ import annotations

from app.ai.reasoning.belief import (
    BeliefEngine,
    EvidenceVisibility,
    UtilityInputs,
    execution_utility,
)
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import CommonPublicPerspective, PlayerPrivatePerspective
from app.ai.reasoning.solver import build_solver
from app.engine.roles import RoleName
from app.engine.state import VoteRecord
from tests.ai.reasoning.solver import boards

ALL_SEATS = [f"p{i}" for i in range(17)]


def _board(day: int = 2):  # type: ignore[no-untyped-def]
    return boards.deal(
        {
            "p1": RoleName.WEREWOLF,
            "p2": RoleName.WEREWOLF,
            "p3": RoleName.WEREWOLF,
            "p4": RoleName.SEER,
            "p7": RoleName.FOX,
            "p8": RoleName.MADMAN,
            "p9": RoleName.FREEMASON,
            "p10": RoleName.FREEMASON,
        },
        day=day,
    )


def _vote(state, voter: str, target: str, day: int, round_number: int = 1) -> None:  # type: ignore[no-untyped-def]
    state.vote_records.append(
        VoteRecord(voter_id=voter, target_id=target, day=day, round=round_number)
    )


def _observe(engine: BeliefEngine, state, perspective) -> None:  # type: ignore[no-untyped-def]
    observations = ObservationSet.from_state(state)
    engine.observe(
        PublicFactLedger(state), build_solver(observations, perspective), observations
    )


def _observe_publicly(engine: BeliefEngine, state) -> None:  # type: ignore[no-untyped-def]
    _observe(engine, state, CommonPublicPerspective())


def _public_engine() -> BeliefEngine:
    return BeliefEngine("p12", CommonPublicPerspective())


def _confirmed_white_votes(engine: BeliefEngine, voter: str):  # type: ignore[no-untyped-def]
    return [
        record
        for record in engine.active_evidence()
        if record.subject_id == voter and record.evidence_id.startswith("vote_confirmed_white:")
    ]


# -- what the table holds as 確定白 --


def test_an_uncontested_freemason_pair_is_conventionally_confirmed():
    state = _board()
    boards.claim(state, "p9", RoleName.FREEMASON)
    boards.claim(state, "p10", RoleName.FREEMASON)

    assert PublicFactLedger(state).conventionally_confirmed_village_ids() == ("p10", "p9")


def test_a_lone_or_a_contested_freemason_claim_confirms_nobody():
    state = _board()
    boards.claim(state, "p9", RoleName.FREEMASON)
    assert PublicFactLedger(state).conventionally_confirmed_village_ids() == ()

    boards.claim(state, "p10", RoleName.FREEMASON)
    boards.claim(state, "p5", RoleName.FREEMASON)
    # Three claims for two seats: somebody is lying, so nobody is white yet.
    assert PublicFactLedger(state).conventionally_confirmed_village_ids() == ()


# -- the consequence: the voter becomes the suspect --


def test_a_ballot_on_the_freemason_pair_is_public_evidence_against_the_voter():
    state = _board()
    boards.claim(state, "p9", RoleName.FREEMASON)
    boards.claim(state, "p10", RoleName.FREEMASON)
    _vote(state, "p1", "p9", day=2)
    _vote(state, "p5", "p11", day=2)
    engine = _public_engine()

    _observe_publicly(engine, state)

    [record] = _confirmed_white_votes(engine, "p1")
    assert record.category == "voted_for_cleared"
    assert record.visibility is EvidenceVisibility.PUBLIC_ARGUMENT
    assert record.weight == 2.0
    assert "確定白" in record.explanation
    assert engine.state.public_suspicion_scores["p1"] == 2.0
    # A ballot on a gray seat is just a ballot.
    assert _confirmed_white_votes(engine, "p5") == []
    assert engine.state.public_suspicion_scores.get("p5", 0.0) == 0.0


def test_a_runoff_ballot_on_the_same_seat_is_the_same_act_not_a_second_one():
    state = _board()
    boards.claim(state, "p9", RoleName.FREEMASON)
    boards.claim(state, "p10", RoleName.FREEMASON)
    _vote(state, "p1", "p9", day=2, round_number=1)
    _vote(state, "p1", "p9", day=2, round_number=2)
    engine = _public_engine()

    _observe_publicly(engine, state)
    _observe_publicly(engine, state)

    assert len(_confirmed_white_votes(engine, "p1")) == 1
    assert engine.state.public_suspicion_scores["p1"] == 2.0


def test_a_pair_completed_after_the_ballot_does_not_convict_it_in_hindsight():
    """p10 claims only after the first round -- a human can still speak during
    the runoff. The round-1 ballot on p9 was cast on a single, unconfirmed
    claim, and it stays judged that way however the day ends."""
    state = _board()
    boards.claim(state, "p9", RoleName.FREEMASON, day=2)
    _vote(state, "p1", "p9", day=2)
    engine = _public_engine()
    _observe_publicly(engine, state)

    boards.claim(state, "p10", RoleName.FREEMASON, day=2)
    _observe_publicly(engine, state)

    assert _confirmed_white_votes(engine, "p1") == []
    assert engine.state.public_suspicion_scores.get("p1", 0.0) == 0.0


# -- the anticipation: non-village seats price it in --


def _inputs(role: RoleName, actor: str, **overrides) -> UtilityInputs:  # type: ignore[no-untyped-def]
    fields = {
        "actor_id": actor,
        "actor_role": role,
        "ally_ids": frozenset({"p2", "p3"}) if role is RoleName.WEREWOLF else frozenset(),
        "wolf_certainty": {},
        "public_suspicion": {"p11": 1.0},
        "fox_suspicion": {},
        # A trusted freemason is exactly what every non-village seat wants gone.
        "claim_trust": {"p9": 1.0},
        "claimed_roles": {"p9": RoleName.FREEMASON},
        "alive_ids": frozenset(ALL_SEATS),
    }
    fields.update(overrides)
    return UtilityInputs(**fields)


def test_every_non_village_seat_prefers_a_gray_to_a_confirmed_white():
    for role, actor in (
        (RoleName.WEREWOLF, "p1"),
        (RoleName.MADMAN, "p8"),
        (RoleName.FOX, "p7"),
    ):
        unconfirmed = _inputs(role, actor)
        confirmed = _inputs(role, actor, confirmed_white=frozenset({"p9", "p10"}))

        # The pull toward the freemason is real, which is why it has to be priced.
        assert execution_utility(unconfirmed, "p9") > execution_utility(unconfirmed, "p11"), role
        assert execution_utility(confirmed, "p9") < execution_utility(confirmed, "p11"), role


def test_the_exposure_is_a_price_not_a_ban():
    """Enough table-wide suspicion on a confirmed white still carries a wolf
    along -- then the ballot hides in the crowd, and the evidence falls on
    everyone who cast it."""
    inputs = _inputs(
        RoleName.WEREWOLF,
        "p1",
        confirmed_white=frozenset({"p9", "p10"}),
        public_suspicion={"p9": 20.0, "p11": 1.0},
    )

    assert execution_utility(inputs, "p9") > execution_utility(inputs, "p11")


def test_a_village_seat_does_not_spend_the_rope_on_a_confirmed_white():
    gray = _inputs(RoleName.VILLAGER, "p12", claim_trust={}, claimed_roles={})
    confirmed = _inputs(
        RoleName.VILLAGER,
        "p12",
        claim_trust={},
        claimed_roles={},
        confirmed_white=frozenset({"p9", "p10"}),
    )

    assert execution_utility(confirmed, "p9") < execution_utility(gray, "p9")


def test_the_wolf_engine_steers_off_the_claimed_freemason_pair():
    state = _board()
    boards.claim(state, "p9", RoleName.FREEMASON)
    boards.claim(state, "p10", RoleName.FREEMASON)
    perspective = PlayerPrivatePerspective("p1")
    engine = BeliefEngine("p1", perspective)

    _observe(engine, state, perspective)

    assert engine.state.current_execution_target not in ("p9", "p10")
    for pid in ("p9", "p10"):
        assert engine.state.execution_utility_scores[pid] < 0
