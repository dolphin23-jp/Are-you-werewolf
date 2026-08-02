"""Reasoning over what was claimed, without believing it.

A public CO is an observation, never a constraint. It only bites the world model
when a caller supposes it is genuine -- and supposing that is separate again
from supposing the published verdicts are correct.
"""

from __future__ import annotations

from app.ai.reasoning.perspectives import (
    ClaimedStoryPerspective,
    CommonPublicPerspective,
    PlayerPrivatePerspective,
)
from app.ai.reasoning.solver import (
    Certainty,
    CompleteResultDisclosure,
    GenuineClaim,
    HonestResults,
    Hypothesis,
    NoLatentClaim,
    build_solver,
    has_role,
    not_role,
)
from app.ai.reasoning.solver.assumptions import expand
from app.engine.roles import RoleName
from tests.ai.reasoning.solver import boards


def _public(state):  # type: ignore[no-untyped-def]
    return build_solver(boards.observe(state), CommonPublicPerspective())


# -- a claim is an observation, not a constraint --


def test_a_public_co_alone_settles_nothing():
    state = boards.deal({"p3": RoleName.VILLAGER})
    boards.claim(state, "p3", RoleName.SEER)
    solver = _public(state)

    # The board records that p3 said it. Nothing about the world follows.
    assert solver.observations.claimed_role_of("p3") is RoleName.SEER
    assert solver.is_forced(has_role("p3", RoleName.SEER)) is False
    assert solver.is_possible(has_role("p3", RoleName.WEREWOLF)) is True


def test_two_seer_claims_are_not_a_contradiction_by_themselves():
    state = boards.deal({})
    boards.claim(state, "p3", RoleName.SEER)
    boards.claim(state, "p7", RoleName.SEER)
    solver = _public(state)

    # One of them is lying -- which is ordinary werewolf, not an impossibility.
    assert solver.is_possible(Hypothesis()) is True
    assert solver.assuming(GenuineClaim("p3", RoleName.SEER)).is_possible(Hypothesis())


def test_supposing_both_seer_claims_genuine_is_impossible_and_says_why():
    state = boards.deal({})
    boards.claim(state, "p3", RoleName.SEER)
    boards.claim(state, "p7", RoleName.SEER)

    solver = _public(state).assuming(
        GenuineClaim("p3", RoleName.SEER), GenuineClaim("p7", RoleName.SEER)
    )
    result = solver.explain_contradiction()

    assert result.is_contradictory is True
    assert "role_count:seer" in result.constraint_ids
    assert "genuine:p3:seer" in result.constraint_ids
    assert "genuine:p7:seer" in result.constraint_ids
    assert any("占い師はちょうど1人です。" in text for text in result.explanations)


# -- latent roles --


def test_a_lone_seer_co_is_confirmed_only_once_lurking_is_ruled_out():
    state = boards.deal({})
    boards.claim(state, "p3", RoleName.SEER)
    solver = _public(state)

    # Allowing a quiet real seer, one CO proves nothing.
    assert solver.is_forced(has_role("p3", RoleName.SEER)) is False
    assert solver.is_possible(has_role("p3", RoleName.WEREWOLF)) is True

    # Assume nobody holds seer without saying so, and the lone claim is settled.
    strict = solver.assuming(NoLatentClaim(RoleName.SEER))
    assert strict.is_forced(has_role("p3", RoleName.SEER)) is True
    assert strict.certain_role("p3") is RoleName.SEER


def test_ruling_out_lurkers_is_an_assumption_not_a_rule():
    state = boards.deal({})
    boards.claim(state, "p3", RoleName.SEER)
    base = _public(state)

    strict = base.assuming(NoLatentClaim(RoleName.SEER))

    # The supposition lives on the derived solver only; the board itself never
    # acquired a rule saying the seer must CO.
    assert strict.is_forced(has_role("p3", RoleName.SEER)) is True
    assert base.is_forced(has_role("p3", RoleName.SEER)) is False
    assert base.assumptions == ()


def test_a_story_reports_which_roles_it_needs_hiding():
    state = boards.deal({})
    boards.claim(state, "p3", RoleName.SEER)
    solver = _public(state)

    # Nobody has claimed medium, so any world has a latent medium.
    required = solver.required_latent_roles()

    assert RoleName.MEDIUM in required
    assert RoleName.WEREWOLF in required
    # Seer is not required to be latent: p3's claim could be the real one.
    assert RoleName.SEER not in required


# -- honest results --


def test_a_genuine_seers_black_lands_on_a_real_wolf():
    state = boards.deal({"p3": RoleName.SEER, "p9": RoleName.WEREWOLF})
    boards.claim(state, "p3", RoleName.SEER)
    boards.verdict(state, "p3", "seer", "p9", True)

    solver = _public(state).assuming(
        GenuineClaim("p3", RoleName.SEER), HonestResults("p3")
    )

    assert solver.is_forced(has_role("p9", RoleName.WEREWOLF)) is True


def test_a_genuine_seers_white_rules_the_target_out_as_a_wolf():
    state = boards.deal({"p3": RoleName.SEER})
    boards.claim(state, "p3", RoleName.SEER)
    boards.verdict(state, "p3", "seer", "p9", False)

    solver = _public(state).assuming(
        GenuineClaim("p3", RoleName.SEER), HonestResults("p3")
    )

    assert solver.is_forced(not_role("p9", RoleName.WEREWOLF)) is True
    # A white is not a clean bill of health: the fox is still open.
    assert solver.is_possible(has_role("p9", RoleName.FOX)) is True


def test_a_genuine_mediums_verdicts_behave_the_same_way():
    state = boards.deal({"p4": RoleName.MEDIUM})
    boards.execute(state, "p8", day=1)
    boards.execute(state, "p9", day=2)
    boards.claim(state, "p4", RoleName.MEDIUM, day=2)
    boards.verdict(state, "p4", "medium", "p8", True, day=2)
    boards.verdict(state, "p4", "medium", "p9", False, day=3)
    state.day = 3

    solver = _public(state).assuming(
        GenuineClaim("p4", RoleName.MEDIUM), HonestResults("p4")
    )

    assert solver.is_forced(has_role("p8", RoleName.WEREWOLF)) is True
    assert solver.is_forced(not_role("p9", RoleName.WEREWOLF)) is True


def test_a_madmans_invented_black_can_still_land_on_a_wolf():
    state = boards.deal({"p6": RoleName.MADMAN, "p9": RoleName.WEREWOLF})
    boards.claim(state, "p6", RoleName.SEER)
    boards.verdict(state, "p6", "seer", "p9", True)
    solver = _public(state)

    # Honest results without a genuine claim: the fake seer guessed right. That
    # world has to stay reachable, which is why the two are separate suppositions.
    lucky = solver.assuming(HonestResults("p6"))
    assert lucky.is_possible(has_role("p6", RoleName.MADMAN)) is True
    assert lucky.is_forced(has_role("p9", RoleName.WEREWOLF)) is True
    assert lucky.is_possible(has_role("p6", RoleName.SEER)) is True


def test_disclosure_and_honesty_are_separable_from_holding_the_role():
    state = boards.deal({}, day=3)
    boards.claim(state, "p3", RoleName.SEER, day=1)
    boards.verdict(state, "p3", "seer", "p9", False, day=1)
    solver = _public(state)

    # Day 3 with one published result: a real seer would hold three.
    assert solver.assuming(GenuineClaim("p3", RoleName.SEER)).is_possible(Hypothesis())
    assert solver.assuming(CompleteResultDisclosure("p3")).is_possible(Hypothesis()) is False

    result = solver.assuming(CompleteResultDisclosure("p3")).explain_contradiction()
    assert "complete:p3:count" in result.constraint_ids
    assert any("3件の結果を持つはず" in text for text in result.explanations)


# -- conditional consequences --


def test_white_pressure_forces_the_remaining_seats_to_be_wolves():
    # Everyone except p14, p15 and p16 is publicly accounted for as a non-wolf,
    # so the three wolf slots have nowhere else to go.
    whited = [f"p{i}" for i in range(1, 14)]
    state = boards.deal({"p0": RoleName.SEER})
    boards.claim(state, "p0", RoleName.SEER)
    for target in whited:
        boards.verdict(state, "p0", "seer", target, False)

    solver = _public(state).assuming(GenuineClaim("p0", RoleName.SEER), HonestResults("p0"))

    for seat in ("p14", "p15", "p16"):
        assert solver.is_forced(has_role(seat, RoleName.WEREWOLF)) is True


def test_the_same_deduction_is_only_conditional_from_the_public_view():
    whited = [f"p{i}" for i in range(1, 14)]
    state = boards.deal({"p0": RoleName.SEER})
    boards.claim(state, "p0", RoleName.SEER)
    for target in whited:
        boards.verdict(state, "p0", "seer", target, False)
    solver = _public(state)

    premise = Hypothesis(
        claims=tuple(
            item.constraint
            for item in expand(
                [GenuineClaim("p0", RoleName.SEER), HonestResults("p0")],
                solver.observations,
            )
        ),
        label="p0 is a genuine, honest seer",
    )

    # Unconditionally p14 is merely a candidate; under the premise it is settled.
    assert solver.assess(has_role("p14", RoleName.WEREWOLF)) is Certainty.POSSIBLE
    assert (
        solver.assess(has_role("p14", RoleName.WEREWOLF), given=premise)
        is Certainty.CONDITIONAL
    )


def test_a_private_view_can_refute_a_claim_the_table_cannot():
    whited = [f"p{i}" for i in range(1, 14)]
    villager_state = boards.deal({"p0": RoleName.SEER, "p14": RoleName.VILLAGER})
    wolf_state = boards.deal(
        {
            "p0": RoleName.SEER,
            "p14": RoleName.WEREWOLF,
            "p15": RoleName.WEREWOLF,
            "p16": RoleName.WEREWOLF,
        }
    )
    for state in (villager_state, wolf_state):
        boards.claim(state, "p0", RoleName.SEER)
        for target in whited:
            boards.verdict(state, "p0", "seer", target, False)

    def under_story(state):  # type: ignore[no-untyped-def]
        return build_solver(
            boards.observe(state), PlayerPrivatePerspective("p14")
        ).assuming(GenuineClaim("p0", RoleName.SEER), HonestResults("p0"))

    # p14 knows their own card, so the story that makes them a wolf either
    # collapses or holds -- and only they can tell which.
    assert under_story(villager_state).is_possible(Hypothesis()) is False
    assert under_story(wolf_state).is_possible(Hypothesis()) is True


# -- bluffing --


def test_a_wolfs_public_story_never_carries_their_real_team():
    state = boards.deal(
        {"p2": RoleName.WEREWOLF, "p5": RoleName.WEREWOLF, "p8": RoleName.WEREWOLF}
    )
    boards.claim(state, "p2", RoleName.SEER)
    observations = boards.observe(state)

    story = build_solver(observations, ClaimedStoryPerspective("p2", RoleName.SEER))
    private = build_solver(observations, PlayerPrivatePerspective("p2"))

    assert story.is_forced(has_role("p2", RoleName.SEER)) is True
    # The story cannot see p5 and p8; the wolf's own view can.
    assert story.is_forced(has_role("p5", RoleName.WEREWOLF)) is False
    assert story.is_possible(has_role("p5", RoleName.VILLAGER)) is True
    assert private.is_forced(has_role("p5", RoleName.WEREWOLF)) is True
    assert story.perspective.perspective_id != private.perspective.perspective_id


def test_a_wolf_selling_out_a_teammate_is_logically_allowed():
    state = boards.deal({"p2": RoleName.WEREWOLF, "p5": RoleName.WEREWOLF})
    boards.claim(state, "p2", RoleName.SEER)
    boards.verdict(state, "p2", "seer", "p5", True)

    story = build_solver(
        boards.observe(state), ClaimedStoryPerspective("p2", RoleName.SEER)
    ).assuming(HonestResults("p2"))

    # Nothing in the rules forbids throwing your own partner under the bus; it
    # is a strategy choice, so the solver must not rule it out.
    assert story.is_possible(Hypothesis()) is True
    assert story.is_forced(has_role("p5", RoleName.WEREWOLF)) is True


def test_a_bluff_collapses_once_its_verdicts_outrun_the_wolf_count():
    state = boards.deal({"p2": RoleName.MADMAN})
    boards.claim(state, "p2", RoleName.SEER)
    # Four blacks over four days: one more wolf than the game contains.
    for day, target in enumerate(("p6", "p9", "p12", "p15"), start=1):
        boards.verdict(state, "p2", "seer", target, True, day=day)
    state.day = 4

    story = build_solver(
        boards.observe(state), ClaimedStoryPerspective("p2", RoleName.SEER)
    ).assuming(HonestResults("p2"))
    result = story.explain_contradiction()

    assert story.is_possible(Hypothesis()) is False
    assert result.is_contradictory is True
    assert "role_count:werewolf" in result.constraint_ids
    assert sum(cid.startswith("honest:p2") for cid in result.constraint_ids) == 4
    assert any("人狼はちょうど3人です。" in text for text in result.explanations)


def test_a_story_that_holds_still_needs_the_unclaimed_roles_hidden():
    state = boards.deal({})
    boards.claim(state, "p2", RoleName.SEER)
    boards.claim(state, "p5", RoleName.SEER)
    story = build_solver(
        boards.observe(state), ClaimedStoryPerspective("p2", RoleName.SEER)
    )

    required = story.required_latent_roles()

    # p2's story puts the seer card on p2, so the seer is accounted for -- but
    # nobody has claimed medium or hunter, so those must be sitting quiet.
    assert RoleName.SEER not in required
    assert RoleName.MEDIUM in required
    assert RoleName.HUNTER in required


def test_disproving_the_only_seer_claimant_forces_a_latent_seer():
    state = boards.deal({}, day=2)
    boards.execute(state, "p2", day=1)
    boards.claim(state, "p2", RoleName.SEER, day=1)
    boards.claim(state, "p4", RoleName.MEDIUM, day=2)
    boards.verdict(state, "p4", "medium", "p2", True, day=2)

    solver = _public(state).assuming(
        GenuineClaim("p4", RoleName.MEDIUM), HonestResults("p4")
    )

    # The medium's black makes the lone seer claimant a wolf, so the real seer
    # is still hiding -- a conclusion the table can act on.
    assert solver.is_forced(has_role("p2", RoleName.WEREWOLF)) is True
    assert RoleName.SEER in solver.required_latent_roles()


# -- first victim --


def test_a_freemasons_partner_knows_every_other_role_holder_survived():
    state = boards.deal({"p1": RoleName.FREEMASON, "p2": RoleName.FREEMASON})
    boards.kill_first_victim(state, "p1")
    solver = build_solver(boards.observe(state), PlayerPrivatePerspective("p2"))

    # The only death is the partner, and the partner's card is known, so every
    # power role is still at the table.
    for role in (RoleName.SEER, RoleName.MEDIUM, RoleName.HUNTER, RoleName.FOX):
        assert solver.is_forced(not_role("p1", role)) is True
        assert solver.is_possible(has_role("p1", role)) is False
    dead = [pid for pid, alive in solver.observations.alive.items() if not alive]
    assert dead == ["p1"]


def test_the_table_only_knows_the_first_victim_was_not_wolf_or_fox():
    state = boards.deal({"p1": RoleName.FREEMASON, "p2": RoleName.FREEMASON})
    boards.kill_first_victim(state, "p1")
    solver = _public(state)

    assert solver.is_possible(has_role("p1", RoleName.WEREWOLF)) is False
    assert solver.is_possible(has_role("p1", RoleName.FOX)) is False
    # Everything else about the victim is still open from the public view.
    assert solver.is_possible(has_role("p1", RoleName.SEER)) is True
    assert solver.is_possible(has_role("p1", RoleName.FREEMASON)) is True
