"""A bluffer's two heads: what they know, and what they are telling the table.

A wolf reasoning about the board and a wolf defending a fake seer claim are
answering different questions from different information, and conflating them
breaks the bluff in both directions -- the story starts leaning on facts only
the wolf team has, and the wolf starts believing their own account.

So a deceiver carries two perspectives at once. The private one is their real
seat; the public one knows only the role being claimed. They are separate
instances with separate ids, and the story is never allowed to read the team.

The madman gets the same structure and less information: their private view is
just their own card. Not knowing where the wolves are is the whole shape of the
role, and it is why their invented black can land on a real wolf by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import (
    ClaimedStoryPerspective,
    Perspective,
    PlayerPrivatePerspective,
)
from app.ai.reasoning.solver import (
    AccurateTimeline,
    Assumption,
    HonestResults,
    Hypothesis,
    RoleSolver,
    SolverCache,
    build_solver,
)
from app.engine.roles import RoleName

# Selling out a teammate is legal and sometimes correct, so it is priced, never
# forbidden. The number is a strategy preference; the solver stays neutral.
BETRAYAL_COST = 2.5


class StoryStatus(StrEnum):
    INTACT = "intact"
    STRAINED = "strained"
    COLLAPSED = "collapsed"


@dataclass
class DeceptionState:
    """One deceiver's private view, public story, and how the story is holding."""

    player_id: str
    claimed_role: RoleName | None
    private_perspective: Perspective
    public_story: Perspective | None = None
    status: StoryStatus = StoryStatus.INTACT
    collapse_reasons: tuple[str, ...] = ()
    collapse_constraint_ids: tuple[str, ...] = ()
    ally_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_bluffing(self) -> bool:
        return self.public_story is not None

    @property
    def has_collapsed(self) -> bool:
        return self.status is StoryStatus.COLLAPSED

    def betrayal_cost(self, target_id: str) -> float:
        """What throwing this player under the bus costs *strategically*.

        Zero for anyone outside the team. Naming a teammate stays available --
        it is occasionally the strongest play -- but it should not be chosen by
        accident, so it carries a price the ranker has to overcome.
        """
        return BETRAYAL_COST if target_id in self.ally_ids else 0.0


def deception_state_for(
    player_id: str,
    observations: ObservationSet,
    *,
    claimed_role: RoleName | None = None,
) -> DeceptionState:
    """Build the two-headed state for one seat.

    Allies come from the seat's own knowledge, so the madman -- who has none --
    gets an empty list rather than the wolf roster.
    """
    private = PlayerPrivatePerspective(player_id)
    knowledge = observations.seat_knowledge(player_id)
    story = (
        ClaimedStoryPerspective(player_id, claimed_role)
        if claimed_role is not None
        else None
    )
    return DeceptionState(
        player_id=player_id,
        claimed_role=claimed_role,
        private_perspective=private,
        public_story=story,
        ally_ids=knowledge.ally_ids,
    )


def refresh_story(
    state: DeceptionState,
    observations: ObservationSet,
    *,
    assumptions: tuple[Assumption, ...] = (),
    cache: SolverCache | None = None,
) -> DeceptionState:
    """Re-check the public story against the board and record how it fared.

    A story that has gone UNSAT is not a story any more -- the table can derive
    the contradiction too -- so the state moves to COLLAPSED carrying the
    constraint ids that killed it, which is what a change of plan needs to be
    based on.
    """
    if state.public_story is None:
        return state
    solver = story_solver(state, observations, assumptions=assumptions, cache=cache)
    result = solver.explain_contradiction()
    if not result.is_contradictory:
        state.status = StoryStatus.INTACT
        state.collapse_reasons = ()
        state.collapse_constraint_ids = ()
        return state
    state.status = StoryStatus.COLLAPSED
    state.collapse_reasons = result.explanations
    state.collapse_constraint_ids = result.constraint_ids
    return state


def story_solver(
    state: DeceptionState,
    observations: ObservationSet,
    *,
    assumptions: tuple[Assumption, ...] = (),
    cache: SolverCache | None = None,
) -> RoleSolver:
    """The solver the bluffer uses to sanity-check their own account.

    Defaults to supposing their published verdicts are true, because that is
    what the story asserts. It never gets the private perspective.
    """
    assert state.public_story is not None
    combined = tuple(
        {item.key: item for item in (
            HonestResults(state.player_id), AccurateTimeline(state.player_id), *assumptions
        )}.values()
    )
    return build_solver(
        observations, state.public_story, cache=cache, assumptions=combined
    )


def private_solver(
    state: DeceptionState, observations: ObservationSet, *, cache: SolverCache | None = None
) -> RoleSolver:
    """What the deceiver actually knows. Never handed to the story."""
    return build_solver(observations, state.private_perspective, cache=cache)


def story_is_possible(
    state: DeceptionState, observations: ObservationSet, *, cache: SolverCache | None = None
) -> bool:
    if state.public_story is None:
        return True
    return story_solver(state, observations, cache=cache).is_possible(Hypothesis())
