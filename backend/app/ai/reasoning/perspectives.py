"""Whose knowledge a deduction is being made from.

The perspective is the *only* gate between the observations and the solver. A
rule module never reads `ObservationSet.true_roles`; it asks the perspective
what is known, and the perspective decides. That is what keeps a village-side
deduction from quietly resting on the real assignment.

`TrueWorldPerspective` sees everything and is for tests, evaluation and
debugging. It is not reachable from `perspective_for`, the function AI code
uses, and it is flagged `debug_only` so a caller that ends up holding one can be
made to fail loudly rather than silently reason with the answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.reasoning.observations import (
    EMPTY_NIGHT_KNOWLEDGE,
    NightKnowledge,
    ObservationSet,
    PrivateDivineResult,
    PrivateMediumResult,
)
from app.engine.roles import RoleName


@dataclass(frozen=True)
class Perspective:
    """Base viewpoint: knows nothing beyond the public rules."""

    perspective_id: str
    viewer_id: str | None = None
    debug_only: bool = False

    def known_roles(self, observations: ObservationSet) -> dict[str, RoleName]:
        return {}

    def known_night_actions(self, observations: ObservationSet) -> NightKnowledge:
        """Night actions this viewpoint may reason from. Nothing, by default.

        The same gate as `known_roles`, for the same reason: the real guard and
        attack targets are the two facts a village-side deduction must never be
        able to reach around and pick up.
        """
        return EMPTY_NIGHT_KNOWLEDGE

    def known_divine_results(
        self, observations: ObservationSet
    ) -> tuple[PrivateDivineResult, ...]:
        """Ability verdicts this viewpoint holds. None, by default.

        An unpublished seer result is the strongest fact in the game, and the
        one a village-side deduction must never be able to reach for.
        """
        return ()

    def known_medium_results(
        self, observations: ObservationSet
    ) -> tuple[PrivateMediumResult, ...]:
        return ()


@dataclass(frozen=True)
class CommonPublicPerspective(Perspective):
    """What the whole table shares. No private card, not even the viewer's own."""

    perspective_id: str = "public"


@dataclass(frozen=True)
class GenericVillagerPerspective(Perspective):
    """A plain villager reasoning about the rest of the board.

    Used to answer "what would an ordinary villager in this seat be able to
    conclude", without consulting what the seat's card actually is.
    """

    def __init__(self, self_id: str) -> None:
        object.__setattr__(self, "perspective_id", f"generic_villager:{self_id}")
        object.__setattr__(self, "viewer_id", self_id)
        object.__setattr__(self, "debug_only", False)

    def known_roles(self, observations: ObservationSet) -> dict[str, RoleName]:
        assert self.viewer_id is not None
        return {self.viewer_id: RoleName.VILLAGER}


@dataclass(frozen=True)
class PlayerPrivatePerspective(Perspective):
    """One real seat, seeing exactly what that seat was dealt.

    Werewolves see their team, freemasons see their partner. The madman and the
    fox see only their own card -- their whole difficulty is that they must infer
    the rest like everybody else.
    """

    def __init__(self, player_id: str) -> None:
        object.__setattr__(self, "perspective_id", f"private:{player_id}")
        object.__setattr__(self, "viewer_id", player_id)
        object.__setattr__(self, "debug_only", False)

    def known_roles(self, observations: ObservationSet) -> dict[str, RoleName]:
        assert self.viewer_id is not None
        return observations.seat_knowledge(self.viewer_id).as_role_map()

    def known_night_actions(self, observations: ObservationSet) -> NightKnowledge:
        assert self.viewer_id is not None
        return observations.night_knowledge_for(self.viewer_id)

    def known_divine_results(
        self, observations: ObservationSet
    ) -> tuple[PrivateDivineResult, ...]:
        assert self.viewer_id is not None
        return observations.divine_results_of(self.viewer_id)

    def known_medium_results(
        self, observations: ObservationSet
    ) -> tuple[PrivateMediumResult, ...]:
        assert self.viewer_id is not None
        return observations.medium_results_of(self.viewer_id)


@dataclass(frozen=True)
class ClaimedStoryPerspective(Perspective):
    """The world as a bluffer is publicly telling it.

    A wolf's fake seer story has to hold together on its own terms: the only
    thing it may take for granted is the role being claimed. Their real card and
    their team are deliberately absent -- a story that quietly leans on knowing
    where the wolves are is not a story the table could ever accept, and would
    also let the bluffer "prove" things they have no public reason to believe.

    Always a separate instance from the same player's `PlayerPrivatePerspective`,
    with a distinct id, so the two can never share a cached deduction.
    """

    claimed_role: RoleName = RoleName.VILLAGER

    def __init__(self, actor_id: str, claimed_role: RoleName) -> None:
        object.__setattr__(self, "perspective_id", f"story:{actor_id}:{claimed_role.value}")
        object.__setattr__(self, "viewer_id", actor_id)
        object.__setattr__(self, "debug_only", False)
        object.__setattr__(self, "claimed_role", claimed_role)

    def known_roles(self, observations: ObservationSet) -> dict[str, RoleName]:
        assert self.viewer_id is not None
        return {self.viewer_id: self.claimed_role}

    def known_night_actions(self, observations: ObservationSet) -> NightKnowledge:
        """The nights the story says happened, not the ones that did.

        A fake seer's account of who they looked at is exactly the list of
        targets they published. Reasoning from it is how the bluffer finds out
        whether their own story survives contact with the corpses.
        """
        assert self.viewer_id is not None
        if self.claimed_role is not RoleName.SEER:
            return EMPTY_NIGHT_KNOWLEDGE
        return NightKnowledge(
            divines={
                verdict.source_night: verdict.target_id
                for verdict in observations.verdicts_by(self.viewer_id)
                if verdict.result_type == "seer"
            }
        )


@dataclass(frozen=True)
class TrueWorldPerspective(Perspective):
    """Omniscient. Tests, evaluation and debugging only -- see module docstring."""

    perspective_id: str = "true_world"
    debug_only: bool = True

    def known_roles(self, observations: ObservationSet) -> dict[str, RoleName]:
        return dict(observations.true_roles)

    def known_night_actions(self, observations: ObservationSet) -> NightKnowledge:
        return observations.all_night_knowledge()

    def known_divine_results(
        self, observations: ObservationSet
    ) -> tuple[PrivateDivineResult, ...]:
        return observations.divine_results

    def known_medium_results(
        self, observations: ObservationSet
    ) -> tuple[PrivateMediumResult, ...]:
        return observations.medium_results


class PerspectiveLeakError(RuntimeError):
    """Raised when an omniscient perspective reaches in-game reasoning."""


def perspective_for(player_id: str | None) -> Perspective:
    """The perspective AI code uses. Cannot produce an omniscient one."""
    if player_id is None:
        return CommonPublicPerspective()
    return PlayerPrivatePerspective(player_id)


def require_in_game(perspective: Perspective) -> Perspective:
    """Guard for anything that feeds a live player's decisions."""
    if perspective.debug_only:
        raise PerspectiveLeakError(
            f"{perspective.perspective_id} sees the true assignment and must not "
            "drive in-game reasoning"
        )
    return perspective
