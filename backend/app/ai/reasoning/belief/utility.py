"""What a seat wants, as opposed to what a seat knows.

Knowing someone is a werewolf and wanting them executed are the same thing for a
villager and opposite things for a werewolf. Collapsing the two into one
"suspicion" number is how a wolf ends up voting for its own team: the solver
tells it, correctly and privately, that its partners are wolves, and a village
utility function reads that as "execute them".

So certainty and desire are separate. `RoleCertainty` is what the rules and the
seat's own cards have settled. The utilities below turn that -- plus what the
table can publicly argue -- into a preference, and they differ by faction
because the win conditions do.

Betrayal is never forbidden. It is priced, so it happens when it pays and not by
accident.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from app.engine.roles import ROLE_DEFINITIONS, RoleName, Team


class RoleCertainty(StrEnum):
    """What a seat privately knows about another seat holding a given role."""

    CONFIRMED = "confirmed"
    EXCLUDED = "excluded"
    UNKNOWN = "unknown"


# -- weights, collected so the whole scale is readable in one place --

# Village
KNOWN_WOLF_BONUS = 10.0
CLEARED_PENALTY = 6.0
FOX_SUSPICION_WEIGHT = 0.8
POWER_ROLE_LOSS_PENALTY = 2.0

# Werewolf
ALLY_PROTECTION = 12.0
VILLAGE_TRUST_BONUS = 2.0
RIDE_THE_WAVE_WEIGHT = 0.6
THREAT_WEIGHT = 1.5

# Madman: pushes the village into misexecutions without knowing who the wolves are.
MADMAN_TRUSTED_TARGET_BONUS = 2.0

# Fox: survive. Thin both sides, and above all stay off the block.
FOX_SELF_PRESSURE_RELIEF = 3.0

# Night
ALREADY_DIVINED_EXCLUSION = True
CLAIM_VERIFICATION_VALUE = 1.5
DOOMED_TARGET_PENALTY = 2.0
GUARD_POWER_ROLE_VALUE = 3.0
GUARD_ATTACK_RISK_WEIGHT = 1.0
ATTACK_THREAT_WEIGHT = 3.0
ATTACK_FOX_PENALTY = 4.0
ATTACK_GUARD_RISK_PENALTY = 1.5


@dataclass(frozen=True)
class UtilityInputs:
    """Everything a preference is computed from, gathered once per seat."""

    actor_id: str
    actor_role: RoleName | None
    ally_ids: frozenset[str]
    wolf_certainty: Mapping[str, RoleCertainty]
    public_suspicion: Mapping[str, float]
    fox_suspicion: Mapping[str, float]
    claim_trust: Mapping[str, float]
    claimed_roles: Mapping[str, RoleName]
    alive_ids: frozenset[str]
    already_divined: frozenset[str] = frozenset()

    @property
    def actor_team(self) -> Team | None:
        if self.actor_role is None:
            return None
        return ROLE_DEFINITIONS[self.actor_role].team


def execution_utility(inputs: UtilityInputs, target_id: str) -> float:
    """How much this seat wants that seat executed today.

    Dispatched by faction because the win conditions differ. A shared
    "suspicion" ranking cannot express that a wolf's certainty about a partner
    is a reason to protect them.
    """
    if inputs.actor_role is RoleName.WEREWOLF:
        return _wolf_execution_utility(inputs, target_id)
    if inputs.actor_role is RoleName.MADMAN:
        return _madman_execution_utility(inputs, target_id)
    if inputs.actor_role is RoleName.FOX:
        return _fox_execution_utility(inputs, target_id)
    return _village_execution_utility(inputs, target_id)


def _village_execution_utility(inputs: UtilityInputs, target_id: str) -> float:
    certainty = inputs.wolf_certainty.get(target_id, RoleCertainty.UNKNOWN)
    score = inputs.public_suspicion.get(target_id, 0.0)
    score += FOX_SUSPICION_WEIGHT * inputs.fox_suspicion.get(target_id, 0.0)
    if certainty is RoleCertainty.CONFIRMED:
        score += KNOWN_WOLF_BONUS
    elif certainty is RoleCertainty.EXCLUDED:
        # Not a wolf, so the rope is spent for nothing -- unless fox suspicion
        # is high enough to be worth it, which the weight above still allows.
        score -= CLEARED_PENALTY
    score -= POWER_ROLE_LOSS_PENALTY * max(0.0, inputs.claim_trust.get(target_id, 0.0))
    return score


def _wolf_execution_utility(inputs: UtilityInputs, target_id: str) -> float:
    if target_id in inputs.ally_ids:
        # Betrayal is available, not free. It has to beat a real teammate's
        # worth of value before it is chosen.
        return -ALLY_PROTECTION + RIDE_THE_WAVE_WEIGHT * inputs.public_suspicion.get(
            target_id, 0.0
        )
    score = RIDE_THE_WAVE_WEIGHT * inputs.public_suspicion.get(target_id, 0.0)
    certainty = inputs.wolf_certainty.get(target_id, RoleCertainty.UNKNOWN)
    if certainty is RoleCertainty.EXCLUDED:
        # Privately known not to be a wolf, so executing them costs the village
        # a body and costs the wolves nothing.
        score += VILLAGE_TRUST_BONUS
    score += THREAT_WEIGHT * _village_threat(inputs, target_id)
    return score


def _madman_execution_utility(inputs: UtilityInputs, target_id: str) -> float:
    """The madman wants misexecutions and does not know where the wolves are.

    Pushing at the seats the table trusts is the whole job; the inverted
    suspicion term is what makes that a preference rather than noise.
    """
    score = -inputs.public_suspicion.get(target_id, 0.0)
    score += MADMAN_TRUSTED_TARGET_BONUS * max(
        0.0, inputs.claim_trust.get(target_id, 0.0)
    )
    return score


def _fox_execution_utility(inputs: UtilityInputs, target_id: str) -> float:
    """Survive. Thin both sides, and above all keep the block pointed elsewhere."""
    score = inputs.public_suspicion.get(target_id, 0.0)
    score += THREAT_WEIGHT * _village_threat(inputs, target_id)
    if target_id != inputs.actor_id:
        score += FOX_SELF_PRESSURE_RELIEF * max(
            0.0, inputs.public_suspicion.get(inputs.actor_id, 0.0)
        )
    return score


def _village_threat(inputs: UtilityInputs, target_id: str) -> float:
    """How dangerous this seat is to the werewolf team.

    A trusted seer or medium claim is the threat; an untrusted one is a gift.
    """
    claimed = inputs.claimed_roles.get(target_id)
    if claimed not in (RoleName.SEER, RoleName.MEDIUM, RoleName.FREEMASON):
        return 0.0
    return max(0.0, 1.0 + inputs.claim_trust.get(target_id, 0.0))


# -- night actions, one preference each --


def divine_utility(inputs: UtilityInputs, target_id: str) -> float:
    """Information, not suspicion.

    Looking again at someone you already divined tells you nothing, and looking
    at someone the table is about to execute buys a result you were going to get
    from the medium anyway.
    """
    if ALREADY_DIVINED_EXCLUSION and target_id in inputs.already_divined:
        return float("-inf")
    score = inputs.public_suspicion.get(target_id, 0.0)
    score += FOX_SUSPICION_WEIGHT * inputs.fox_suspicion.get(target_id, 0.0)
    claimed = inputs.claimed_roles.get(target_id)
    if claimed in (RoleName.SEER, RoleName.MEDIUM):
        # Settling a contested claim is worth more than one more grey.
        score += CLAIM_VERIFICATION_VALUE
    if inputs.wolf_certainty.get(target_id) is RoleCertainty.EXCLUDED:
        score -= CLEARED_PENALTY
    return score


def guard_utility(inputs: UtilityInputs, target_id: str) -> float:
    """Cover what the village cannot afford to lose, not merely who looks safe.

    "Guard the seat with the lowest suspicion" protects a random villager while
    the seer dies, which is the opposite of the role's job.
    """
    if target_id == inputs.actor_id:
        return float("-inf")
    score = 0.0
    claimed = inputs.claimed_roles.get(target_id)
    if claimed in (RoleName.SEER, RoleName.MEDIUM, RoleName.FREEMASON):
        score += GUARD_POWER_ROLE_VALUE * max(
            0.5, 1.0 + inputs.claim_trust.get(target_id, 0.0)
        )
    # Whoever the wolves most want dead is whoever most needs covering.
    score += GUARD_ATTACK_RISK_WEIGHT * _village_threat(inputs, target_id)
    if inputs.wolf_certainty.get(target_id) is RoleCertainty.CONFIRMED:
        score -= KNOWN_WOLF_BONUS
    score -= inputs.public_suspicion.get(target_id, 0.0)
    return score


def attack_utility(inputs: UtilityInputs, target_id: str) -> float:
    """Kill what threatens the team, which is not the same as what looks wolfish.

    Suspicion is the village's currency. Reusing it here makes the wolves bite
    whoever the village already distrusts -- removing their own best cover.
    """
    if target_id in inputs.ally_ids or target_id == inputs.actor_id:
        return float("-inf")
    score = ATTACK_THREAT_WEIGHT * _village_threat(inputs, target_id)
    if inputs.wolf_certainty.get(target_id) is RoleCertainty.CONFIRMED:
        # Another known wolf: never a target, whatever the table thinks.
        return float("-inf")
    score -= ATTACK_FOX_PENALTY * inputs.fox_suspicion.get(target_id, 0.0)
    # A heavily guarded-looking seat wastes the night.
    score -= ATTACK_GUARD_RISK_PENALTY * max(
        0.0, inputs.claim_trust.get(target_id, 0.0)
    ) * (1.0 if inputs.claimed_roles.get(target_id) is RoleName.HUNTER else 0.0)
    # Leaving the table's current favourite alive keeps the misexecution alive.
    score -= RIDE_THE_WAVE_WEIGHT * inputs.public_suspicion.get(target_id, 0.0)
    return score


def rank_by(
    utility: Mapping[str, float],
    candidates: Sequence[str],
    tiebreaker: Callable[[str], float],
) -> list[str]:
    """Highest utility first, with an externally supplied secondary key."""
    return sorted(
        candidates,
        key=lambda pid: (-utility.get(pid, 0.0), tiebreaker(pid), pid),
    )
