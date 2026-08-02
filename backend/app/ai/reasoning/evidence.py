"""Soft evidence: the layer where "plausible" lives, kept apart from "possible".

A rule module produces two very different things. Its hard constraints say what
the rules of the game permit; its soft evidence says what experienced play
suggests. Mixing them is how an AI ends up treating a hunch as a proof, or
refusing to consider a world that is merely unusual.

Nothing weighs anything yet -- deliberately. Committing to attack likelihoods
before the belief engine exists would bake guesses into the foundation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoftEvidence:
    """One reason to shift belief in a hypothesis, with where it came from.

    `log_weight` is additive in log-odds, so evidence composes without anyone
    having to invent a joint probability. `source_event_ids` is what lets the
    belief engine retract this later: when the fact behind it turns out to be
    wrong, the evidence that rested on it has to go too.
    """

    evidence_id: str
    hypothesis_id: str
    log_weight: float
    category: str
    source_event_ids: tuple[str, ...] = ()
    explanation: str = ""
