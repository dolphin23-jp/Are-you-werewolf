"""Records what every AI player actually said and decided, with the hidden
context needed to judge it: true role, assigned personality, and (for
wolves) the deception role they were pre-committed to.

Chat text alone cannot answer "did this player stay in character" or "did
the fake-seer execute the plan" -- those need the hidden assignment sitting
next to the utterance, which is exactly what this captures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Utterance:
    day: int
    phase: str
    kind: str  # discussion | vote | night_action | wolf_chat | freemason_chat | summary
    player_id: str
    player_name: str
    role: str
    team: str
    personality: str
    deception_role: str | None
    text: str = ""
    target: str | None = None
    reasoning_memo: dict[str, Any] | None = None
    public_claim_role: str | None = None
    public_results: list[dict[str, Any]] = field(default_factory=list)
    directed_question_targets: list[str] = field(default_factory=list)
    ready_to_vote: bool | None = None
    used_fallback: bool = False
    effective_length_limit: int | None = None
    key_point: str = ""
    agrees_with: list[str] = field(default_factory=list)


@dataclass
class GameTranscript:
    seed: int | None = None
    provider: str = "unknown"
    names: dict[str, str] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    teams: dict[str, str] = field(default_factory=dict)
    personalities: dict[str, str] = field(default_factory=dict)
    deception: dict[str, Any] = field(default_factory=dict)
    utterances: list[Utterance] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "provider": self.provider,
            "names": self.names,
            "roles": self.roles,
            "teams": self.teams,
            "personalities": self.personalities,
            "deception": self.deception,
            "utterances": [asdict(u) for u in self.utterances],
            "final_state": self.final_state,
        }

    def by_kind(self, kind: str) -> list[Utterance]:
        return [u for u in self.utterances if u.kind == kind]

    def by_player(self, player_id: str) -> list[Utterance]:
        return [u for u in self.utterances if u.player_id == player_id]


class TranscriptRecorder:
    """Captures AI decisions for evaluation runs and post-game play analysis."""

    def __init__(self) -> None:
        self.transcript = GameTranscript()

    def set_roster(
        self,
        *,
        names: dict[str, str],
        roles: dict[str, str],
        teams: dict[str, str],
        personalities: dict[str, str],
        deception: dict[str, Any],
        seed: int | None,
        provider: str,
    ) -> None:
        t = self.transcript
        t.names, t.roles, t.teams = names, roles, teams
        t.personalities, t.deception = personalities, deception
        t.seed, t.provider = seed, provider

    def record(self, utterance: Utterance) -> None:
        self.transcript.utterances.append(utterance)

    def finalize(self, final_state: dict[str, Any]) -> GameTranscript:
        self.transcript.final_state = final_state
        return self.transcript
