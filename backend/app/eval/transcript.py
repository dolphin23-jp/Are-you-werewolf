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


@dataclass(frozen=True)
class DecisionAuditRecord:
    game_id: str
    day: int
    phase: str
    player_id: str
    decision_id: str = ""
    decision_target: str | None = None
    displayed_target: str | None = None
    vote_target: str | None = None
    public_evidence_ids: tuple[str, ...] = ()
    private_evidence_ids: tuple[str, ...] = ()
    team_private_evidence_ids: tuple[str, ...] = ()
    required_public_result_ids: tuple[str, ...] = ()
    published_result_ids: tuple[str, ...] = ()
    model_requested_target: str | None = None
    model_target_was_overridden: bool = False
    target_change_classification: str | None = None
    target_change_reason: str = ""
    active_evidence_ids: tuple[str, ...] = ()
    publicly_emitted_evidence_ids: tuple[str, ...] = ()
    target_alive: bool = True
    ally_vote: bool = False
    ally_vote_planned: bool = False
    public_result_fingerprints: tuple[str, ...] = ()
    correction_event_ids: tuple[str, ...] = ()
    alive_player_ids: tuple[str, ...] = ()
    eligible_vote_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorrectionAuditRecord:
    correction_id: str
    source_message_id: str
    speaker_id: str
    verdict: str
    affected_seat_ids: tuple[str, ...] = ()
    retracted_evidence_ids: tuple[str, ...] = ()
    belief_delta_by_seat: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultPublicationAuditRecord:
    player_id: str
    day: int
    required_result_ids: tuple[str, ...] = ()
    published_result_ids: tuple[str, ...] = ()
    omitted_result_ids: tuple[str, ...] = ()
    duplicate_result_ids: tuple[str, ...] = ()


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
    decision_evidence: str = ""
    countercase: str = ""
    alternative_target: str | None = None


@dataclass
class GameTranscript:
    game_id: str = ""
    seed: int | None = None
    provider: str = "unknown"
    names: dict[str, str] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    teams: dict[str, str] = field(default_factory=dict)
    personalities: dict[str, str] = field(default_factory=dict)
    deception: dict[str, Any] = field(default_factory=dict)
    utterances: list[Utterance] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    decision_audits: list[DecisionAuditRecord] = field(default_factory=list)
    correction_audits: list[CorrectionAuditRecord] = field(default_factory=list)
    result_publication_audits: list[ResultPublicationAuditRecord] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

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
            "decision_audits": [asdict(record) for record in self.decision_audits],
            "correction_audits": [asdict(record) for record in self.correction_audits],
            "result_publication_audits": [
                asdict(record) for record in self.result_publication_audits
            ],
            "metrics": self.metrics,
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

    def record_decision_audit(self, record: DecisionAuditRecord) -> None:
        self.transcript.decision_audits.append(record)

    def latest_open_decision(
        self, *, game_id: str, day: int, player_id: str
    ) -> tuple[int, DecisionAuditRecord] | None:
        """Return the latest same-day decision not yet linked to a ballot."""
        for index in range(len(self.transcript.decision_audits) - 1, -1, -1):
            record = self.transcript.decision_audits[index]
            if (
                record.game_id == game_id
                and record.day == day
                and record.player_id == player_id
                and record.decision_target is not None
                and record.vote_target is None
            ):
                return index, record
        return None

    def replace_decision_audit(self, index: int, record: DecisionAuditRecord) -> None:
        self.transcript.decision_audits[index] = record

    def record_correction_audit(self, record: CorrectionAuditRecord) -> None:
        self.transcript.correction_audits.append(record)

    def record_result_publication_audit(self, record: ResultPublicationAuditRecord) -> None:
        self.transcript.result_publication_audits.append(record)

    def finalize(self, final_state: dict[str, Any]) -> GameTranscript:
        self.transcript.final_state = final_state
        return self.transcript
