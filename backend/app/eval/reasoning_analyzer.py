"""Recompute release-quality facts from immutable transcript audit events."""

from __future__ import annotations

import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

from app.eval.transcript import DecisionAuditRecord, GameTranscript


class VoteChangeKind(StrEnum):
    NONE = "none"
    NEW_PUBLIC_EVIDENCE = "new_public_evidence"
    TARGET_BECAME_INVALID = "target_became_invalid"
    RUNOFF_RESTRICTION = "runoff_restriction"
    PUBLIC_RESULT_CHANGED = "public_result_changed"
    CORRECTION_ACCEPTED = "correction_accepted"
    STRATEGIC_TEAM_DECISION = "strategic_team_decision"
    # Someone else named the old or the new target in public after the seat's
    # own last word that day, and the seat's ballot reason names that someone.
    # Being moved by an argument you heard later is the listening the game
    # asks for; without this kind it was filed as unexplained, and the release
    # gate failed on the one change in the first Claude game that cited
    # exactly such an argument. A later mention the reason does not cite is
    # no cause: the day's summary and designation name the candidate after
    # nearly everyone has spoken.
    PUBLIC_ARGUMENT = "public_argument"
    UNEXPLAINED = "unexplained"


JUSTIFIED_CHANGES = frozenset(VoteChangeKind) - {VoteChangeKind.NONE, VoteChangeKind.UNEXPLAINED}


# -- seat names, matched whole --
#
# The rule `app.ai.context._mention_spans` reads seat names by, restated over a
# plain name table so a transcript can be read without a live game state:
# a label is not read inside a longer word of its own script (レン is not in
# チャレンジ, Player1 is not in Player11), and a label inside a longer seat's
# label at the same place is that longer seat (サク inside サクラ).
# `tests/eval/test_ballot_names.py` holds the two to the same answers.

_KATAKANA = "ァ-ヺヽ-ヿ"
_LONG_VOWEL = "ー"
_KATAKANA_OR_LONG_RE = re.compile(f"[{_KATAKANA}{_LONG_VOWEL}]")


def _label_pattern(label: str) -> re.Pattern[str]:
    lead = r"(?<![0-9A-Za-z])"
    if _KATAKANA_OR_LONG_RE.fullmatch(label[0]):
        lead += rf"(?<![{_KATAKANA}])(?<![{_KATAKANA}]{_LONG_VOWEL})"
    last = label[-1]
    if last.isdigit():
        trail = r"(?![0-9])"
    elif last.isascii() and last.isalpha():
        trail = r"(?![A-Za-z])"
    elif _KATAKANA_OR_LONG_RE.fullmatch(last):
        trail = rf"(?!{_LONG_VOWEL}*[{_KATAKANA}])"
    else:
        trail = ""
    return re.compile(f"{lead}{re.escape(label)}{trail}")


def mention_spans(text: str, names: Mapping[str, str]) -> list[tuple[int, int, str]]:
    """Every seat `text` names, as (start, end, player_id), in order. By id or by name."""
    found: list[tuple[int, int, str]] = []
    for player_id, name in names.items():
        for label in {player_id, name} - {""}:
            found.extend(
                (m.start(), m.end(), player_id) for m in _label_pattern(label).finditer(text)
            )
    found.sort(key=lambda span: (span[0], span[0] - span[1]))
    kept: list[tuple[int, int, str]] = []
    for span in found:
        if not any(start <= span[0] and span[1] <= end for start, end, _ in kept):
            kept.append(span)
    return kept


def named_seat_ids(text: str, names: Mapping[str, str]) -> tuple[str, ...]:
    """The seats `text` names, each once, in the order first named."""
    return tuple(dict.fromkeys(seat for _start, _end, seat in mention_spans(text, names)))


# -- one ballot, classified --


@dataclass(frozen=True)
class SpokenLine:
    """One public line of the day, in the order the table saw it."""

    message_id: str
    author_id: str
    content: str


@dataclass(frozen=True)
class PublishedResult:
    """An ability result (or its correction) stated in public, and where."""

    claimant_id: str
    target_id: str
    source_message_id: str


@dataclass(frozen=True)
class BallotFacts:
    """Everything a ballot's change of target is judged on, as plain data.

    The live coordinator fills it from the game state and the offline audit
    from a transcript's final state, so a committed game is re-classified by
    the very function that classified it live.
    """

    voter_id: str
    # The candidate the ballot is compared against: the seat's memo target
    # from its last turn of the day with notes. None when it had no such turn.
    basis: str | None
    vote_target: str | None
    reason: str
    names: Mapping[str, str]
    # The day's public lines, in order, the voter's own included.
    day_lines: Sequence[SpokenLine] = ()
    # The day's ability results and corrections, with the line each was said in.
    results: Sequence[PublishedResult] = ()
    # Lines a fact correction was heard from.
    correction_message_ids: Sequence[str] = ()
    runoff: bool = False
    candidates: Sequence[str] = ()
    alive_ids: frozenset[str] = frozenset()

    def lines_after_voter(self) -> Sequence[SpokenLine]:
        """The day's lines after the voter's last one: what it heard since it spoke."""
        last_own = max(
            (index for index, line in enumerate(self.day_lines) if line.author_id == self.voter_id),
            default=-1,
        )
        return self.day_lines[last_own + 1 :]


def classify_vote_change(facts: BallotFacts) -> tuple[VoteChangeKind, str]:
    """Why the ballot left the seat's candidate, if it did. See `VoteChangeKind`.

    Each cause is something the table could see happen after the seat last
    spoke: a runoff that left the candidate off the ballot, the candidate's
    death, a correction, another seat arguing about the old or the new target
    whom the reason names, or a new ability result about either target or by
    either of them. Only a change none of them covers is unexplained. The
    reason the seat wrote is kept either way, but it is not a cause on its
    own -- the vote prompt asks every ballot for one.

    A later message is an argument only when the reason names its author:
    the day's summary and designation name the candidate after almost
    everyone has spoken, so counting any mention made nearly every change
    "persuaded". An argument the reason cites outranks a result published
    beside it -- the seat said what moved it.

    A result counts only when it is new to the seat and touches the change.
    The seat's own results are not news to it, and one verdict about a third
    seat by a third claimant says nothing about leaving the old target for
    the new one; counting any new result, as the audit used to, filed seed 1's
    ダイキ as moved by his own result and by アオイ's on ソウタ.
    """
    basis, target, voter = facts.basis, facts.vote_target, facts.voter_id
    if basis is None or target == basis:
        return VoteChangeKind.NONE, ""
    if facts.runoff and basis not in facts.candidates:
        return VoteChangeKind.RUNOFF_RESTRICTION, "決選投票の対象制限"
    if basis not in facts.alive_ids:
        return VoteChangeKind.TARGET_BECAME_INVALID, "発言後に対象が無効化"
    later = facts.lines_after_voter()
    later_ids = {line.message_id for line in later}
    if later_ids & set(facts.correction_message_ids):
        return VoteChangeKind.CORRECTION_ACCEPTED, "発言後に事実訂正を受理"
    wanted = {basis, target}
    cited_authors = set(named_seat_ids(facts.reason, facts.names))
    arguments = [
        line.message_id
        for line in later
        if line.author_id != voter
        and line.author_id in cited_authors
        and (set(named_seat_ids(line.content, facts.names)) - {line.author_id}) & wanted
    ]
    if arguments:
        cited = "、".join(arguments)
        return (
            VoteChangeKind.PUBLIC_ARGUMENT,
            f"発言後の公開発言({cited})を受けて変更: {facts.reason}",
        )
    news = [
        result.source_message_id
        for result in facts.results
        if result.source_message_id in later_ids
        and result.claimant_id != voter
        and {result.claimant_id, result.target_id} & wanted
    ]
    if news:
        cited = "、".join(dict.fromkeys(news))
        return VoteChangeKind.NEW_PUBLIC_EVIDENCE, f"発言後に能力結果が公開({cited})"
    return VoteChangeKind.UNEXPLAINED, facts.reason


@dataclass(frozen=True)
class ReasoningQualityReport:
    decision_count: int = 0
    vote_count: int = 0
    dead_target_selection_count: int = 0
    invalid_action_target_count: int = 0
    public_fact_flip_count: int = 0
    vote_history_fabrication_count: int = 0
    private_evidence_exposed_count: int = 0
    team_private_evidence_exposed_count: int = 0
    displayed_target_mismatch_count: int = 0
    missing_required_result_count: int = 0
    duplicate_required_result_count: int = 0
    stale_evidence_used_in_brief_count: int = 0
    stale_evidence_cited_publicly_count: int = 0
    stale_evidence_attempt_count: int = 0
    stale_evidence_blocked_count: int = 0
    stale_evidence_publicly_emitted_count: int = 0
    justified_vote_change_count: int = 0
    unexplained_vote_change_count: int = 0
    wolf_ally_vote_count: int = 0
    planned_wolf_ally_vote_count: int = 0
    unplanned_wolf_ally_vote_count: int = 0
    illegal_timeline_claim_count: int = 0
    duplicate_night_action_count: int = 0
    correction_messages_heard: int = 0
    corrections_confirmed: int = 0
    corrections_refuted: int = 0
    corrections_unverifiable: int = 0
    confirmed_corrections_with_effect: int = 0
    seat_evidence_retractions: int = 0
    affected_seat_count: int = 0
    mean_belief_delta_per_affected_seat: float = 0.0
    median_belief_delta_per_affected_seat: float = 0.0

    @property
    def vote_change_rate(self) -> float:
        return (self.justified_vote_change_count + self.unexplained_vote_change_count) / max(
            self.vote_count, 1
        )

    @property
    def unexplained_vote_change_rate(self) -> float:
        return self.unexplained_vote_change_count / max(self.vote_count, 1)

    def to_dict(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "vote_change_rate": self.vote_change_rate,
            "unexplained_vote_change_rate": self.unexplained_vote_change_rate,
        }


class ReasoningTranscriptAnalyzer:
    def analyze(self, transcript: GameTranscript) -> ReasoningQualityReport:
        decisions = transcript.decision_audits
        publications = transcript.result_publication_audits
        corrections = {
            record.correction_id: record for record in transcript.correction_audits
        }.values()
        stale_attempts = sum(_stale(record) for record in decisions)
        stale_emitted = sum(_stale_emitted(record) for record in decisions)
        # Every ballot of every round has its own record. A seat that made the
        # same change, or the same vote on its partner, in two rounds of one
        # day did it once; a different change in the runoff is another.
        ballots = [record for record in decisions if record.vote_target is not None]
        changes = {
            (record.player_id, record.day, record.vote_target, _change_kind(record))
            for record in ballots
        }
        change_kinds = [kind for (_seat, _day, _target, kind) in changes]
        ally_votes: dict[tuple[str, int, str | None], bool] = {}
        for record in decisions:
            if record.ally_vote:
                ballot = (record.player_id, record.day, record.vote_target)
                ally_votes[ballot] = ally_votes.get(ballot, True) and record.ally_vote_planned
        deltas = [
            abs(delta)
            for correction in corrections
            for delta in correction.belief_delta_by_seat.values()
        ]
        public_results: dict[tuple[str, str, int | None], set[bool]] = {}
        result_nights: dict[tuple[str, int], int] = {}
        illegal_timeline = 0
        for utterance in transcript.utterances:
            for result in utterance.public_results:
                referenced = result.get("referenced_day")
                if isinstance(referenced, int):
                    illegal_timeline += int(referenced >= utterance.day)
                    night_key = (utterance.player_id, referenced)
                    result_nights[night_key] = result_nights.get(night_key, 0) + 1
                key = (
                    utterance.player_id,
                    str(result.get("target_id", "")),
                    result.get("referenced_day"),
                )
                public_results.setdefault(key, set()).add(bool(result.get("is_werewolf")))
        accepted_night_keys = [
            (record.actor_id, record.night, record.action_type)
            for record in transcript.night_action_audits
            if record.accepted
        ]
        return ReasoningQualityReport(
            # A ballot-only audit records a vote, not a decision made in public:
            # it counts towards `vote_count` and the ally checks, not here.
            decision_count=sum(not record.vote_only for record in decisions),
            vote_count=len(ballots),
            dead_target_selection_count=sum(
                not record.target_alive for record in decisions if record.decision_target
            ),
            invalid_action_target_count=sum(
                not record.target_was_legal or not record.actor_alive
                for record in transcript.night_action_audits
            ),
            private_evidence_exposed_count=sum(
                len(set(record.public_evidence_ids) & set(record.private_evidence_ids))
                for record in decisions
            ),
            team_private_evidence_exposed_count=sum(
                len(set(record.public_evidence_ids) & set(record.team_private_evidence_ids))
                for record in decisions
            ),
            displayed_target_mismatch_count=sum(
                record.displayed_target != record.decision_target for record in decisions
            ),
            missing_required_result_count=sum(
                len(record.omitted_result_ids) for record in publications
            ),
            duplicate_required_result_count=sum(
                len(record.duplicate_result_ids) for record in publications
            ),
            public_fact_flip_count=sum(len(colours) > 1 for colours in public_results.values()),
            vote_history_fabrication_count=int(
                transcript.metrics.get("inaccurate_vote_citations", 0)
            ),
            illegal_timeline_claim_count=illegal_timeline
            + sum(count - 1 for count in result_nights.values() if count > 1),
            duplicate_night_action_count=len(accepted_night_keys)
            - len(set(accepted_night_keys)),
            stale_evidence_used_in_brief_count=stale_attempts,
            stale_evidence_cited_publicly_count=stale_emitted,
            stale_evidence_attempt_count=stale_attempts,
            stale_evidence_blocked_count=stale_attempts - stale_emitted,
            stale_evidence_publicly_emitted_count=stale_emitted,
            justified_vote_change_count=sum(kind in JUSTIFIED_CHANGES for kind in change_kinds),
            unexplained_vote_change_count=change_kinds.count(VoteChangeKind.UNEXPLAINED),
            wolf_ally_vote_count=len(ally_votes),
            planned_wolf_ally_vote_count=sum(ally_votes.values()),
            unplanned_wolf_ally_vote_count=sum(not planned for planned in ally_votes.values()),
            correction_messages_heard=len(tuple(corrections)),
            corrections_confirmed=sum(record.verdict == "confirmed" for record in corrections),
            corrections_refuted=sum(record.verdict == "refuted" for record in corrections),
            corrections_unverifiable=sum(
                record.verdict == "unverifiable" for record in corrections
            ),
            confirmed_corrections_with_effect=sum(
                record.verdict == "confirmed" and bool(record.retracted_evidence_ids)
                for record in corrections
            ),
            seat_evidence_retractions=sum(
                len(record.retracted_evidence_ids) for record in corrections
            ),
            affected_seat_count=len(
                {seat for record in corrections for seat in record.affected_seat_ids}
            ),
            mean_belief_delta_per_affected_seat=statistics.fmean(deltas) if deltas else 0.0,
            median_belief_delta_per_affected_seat=statistics.median(deltas) if deltas else 0.0,
        )


def _change_kind(record: DecisionAuditRecord) -> VoteChangeKind:
    if record.vote_target == record.decision_target:
        return VoteChangeKind.NONE
    try:
        return VoteChangeKind(record.target_change_classification or VoteChangeKind.UNEXPLAINED)
    except ValueError:
        return VoteChangeKind.UNEXPLAINED


def _stale(record: DecisionAuditRecord) -> int:
    attempted = record.attempted_public_evidence_ids or record.public_evidence_ids
    return int(bool(set(attempted) - set(record.active_evidence_ids)))


def _stale_emitted(record: DecisionAuditRecord) -> int:
    attempted = record.attempted_public_evidence_ids or record.public_evidence_ids
    emitted = record.emitted_public_evidence_ids or record.publicly_emitted_evidence_ids
    stale = set(attempted) - set(record.active_evidence_ids)
    return int(bool(stale & set(emitted)))
