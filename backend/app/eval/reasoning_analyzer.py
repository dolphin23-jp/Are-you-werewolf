"""Recompute release-quality facts from immutable transcript audit events."""

from __future__ import annotations

import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

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
    # Nothing to compare the ballot with: the seat had no turn that day, or its
    # memo named nobody. Not a change and not "unchanged" -- the record used to
    # print these as 「変更なし」, a third of the ballots of a game.
    NO_BASIS = "no_basis"
    # The seat left a memo target it never said at the table that day (the
    # name matched whole in its own public lines). The gate measures the
    # public stance against the vote, and the table saw no stance to leave;
    # the report lists these with quotes instead of failing them.
    MEMO_TARGET_UNSPOKEN = "memo_target_unspoken"


# Changes with a cause the table could see. `no_basis` compared nothing, and
# `memo_target_unspoken` left a stance nobody heard: neither is judged.
JUSTIFIED_CHANGES = frozenset(VoteChangeKind) - {
    VoteChangeKind.NONE,
    VoteChangeKind.UNEXPLAINED,
    VoteChangeKind.NO_BASIS,
    VoteChangeKind.MEMO_TARGET_UNSPOKEN,
}


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


_SENTENCE_ENDS = frozenset("。！？!?")
_QUOTE_OPENS = frozenset("「『")
_QUOTE_CLOSES = frozenset("」』")


def split_sentences(text: str) -> list[str]:
    """`text` split after 。！？ and at line breaks, never inside 「」『』; a closing 。 dropped."""
    sentences: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        ended = bool(current) and current[-1] in _SENTENCE_ENDS and depth == 0
        if char == "\n" or (ended and char not in _SENTENCE_ENDS):
            sentences.append("".join(current))
            current = []
            if char == "\n":
                depth = 0
                continue
        if char in _QUOTE_OPENS:
            depth += 1
        elif char in _QUOTE_CLOSES and depth:
            depth -= 1
        current.append(char)
    sentences.append("".join(current))
    cleaned = (sentence.strip().removesuffix("。").strip() for sentence in sentences)
    return [sentence for sentence in cleaned if sentence]


def sentence_naming(text: str, seat: str, names: Mapping[str, str]) -> str:
    """The sentence of `text` that puts something to `seat`: "" when none names it.

    The first one naming the seat that ends as a question, else the last one
    naming it -- a request (「理由を教えて」) usually closes the line.
    """
    naming = [s for s in split_sentences(text) if seat in named_seat_ids(s, names)]
    asked = [s for s in naming if s[-1] in "？?"]
    if asked:
        return asked[0]
    return naming[-1] if naming else ""


# -- a ballot's reason that names another seat as where it goes --

_HONORIFIC = r"(?:さん|くん|君|ちゃん|氏|様)?"
# The voter's own ballot, in the forms that state or propose it: 入れる / 入れます /
# 入れよう / 入れとく / 入れておく / 入れちゃう, 投票する / します / しよう / しとく /
# しておく, or 一票 (に一票。 / に一票入れる).
_OWN_BALLOT = (
    r"(?:に|へ)\s*(?:一票(?:を)?(?:入れ(?:る|ます|よう|とく|ておく|ちゃう))?"
    r"|入れ(?:る|ます|よう|とく|ておく|ちゃう)"
    r"|投票(?:する|します|しよう|しとく|しておく))"
)
# What may follow: the end of the clause or a sentence-final particle. Not
# って / と (a quotation), か (a question), な (入れるな), の (入れるのは違う),
# or a noun the verb modifies (入れる人) -- each of those is about a ballot,
# not the voter's own.
_BALLOT_END = r"(?=$|[。．.！!、,，\s…〜~ー」』）)]|[よねわぞぜ]|けど|けれど|から)"
# Another seat standing as the subject of the clause: 「アオイはレンに入れる」.
_SUBJECT_AFTER = r"\s*" + _HONORIFIC + r"\s*[はがも]"
_CONNECTIVE = re.compile(r"から|ので|けど|けれど|ため|のに")


def seats_reason_votes_for(
    reason: str, voter_id: str, target_id: str | None, names: Mapping[str, str]
) -> tuple[str, ...]:
    """Seats other than the ballot's target that the reason says the ballot goes to.

    Read from the form 「Xに入れる」/「Xに投票します」/「Xに一票」 with the voter as
    the subject: past and progressive forms (入れた, 入れてる), a quotation, a
    question and a modified noun describe somebody's ballot, not this one, and
    a clause whose subject is another seat is that seat's. Seed 1's ツムギ
    voted ホノカ with 「寡黙気味のレンさんに入れるよ」. Listed for a reader, never
    used to re-decide the ballot.
    """
    found: list[tuple[int, str]] = []
    for seat, name in names.items():
        if seat in (voter_id, target_id):
            continue
        for label in {seat, name} - {""}:
            pattern = re.compile(
                _label_pattern(label).pattern + _HONORIFIC + _OWN_BALLOT + _BALLOT_END
            )
            for match in pattern.finditer(reason):
                clause = re.split(r"[。．！!？?、,，\n]", reason[: match.start()])[-1]
                if not _another_subject(clause, names):
                    found.append((match.start(), seat))
    return tuple(dict.fromkeys(seat for _start, seat in sorted(found)))


def _another_subject(clause: str, names: Mapping[str, str]) -> bool:
    """Whether a seat is the subject of the ballot in this clause (「アオイはレンに入れる」).

    Not when a connective closes the seat's own predicate first: in
    「ハルトが怪しいからレンに入れる」 ハルト is the subject of 怪しい, and the
    ballot is the voter's.
    """
    for _start, end, _seat in mention_spans(clause, names):
        subject = re.match(_SUBJECT_AFTER, clause[end:])
        if subject and not _CONNECTIVE.search(clause[end + subject.end() :]):
            return True
    return False


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

    The release gate measures the seat's public stance against its vote
    (owner's decision, 2026-11-17). A ballot with no candidate to compare
    with is `no_basis`. A memo candidate the seat never said at the table
    that day is no public stance, so leaving it is `memo_target_unspoken`
    -- after the runoff and the death, which explain any change whatever
    the stance was.

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
    own_lines = [line.content for line in facts.day_lines if line.author_id == voter]
    if basis is None:
        return VoteChangeKind.NO_BASIS, "メモの吊り先が空" if own_lines else "その日の手番なし"
    if target == basis:
        return VoteChangeKind.NONE, ""
    if facts.runoff and basis not in facts.candidates:
        return VoteChangeKind.RUNOFF_RESTRICTION, "決選投票の対象制限"
    if basis not in facts.alive_ids:
        return VoteChangeKind.TARGET_BECAME_INVALID, "発言後に対象が無効化"
    if not any(basis in named_seat_ids(line, facts.names) for line in own_lines):
        name = facts.names.get(basis, basis)
        return (
            VoteChangeKind.MEMO_TARGET_UNSPOKEN,
            f"メモの吊り先{name}は、その日の自分の発言に一度も出ていない",
        )
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


# -- every ballot of a transcript --


class AllyVoteKind(StrEnum):
    """A werewolf's ballot on its own partner, by what stood behind it."""

    # A wolf-chat plan or a public mark on the partner (`WolfAllyVotePlan`).
    PLANNED = "planned"
    # No plan, but it followed the day's final designation and decided
    # nothing: the round recounted without it executes the same seat. Not a
    # hard failure (owner, 2026-11-17); listed with the designation it followed.
    DESIGNATED = "designated"
    # Anything else -- no designation, another seat designated, or the vote
    # that tipped the execution. A hard failure.
    UNPLANNED = "unplanned"


@dataclass(frozen=True)
class BallotVerdict:
    """One audited ballot, judged from the transcript it was recorded in."""

    player_id: str
    day: int
    # The audit's own round (0 in transcripts written before rounds were
    # kept) and the round itself, read from the vote records when it is 0.
    vote_round: int
    round_number: int
    vote_target: str | None
    basis: str | None
    kind: VoteChangeKind
    why: str
    # The ballot's reason, and the seat's own public lines that day.
    reason: str = ""
    own_lines: tuple[str, ...] = ()
    vote_only: bool = False
    ally_vote: bool = False
    ally_vote_planned: bool = False
    # For a ballot on a partner: what stood behind it, the day's final
    # designation (its target and its line), and whether the ballot decided
    # the execution. None where the record cannot tell.
    ally_kind: AllyVoteKind | None = None
    designated_target: str | None = None
    designation_line: str = ""
    ally_decisive: bool | None = None
    # The round's tally, most votes first: what a reader checks "decided
    # nothing" against.
    tally: tuple[tuple[str, int], ...] = ()
    # Seats other than the target the reason says the ballot goes to
    # (`seats_reason_votes_for`). Listed in the report, never judged.
    reason_names_other: tuple[str, ...] = ()

    @property
    def hard_change(self) -> bool:
        """A change of the seat's public stance the table saw no cause for."""
        return self.kind is VoteChangeKind.UNEXPLAINED

    @property
    def hard_ally_vote(self) -> bool:
        return self.ally_kind is AllyVoteKind.UNPLANNED


def audit_ballots(transcript: GameTranscript) -> tuple[BallotVerdict, ...]:
    """Every audited ballot, classified by `classify_vote_change` from the record.

    A transcript carries what the live audit looked at: the day's public
    lines in its final state's chat log, the results in its speech-event log,
    the corrections in its audits, each seat's memo on its utterances. So a
    committed game is re-classified the way a new one is classified live,
    under today's rules. A hand-built record without a final state keeps the
    classification it was recorded with.
    """
    record = _Record(transcript)
    verdicts: list[BallotVerdict] = []
    for audit in transcript.decision_audits:
        if audit.vote_target is None:
            continue
        round_number = audit.vote_round or record.round_of(audit)
        reason = record.reason(audit.player_id, audit.day, round_number, audit.vote_target)
        if record.replayable:
            facts = record.facts(audit, round_number, reason)
            kind, why = classify_vote_change(facts)
            basis = facts.basis
            own_lines = tuple(
                line.content for line in facts.day_lines if line.author_id == audit.player_id
            )
        else:
            kind, why = _change_kind(audit), audit.target_change_reason
            basis = audit.decision_target
            own_lines = record.own_lines_from_utterances(audit.player_id, audit.day)
        ally: dict[str, Any] = {}
        if audit.ally_vote:
            ally = record.ally_vote(audit, round_number)
        verdicts.append(
            BallotVerdict(
                player_id=audit.player_id,
                day=audit.day,
                vote_round=audit.vote_round,
                round_number=round_number,
                vote_target=audit.vote_target,
                basis=basis,
                kind=kind,
                why=why,
                reason=reason,
                own_lines=own_lines,
                vote_only=audit.vote_only,
                ally_vote=audit.ally_vote,
                ally_vote_planned=audit.ally_vote_planned,
                reason_names_other=seats_reason_votes_for(
                    reason, audit.player_id, audit.vote_target, record.names
                ),
                **ally,
            )
        )
    return tuple(verdicts)


_NEW_RESULT_EVENT_TYPES = frozenset({"ability_result", "result_correction"})
# The turns that close a day's discussion (`coordinator._FINAL_DESIGNATION_STAGE`
# and the summary before it).
_CLOSING_STAGES = frozenset({"final_designation", "consensus_summary"})
_BINDING_CONFIDENCE = 0.8  # `speech_events.CLAIM_CONFIDENCE_THRESHOLD`


class _Record:
    """A transcript, indexed for the ballot audit."""

    def __init__(self, transcript: GameTranscript) -> None:
        self.t = transcript
        state = transcript.final_state or {}
        self.replayable = bool(state.get("chat_log"))
        self.names = dict(transcript.names)
        self.lines: dict[int, list[SpokenLine]] = {}
        for message in state.get("chat_log", []) or []:
            if message.get("channel") != "public":
                continue
            self.lines.setdefault(int(message.get("day", -1)), []).append(
                SpokenLine(
                    str(message.get("message_id")),
                    str(message.get("author_id")),
                    str(message.get("content") or ""),
                )
            )
        self.results: dict[int, list[PublishedResult]] = {}
        for event in state.get("speech_events", []) or []:
            if (
                event.get("event_type") in _NEW_RESULT_EVENT_TYPES
                and event.get("target_id")
                and float(event.get("confidence", 1.0)) >= _BINDING_CONFIDENCE
            ):
                self.results.setdefault(int(event.get("day", -1)), []).append(
                    PublishedResult(
                        str(event.get("actor_id")),
                        str(event.get("target_id")),
                        str(event.get("source_message_id") or ""),
                    )
                )
        self.corrections = tuple(
            audit.source_message_id for audit in transcript.correction_audits
        )
        self.death_day: dict[str, int] = {}
        for death in state.get("death_records", []) or []:
            self.death_day.setdefault(str(death.get("player_id")), int(death.get("day", 0)))
        self.votes: list[dict[str, Any]] = list(state.get("vote_records", []) or [])
        self.reasons = self._vote_reasons()

    def _vote_reasons(self) -> dict[tuple[str, int, int], str]:
        """Each ballot's reason by (seat, day, round), rounds read off the vote records."""
        remaining: dict[tuple[int, str, str | None], list[int]] = {}
        for vote in self.votes:
            key = (int(vote.get("day", -1)), str(vote.get("voter_id")), vote.get("target_id"))
            remaining.setdefault(key, []).append(int(vote.get("round", 1)))
        reasons: dict[tuple[str, int, int], str] = {}
        for utterance in self.t.utterances:
            if utterance.kind != "vote":
                continue
            queue = remaining.get((utterance.day, utterance.player_id, utterance.target))
            round_number = queue.pop(0) if queue else 0
            reasons[(utterance.player_id, utterance.day, round_number)] = utterance.text
            # A hand-built record without vote records still finds its reason.
            reasons.setdefault((utterance.player_id, utterance.day, -1), utterance.text)
        return reasons

    def round_of(self, audit: DecisionAuditRecord) -> int:
        """The round of a ballot whose audit predates rounds: the last one it matches."""
        rounds = [
            int(vote.get("round", 1))
            for vote in self.votes
            if vote.get("day") == audit.day
            and vote.get("voter_id") == audit.player_id
            and vote.get("target_id") == audit.vote_target
        ]
        return max(rounds, default=1)

    def reason(self, seat: str, day: int, round_number: int, target: str | None) -> str:
        found = self.reasons.get((seat, day, round_number))
        if found is not None:
            return found
        return next(
            (
                u.text
                for u in reversed(self.t.utterances)
                if u.kind == "vote" and u.player_id == seat and u.day == day and u.target == target
            ),
            "",
        )

    def basis(self, seat: str, day: int) -> str | None:
        """The memo target of the seat's last turn that day that carried notes.

        A turn written in code, or answered through the brief contract, has
        an empty memo and leaves the candidate where it was -- as the live
        game's record of what each seat stated does.
        """
        basis: str | None = None
        for utterance in self.t.utterances:
            if (utterance.kind, utterance.player_id, utterance.day) != ("discussion", seat, day):
                continue
            memo = utterance.reasoning_memo or {}
            if any(value not in (None, "", [], {}) for value in memo.values()):
                target = memo.get("execution_target")
                basis = str(target) if target else None
        return basis

    def own_lines_from_utterances(self, seat: str, day: int) -> tuple[str, ...]:
        return tuple(
            u.text
            for u in self.t.utterances
            if u.kind == "discussion" and u.player_id == seat and u.day == day
        )

    def designation(self, day: int) -> tuple[str | None, str]:
        """The day's final designation: its target and its line.

        The last line of the closing turns -- the designation, or the summary
        when nothing was left to wait for and the round closed on its
        仮指定. Its target is the speaker's memo target, when the line says the
        name; a line that does not name it designated nobody the table heard.
        """
        closing = [
            u
            for u in self.t.utterances
            if u.kind == "discussion" and u.day == day and u.stage in _CLOSING_STAGES
        ]
        if not closing:
            return None, ""
        last = closing[-1]
        target = (last.reasoning_memo or {}).get("execution_target")
        if isinstance(target, str) and target in named_seat_ids(last.text, self.names):
            return target, last.text
        return None, last.text

    def ally_vote(self, audit: DecisionAuditRecord, round_number: int) -> dict[str, Any]:
        """`BallotVerdict`'s ally fields for a ballot on the voter's partner."""
        if audit.ally_vote_planned:
            return {"ally_kind": AllyVoteKind.PLANNED}
        if not self.replayable:
            return {"ally_kind": AllyVoteKind.UNPLANNED}
        designated, line = self.designation(audit.day)
        counts = self.tally(audit.day, round_number)
        cast = any(
            vote.get("day") == audit.day
            and int(vote.get("round", 1)) == round_number
            and vote.get("voter_id") == audit.player_id
            for vote in self.votes
        )
        # A ballot the records do not hold cannot be shown to have decided nothing.
        decisive = not cast or _top(counts) != _top(
            self.tally(audit.day, round_number, without=audit.player_id)
        )
        follows = designated is not None and designated == audit.vote_target
        return {
            "ally_kind": (
                AllyVoteKind.DESIGNATED if follows and not decisive else AllyVoteKind.UNPLANNED
            ),
            "designated_target": designated,
            "designation_line": line,
            "ally_decisive": decisive,
            "tally": tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        }

    def tally(self, day: int, round_number: int, *, without: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for vote in self.votes:
            if vote.get("day") != day or int(vote.get("round", 1)) != round_number:
                continue
            if without is not None and vote.get("voter_id") == without:
                continue
            target = vote.get("target_id")
            if target:
                counts[str(target)] = counts.get(str(target), 0) + 1
        return counts

    def facts(self, audit: DecisionAuditRecord, round_number: int, reason: str) -> BallotFacts:
        day = audit.day
        candidates: tuple[str, ...] = ()
        if round_number >= 2:
            candidates = tuple(
                seat for seat in _top(self.tally(day, round_number - 1)) if seat != audit.player_id
            )
        return BallotFacts(
            voter_id=audit.player_id,
            basis=self.basis(audit.player_id, day),
            vote_target=audit.vote_target,
            reason=reason,
            names=self.names,
            day_lines=tuple(self.lines.get(day, ())),
            results=tuple(self.results.get(day, ())),
            correction_message_ids=self.corrections,
            runoff=round_number >= 2,
            candidates=candidates,
            alive_ids=frozenset(
                seat for seat in self.names if self.death_day.get(seat, day) >= day
            ),
        )


def _top(counts: Mapping[str, int]) -> frozenset[str]:
    """The most-voted seats of a round: the one executed, or the ones tied."""
    if not counts:
        return frozenset()
    most = max(counts.values())
    return frozenset(seat for seat, count in counts.items() if count == most)


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
    # Ballots with nothing to compare against (no turn that day, or a memo
    # that named nobody), and the rest: 「比較できた票／全票」 is
    # compared_vote_count / vote_count.
    no_basis_vote_count: int = 0
    compared_vote_count: int = 0
    # Changes off a memo target the seat never said at the table that day.
    # Listed with quotes in the report; not a gate failure.
    memo_target_unspoken_count: int = 0
    wolf_ally_vote_count: int = 0
    planned_wolf_ally_vote_count: int = 0
    # Unplanned, but following the day's final designation without deciding
    # the execution: listed in the report, not a gate failure.
    designated_wolf_ally_vote_count: int = 0
    unplanned_wolf_ally_vote_count: int = 0
    # Ballots whose reason says they go to another seat (「Xに入れる」 for an X
    # that is not the target). Listed with quotes; not a gate failure.
    vote_reason_names_other_count: int = 0
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
        ballots = audit_ballots(transcript)
        changes = {
            (verdict.player_id, verdict.day, verdict.vote_target, verdict.kind)
            for verdict in ballots
        }
        change_kinds = [kind for (_seat, _day, _target, kind) in changes]
        no_basis = sum(verdict.kind is VoteChangeKind.NO_BASIS for verdict in ballots)
        # A ballot on a partner, once per (seat, day, partner): hard if any of
        # its rounds was, planned only if every round was.
        ally_votes: dict[tuple[str, int, str | None], set[AllyVoteKind]] = {}
        for verdict in ballots:
            if verdict.ally_vote and verdict.ally_kind is not None:
                ballot = (verdict.player_id, verdict.day, verdict.vote_target)
                ally_votes.setdefault(ballot, set()).add(verdict.ally_kind)
        unplanned_ally = sum(AllyVoteKind.UNPLANNED in kinds for kinds in ally_votes.values())
        planned_ally = sum(kinds == {AllyVoteKind.PLANNED} for kinds in ally_votes.values())
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
            no_basis_vote_count=no_basis,
            compared_vote_count=len(ballots) - no_basis,
            memo_target_unspoken_count=change_kinds.count(VoteChangeKind.MEMO_TARGET_UNSPOKEN),
            wolf_ally_vote_count=len(ally_votes),
            planned_wolf_ally_vote_count=planned_ally,
            designated_wolf_ally_vote_count=len(ally_votes) - planned_ally - unplanned_ally,
            unplanned_wolf_ally_vote_count=unplanned_ally,
            vote_reason_names_other_count=sum(bool(v.reason_names_other) for v in ballots),
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
