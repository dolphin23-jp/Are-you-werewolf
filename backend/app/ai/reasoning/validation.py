"""State-consistency validation for AI output, checked against the public ledger.

This layer only rejects things that are *impossible*, never things that are
merely implausible: a dead player named as today's execution candidate, a
medium result about someone who was never executed, a re-published verdict whose
colour flipped since it was said. Being wrong about who the wolf is stays the
AI's prerogative -- being wrong about what happened does not.

Every repair here is deterministic and belief-driven. An invalid target falls
back to the player's own previous belief, then to their own suspect list, then
to the first remaining candidate in seating order -- never to a random pick,
which is how an error used to turn into a vote nobody could explain.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field

from app.ai.reasoning.facts import MEDIUM_RESULT, RESULT_TYPES, SEER_RESULT, PublicFactLedger
from app.ai.schemas import DiscussionOutput, PublicResultClaim, ReasoningMemo
from app.engine.roles import RoleName


@dataclass(frozen=True)
class ValidationIssue:
    """One rejected or repaired piece of AI output."""

    code: str
    detail: str
    player_id: str = ""
    field_name: str = ""


@dataclass(frozen=True)
class MemoValidation:
    memo: ReasoningMemo
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class ResultValidation:
    """`claim` is None when the claim was impossible and had to be dropped."""

    claim: PublicResultClaim | None
    issues: tuple[ValidationIssue, ...] = ()


@dataclass(frozen=True)
class TargetResolution:
    target: str
    issues: tuple[ValidationIssue, ...] = ()


@dataclass(frozen=True)
class VotePlanMismatch:
    """The player said they wanted X executed and then voted for Y.

    Recorded, never corrected: changing your mind at the ballot is legitimate
    play, and only the human reader can tell that apart from incoherence.
    `stated_target_votable` says whether the plan was even available this round,
    which separates a real reversal from a plan the runoff had already removed.
    """

    voter_id: str
    day: int
    round: int
    stated_target: str
    actual_target: str
    stated_target_votable: bool = True


@dataclass
class ValidationLog:
    """Accumulates issues so a caller can assert on them after the fact."""

    issues: list[ValidationIssue] = field(default_factory=list)
    vote_plan_mismatches: list[VotePlanMismatch] = field(default_factory=list)

    def extend(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues.extend(issues)

    def codes(self) -> list[str]:
        return [issue.code for issue in self.issues]

    def by_code(self, code: str) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.code == code]


# -- reasoning memo --


def validate_reasoning_memo(
    memo: ReasoningMemo,
    ledger: PublicFactLedger,
    *,
    owner_id: str,
    previous_execution_target: str | None = None,
    excluded_target_ids: Collection[str] = (),
) -> MemoValidation:
    """Return a memo safe to persist, plus what had to be changed to get there.

    Free-text fields (`overall_thought`, `private_team_thought`,
    `role_hypotheses`) are deliberately untouched -- they are opinion, and this
    layer must not start mining them for facts.
    """
    issues: list[ValidationIssue] = []
    checked = memo.model_copy(deep=True)

    def note(code: str, detail: str, field_name: str) -> None:
        issues.append(
            ValidationIssue(code=code, detail=detail, player_id=owner_id, field_name=field_name)
        )

    if checked.trusted_seer is not None and not _is_other_known_player(
        ledger, checked.trusted_seer, owner_id
    ):
        note(
            "memo_trusted_seer_invalid",
            f"unknown or self: {checked.trusted_seer}",
            "trusted_seer",
        )
        checked.trusted_seer = None

    for field_name in ("suspects", "trusted", "fox_candidates"):
        kept, dropped = _filter_player_ids(ledger, getattr(checked, field_name), owner_id)
        setattr(checked, field_name, kept)
        for dropped_id in dropped:
            note("memo_player_id_invalid", f"unknown, self or duplicate: {dropped_id}", field_name)

    target = checked.execution_target
    # None is a legitimate answer ("nobody yet"); only a *stated* target that
    # cannot be executed gets repaired, so an intentional withdrawal is preserved.
    if target is not None:
        reason = _execution_target_problem(ledger, target, owner_id, excluded_target_ids)
        if reason is not None:
            repaired = _repair_execution_target(
                ledger,
                owner_id=owner_id,
                previous_target=previous_execution_target,
                suspects=checked.suspects,
                excluded_target_ids=excluded_target_ids,
            )
            note(
                reason,
                f"{target} cannot be this day's execution target; using {repaired}",
                "execution_target",
            )
            checked.execution_target = repaired

    return MemoValidation(memo=checked, issues=tuple(issues))


def _is_other_known_player(ledger: PublicFactLedger, player_id: str, owner_id: str) -> bool:
    return ledger.is_known(player_id) and player_id != owner_id


def _filter_player_ids(
    ledger: PublicFactLedger, values: Sequence[str], owner_id: str
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for value in values:
        if _is_other_known_player(ledger, value, owner_id) and value not in kept:
            kept.append(value)
        else:
            dropped.append(value)
    return kept, dropped


def _execution_target_problem(
    ledger: PublicFactLedger,
    target: str,
    owner_id: str,
    excluded_target_ids: Collection[str],
) -> str | None:
    if not ledger.is_known(target):
        return "memo_execution_target_unknown"
    if target == owner_id:
        return "memo_execution_target_self"
    if not ledger.is_alive(target):
        return "memo_execution_target_dead"
    if target in excluded_target_ids:
        return "memo_execution_target_excluded"
    return None


def _repair_execution_target(
    ledger: PublicFactLedger,
    *,
    owner_id: str,
    previous_target: str | None,
    suspects: Sequence[str],
    excluded_target_ids: Collection[str],
) -> str | None:
    """Deterministic, belief-driven repair: yesterday's conclusion first, then
    this player's own suspect list, then nothing. Never a random substitute."""
    for candidate in (previous_target, *suspects):
        if candidate is None:
            continue
        if _execution_target_problem(ledger, candidate, owner_id, excluded_target_ids) is None:
            return candidate
    return None


# -- published seer/medium results --


def validate_public_result_claim(
    claim: PublicResultClaim,
    ledger: PublicFactLedger,
    *,
    claimant_id: str,
    is_correction: bool = False,
) -> ResultValidation:
    """Check one published verdict against the board.

    Seer and medium are validated separately on purpose: they answer different
    questions and are only available about different people. Treating them as
    one "result" is how a medium ends up reporting on a living player.

    `is_correction` marks a verdict the claimant has explicitly said replaces an
    earlier one. Everything else is checked identically -- the only difference is
    that changing the colour is the declared intent rather than a silent flip.
    """
    issues: list[ValidationIssue] = []

    def note(code: str, detail: str, field_name: str = "") -> None:
        issues.append(
            ValidationIssue(
                code=code, detail=detail, player_id=claimant_id, field_name=field_name
            )
        )

    if claim.result_type not in RESULT_TYPES:
        note("result_type_invalid", f"unknown result type {claim.result_type}", "result_type")
        return ResultValidation(claim=None, issues=tuple(issues))
    if not ledger.is_known(claim.target_id):
        note("result_target_unknown", f"unknown target {claim.target_id}", "target_id")
        return ResultValidation(claim=None, issues=tuple(issues))
    if claim.target_id == claimant_id:
        note("result_target_self", "a player cannot publish a result about themselves", "target_id")
        return ResultValidation(claim=None, issues=tuple(issues))

    claimed_role = ledger.claimed_role_of(claimant_id)
    expected_role = RoleName.SEER if claim.result_type == SEER_RESULT else RoleName.MEDIUM
    # No CO yet is normal: the claim and the CO usually arrive in the same message.
    if claimed_role in (RoleName.SEER, RoleName.MEDIUM) and claimed_role != expected_role:
        note(
            "result_role_mismatch",
            f"{claimant_id} claims {claimed_role.value} but published a {claim.result_type} result",
            "result_type",
        )
        return ResultValidation(claim=None, issues=tuple(issues))

    if claim.result_type == SEER_RESULT:
        if claim.target_id == ledger.first_victim_id:
            note(
                "result_seer_target_first_victim",
                f"{claim.target_id} was the first victim and cannot be divined",
                "target_id",
            )
            return ResultValidation(claim=None, issues=tuple(issues))
    else:
        execution_day = ledger.execution_day(claim.target_id)
        if execution_day is None:
            note(
                "result_medium_target_not_executed",
                f"{claim.target_id} has never been executed, so no medium result exists",
                "target_id",
            )
            return ResultValidation(claim=None, issues=tuple(issues))
        if execution_day > ledger.day:
            note(
                "result_medium_target_not_executed",
                f"{claim.target_id} is executed on day {execution_day}, after day {ledger.day}",
                "target_id",
            )
            return ResultValidation(claim=None, issues=tuple(issues))

    checked = claim.model_copy(deep=True)
    existing = ledger.find_result(claimant_id, claim.result_type, claim.target_id)
    if existing is not None and existing.is_werewolf != claim.is_werewolf and not is_correction:
        note(
            "result_polarity_conflict",
            (
                f"{claimant_id} published {claim.target_id}="
                f"{'黒' if existing.is_werewolf else '白'} on day {existing.day}; "
                "keeping the recorded verdict"
            ),
            "is_werewolf",
        )
        checked.is_werewolf = existing.is_werewolf
    return ResultValidation(claim=checked, issues=tuple(issues))


def validate_public_result_claims(
    claims: Sequence[PublicResultClaim],
    ledger: PublicFactLedger,
    *,
    claimant_id: str,
) -> tuple[list[PublicResultClaim], tuple[ValidationIssue, ...]]:
    kept: list[PublicResultClaim] = []
    issues: list[ValidationIssue] = []
    for claim in claims:
        validation = validate_public_result_claim(claim, ledger, claimant_id=claimant_id)
        issues.extend(validation.issues)
        if validation.claim is not None:
            kept.append(validation.claim)
    return kept, tuple(issues)


# -- whole discussion turn --

_ALTERNATIVE_TARGET_CODES = {
    "memo_execution_target_unknown": "memo_alternative_target_unknown",
    "memo_execution_target_self": "memo_alternative_target_self",
    "memo_execution_target_dead": "memo_alternative_target_dead",
    "memo_execution_target_excluded": "memo_alternative_target_excluded",
}


def validate_discussion_output(
    output: DiscussionOutput,
    ledger: PublicFactLedger,
    *,
    speaker_id: str,
    previous_execution_target: str | None = None,
    excluded_target_ids: Collection[str] = (),
) -> tuple[DiscussionOutput, tuple[ValidationIssue, ...]]:
    """Repair a discussion turn in place before any of it reaches the engine."""
    issues: list[ValidationIssue] = []

    memo_validation = validate_reasoning_memo(
        output.reasoning_memo,
        ledger,
        owner_id=speaker_id,
        previous_execution_target=previous_execution_target,
        excluded_target_ids=excluded_target_ids,
    )
    output.reasoning_memo = memo_validation.memo
    issues.extend(memo_validation.issues)

    alternative = output.alternative_execution_target
    if alternative is not None:
        problem = _execution_target_problem(ledger, alternative, speaker_id, excluded_target_ids)
        code = _ALTERNATIVE_TARGET_CODES.get(problem or "")
        if problem is None and alternative == output.reasoning_memo.execution_target:
            code = "memo_alternative_target_duplicates_primary"
        if code is not None:
            issues.append(
                ValidationIssue(
                    code=code,
                    detail=f"{alternative} cannot be the second execution candidate",
                    player_id=speaker_id,
                    field_name="alternative_execution_target",
                )
            )
            output.alternative_execution_target = None

    kept_results, result_issues = validate_public_result_claims(
        output.public_results, ledger, claimant_id=speaker_id
    )
    output.public_results = kept_results
    issues.extend(result_issues)

    return output, tuple(issues)


# -- vote and night-action targets --


def resolve_target(
    proposed: str | None,
    *,
    candidates: Sequence[str],
    preferred: Sequence[str | None] = (),
    actor_id: str = "",
    code: str = "target_invalid",
) -> TargetResolution | None:
    """Pick a valid target deterministically, preferring the actor's own beliefs.

    Returns None only when there is nothing legal to pick. `candidates` is
    already filtered by the caller to living, eligible, non-self players, so
    anything chosen here is guaranteed executable by the engine.
    """
    if not candidates:
        return None
    if proposed in candidates:
        return TargetResolution(target=str(proposed))
    fallback = next(
        (item for item in preferred if item is not None and item in candidates), candidates[0]
    )
    return TargetResolution(
        target=fallback,
        issues=(
            ValidationIssue(
                code=code,
                detail=f"{proposed!r} is not a valid target; using {fallback}",
                player_id=actor_id,
                field_name="target",
            ),
        ),
    )


def detect_vote_plan_mismatch(
    ledger: PublicFactLedger,
    *,
    voter_id: str,
    stated_target: str | None,
    day: int,
    round_number: int,
    actual_target: str | None = None,
) -> VotePlanMismatch | None:
    """Compare what the player said they would do with the ballot they cast.

    `actual_target` defaults to the recorded vote, so the comparison is always
    against the ledger rather than against another piece of AI output.
    """
    if actual_target is None:
        recorded = ledger.vote_of(voter_id, day, round_number)
        actual_target = recorded.target_id if recorded is not None else None
    if not stated_target or not actual_target or stated_target == actual_target:
        return None
    return VotePlanMismatch(
        voter_id=voter_id,
        day=day,
        round=round_number,
        stated_target=stated_target,
        actual_target=actual_target,
        stated_target_votable=stated_target in ledger.votable_ids(voter_id),
    )


__all__ = [
    "MEDIUM_RESULT",
    "SEER_RESULT",
    "MemoValidation",
    "ResultValidation",
    "TargetResolution",
    "ValidationIssue",
    "ValidationLog",
    "VotePlanMismatch",
    "detect_vote_plan_mismatch",
    "resolve_target",
    "validate_discussion_output",
    "validate_public_result_claim",
    "validate_public_result_claims",
    "validate_reasoning_memo",
]
