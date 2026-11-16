"""Structured-output contracts requested from the LLM. These are the actual
JSON-schema contract sent to the provider (see provider/luna_openai.py) --
not just documentation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PublicResultClaim(BaseModel):
    result_type: str  # seer | medium
    target_id: str
    is_werewolf: bool
    # Which night this result came from. Omitted means "last night", which is
    # the common case; naming it is what lets a held-back result be published
    # later without the table reading it as a fresh look.
    referenced_day: int | None = None


class ClaimAction(BaseModel):
    """Changing a standing CO, as a declaration rather than a turn of phrase.

    Sliding from seer to medium is a real move with real consequences, and
    inferring it from prose means the ledger and the table can disagree about
    whether it happened.
    """

    action: str  # retract | switch
    role: str | None = None  # required for switch: the role being moved to
    reason: str = ""


class ResultAction(BaseModel):
    """Withdrawing or correcting a verdict already published."""

    action: str  # retract | correct
    result_type: str  # seer | medium
    target_id: str
    is_werewolf: bool | None = None  # required for correct: the new colour
    referenced_day: int | None = None
    reason: str = ""


class DirectedQuestion(BaseModel):
    target_id: str
    question: str = ""
    source_message_id: str | None = None
    topic: str = ""


class ReasoningMemo(BaseModel):
    """Persistent per-AI scratchpad. Each turn's memo REPLACES the previous
    one (a cheap, stateless-server-friendly memory design) rather than
    accumulating indefinitely."""

    trusted_seer: str | None = None
    suspects: list[str] = Field(default_factory=list)
    trusted: list[str] = Field(default_factory=list)
    execution_target: str | None = None
    overall_thought: str = ""
    role_hypotheses: list[str] = Field(default_factory=list)
    fox_candidates: list[str] = Field(default_factory=list)
    private_team_thought: str = ""


class Reassessment(BaseModel):
    player_id: str
    accepted_point: str = ""
    remaining_reason: str = ""
    changed_mind: bool = False


class DiscussionOutput(BaseModel):
    public_message: str
    key_point: str = ""
    agrees_with: list[str] = Field(default_factory=list)
    reply_to: str | None = None
    quote: str | None = None
    reasoning_memo: ReasoningMemo = Field(default_factory=ReasoningMemo)
    contains_co_claim: bool = False
    public_claim_role: str | None = None
    public_results: list[PublicResultClaim] = Field(default_factory=list)
    claim_action: ClaimAction | None = None
    result_actions: list[ResultAction] = Field(default_factory=list)
    directed_questions: list[DirectedQuestion] = Field(default_factory=list)
    ready_to_vote: bool = False
    needs_another_statement: bool = False
    reassessments: list[Reassessment] = Field(default_factory=list)
    alternative_execution_target: str | None = None
    strongest_case_against_execution: str = ""

    @model_validator(mode="after")
    def reject_conflicting_actions(self) -> DiscussionOutput:
        if self.claim_action and self.public_claim_role:
            raise ValueError("claim_action cannot be combined with public_claim_role")
        changed = {(a.result_type, a.target_id) for a in self.result_actions}
        duplicated = changed & {(r.result_type, r.target_id) for r in self.public_results}
        if duplicated:
            raise ValueError("a corrected or retracted result cannot also be republished")
        return self


def _omit_description(schema: dict[str, Any]) -> None:
    """Keep a wire model's docstring out of its JSON schema.

    Pydantic copies a class docstring into the schema's `description`, and the
    provider sends that schema to the model verbatim. The docstrings below are
    written for maintainers -- they name the very words the wire form avoids --
    so they must not travel with the request.
    """
    schema.pop("description", None)


# PlanNote field -> ReasoningMemo field. One table, so the wire conversion and
# the prompt's echo of the previous memo cannot drift apart.
PLAN_NOTE_MEMO_FIELDS: dict[str, str] = {
    "execution_target": "execution_target",
    "suspects": "suspects",
    "trusted": "trusted",
    "trusted_seer": "trusted_seer",
    "fox_candidates": "fox_candidates",
    "plan": "overall_thought",
    "team_plan": "private_team_thought",
}


class PlanNote(BaseModel):
    """Wire form of `ReasoningMemo` for the v3 chat contract.

    The two free-text fields are renamed, not just relabelled: a property that
    asks the model to write out its "thought" or "reasoning" can be refused by
    Anthropic as reasoning extraction (`stop_details.category ==
    "reasoning_extraction"`), and the server-side refusal fallback does not
    retry that category -- the turn is simply lost. What we want here was never
    the model's hidden thinking anyway, only a short plan the next turn can
    build on. `role_hypotheses` is dropped: the v3 contract never asked for it.
    """

    model_config = ConfigDict(json_schema_extra=_omit_description)

    execution_target: str | None = None
    suspects: list[str] = Field(default_factory=list)
    trusted: list[str] = Field(default_factory=list)
    trusted_seer: str | None = None
    fox_candidates: list[str] = Field(default_factory=list)
    plan: str = ""
    team_plan: str = ""

    def to_memo(self) -> ReasoningMemo:
        # Validation copies the lists, so the memo never aliases this note.
        return ReasoningMemo(
            **{
                memo_field: getattr(self, note_field)
                for note_field, memo_field in PLAN_NOTE_MEMO_FIELDS.items()
            }
        )

    @classmethod
    def from_memo(cls, memo: ReasoningMemo) -> PlanNote:
        return cls(
            **{
                note_field: getattr(memo, memo_field)
                for note_field, memo_field in PLAN_NOTE_MEMO_FIELDS.items()
            }
        )


class ChatDiscussionOutput(BaseModel):
    """What the v3 discussion turn asks the provider for, on the wire.

    `DiscussionOutput` has 36 optional parameters once nested objects are
    counted, and Anthropic structured outputs refuse a request whose schema has
    more than 24 optional or 16 union-typed parameters ("Schema is too complex
    for compilation", a 400 no retry can fix). The v3 prompt asks for a small
    subset anyway, so the wire form is exactly that subset (18 optional, 6
    union, pinned by tests/ai/test_schema_complexity.py). Every caller still
    receives a `DiscussionOutput` through `to_discussion_output`, which leaves
    the fields v3 does not request at their defaults.
    """

    model_config = ConfigDict(json_schema_extra=_omit_description)

    public_message: str
    reply_to: str | None = None
    plan_note: PlanNote = Field(default_factory=PlanNote)
    public_claim_role: str | None = None
    public_results: list[PublicResultClaim] = Field(default_factory=list)
    directed_questions: list[DirectedQuestion] = Field(default_factory=list)
    ready_to_vote: bool = False
    needs_another_statement: bool = False

    def to_discussion_output(self) -> DiscussionOutput:
        """The engine-facing form. `contains_co_claim` is derived rather than
        asked for: a CO is exactly a turn that names the role it claims."""
        return DiscussionOutput(
            public_message=self.public_message,
            reply_to=self.reply_to,
            reasoning_memo=self.plan_note.to_memo(),
            contains_co_claim=bool(self.public_claim_role),
            public_claim_role=self.public_claim_role,
            public_results=[item.model_copy() for item in self.public_results],
            directed_questions=[item.model_copy() for item in self.directed_questions],
            ready_to_vote=self.ready_to_vote,
            needs_another_statement=self.needs_another_statement,
        )

    @classmethod
    def from_discussion_output(cls, output: DiscussionOutput) -> ChatDiscussionOutput:
        """The wire form of an engine-side answer -- what offline providers use
        to answer the v3 contract with the same turn they build for the full
        one. Fields the wire form has no room for are dropped."""
        return cls(
            public_message=output.public_message,
            reply_to=output.reply_to,
            plan_note=PlanNote.from_memo(output.reasoning_memo),
            public_claim_role=output.public_claim_role,
            public_results=[item.model_copy() for item in output.public_results],
            directed_questions=[item.model_copy() for item in output.directed_questions],
            ready_to_vote=output.ready_to_vote,
            needs_another_statement=output.needs_another_statement,
        )


class BriefDiscussionOutput(BaseModel):
    """Minimal fallback contract. When the full `DiscussionOutput` fails every
    attempt -- almost always because the JSON was cut off mid-object -- asking for
    only the sentence the player would say still produces a real turn, instead of
    a typing indicator that resolves to nothing."""

    public_message: str


class MorningIntentOutput(BaseModel):
    timing: str = "normal"  # immediate | after_results | normal | hold
    intent: str = "normal"  # publish_result | claim | lead | question | normal
    public_claim_role: str | None = None
    priority_reason: str = ""


class VoteOutput(BaseModel):
    vote_target: str
    reason: str = ""
    decisive_evidence: str = ""
    countercase: str = ""
    alternative_target: str | None = None


class NightActionOutput(BaseModel):
    target: str
    reason: str = ""


class WolfChatOutput(BaseModel):
    message: str


class SummaryOutput(BaseModel):
    summary: str
