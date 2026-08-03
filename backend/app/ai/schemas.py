"""Structured-output contracts requested from the LLM. These are the actual
JSON-schema contract sent to the provider (see provider/luna_openai.py) --
not just documentation."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
