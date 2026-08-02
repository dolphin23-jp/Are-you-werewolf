"""Structured-output contracts requested from the LLM. These are the actual
JSON-schema contract sent to the provider (see provider/luna_openai.py) --
not just documentation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PublicResultClaim(BaseModel):
    result_type: str  # seer | medium
    target_id: str
    is_werewolf: bool


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


class NightActionOutput(BaseModel):
    target: str
    reason: str = ""


class WolfChatOutput(BaseModel):
    message: str


class SummaryOutput(BaseModel):
    summary: str
