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


class DiscussionOutput(BaseModel):
    public_message: str
    reply_to: str | None = None
    quote: str | None = None
    reasoning_memo: ReasoningMemo = Field(default_factory=ReasoningMemo)
    contains_co_claim: bool = False
    public_claim_role: str | None = None
    public_results: list[PublicResultClaim] = Field(default_factory=list)
    directed_questions: list[DirectedQuestion] = Field(default_factory=list)
    ready_to_vote: bool = False
    needs_another_statement: bool = False


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
