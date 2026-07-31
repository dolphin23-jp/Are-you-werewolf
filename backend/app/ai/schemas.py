"""Structured-output contracts requested from the LLM. These are the actual
JSON-schema contract sent to the provider (see provider/luna_openai.py) --
not just documentation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReasoningMemo(BaseModel):
    """Persistent per-AI scratchpad. Each turn's memo REPLACES the previous
    one (a cheap, stateless-server-friendly memory design) rather than
    accumulating indefinitely."""

    trusted_seer: str | None = None
    suspects: list[str] = Field(default_factory=list)
    trusted: list[str] = Field(default_factory=list)
    execution_target: str | None = None
    overall_thought: str = ""


class DiscussionOutput(BaseModel):
    public_message: str
    reasoning_memo: ReasoningMemo = Field(default_factory=ReasoningMemo)
    contains_co_claim: bool = False


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
