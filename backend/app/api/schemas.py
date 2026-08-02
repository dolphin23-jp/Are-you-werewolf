"""Wire-protocol pydantic models for the REST/WS API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CreateGameRequest(BaseModel):
    human_name: str = Field(default="あなた", max_length=32)
    seed: int | None = None


class CreateGameResponse(BaseModel):
    session_id: str
    human_player_id: str
    player_names: dict[str, str]


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1500)
    channel: str = "public"
    reply_to: str | None = None
    quote: str | None = None
    references: list[str] = Field(default_factory=list, max_length=10)


class VoteRequest(BaseModel):
    target_id: str


class NightActionRequest(BaseModel):
    action_type: str
    target_id: str


class CoRequest(BaseModel):
    claimed_role: str


class DiscussionControlRequest(BaseModel):
    action: str  # pause | resume | step


class OkResponse(BaseModel):
    ok: bool = True


class ChatResponse(OkResponse):
    message_id: str
