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
    content: str = Field(min_length=1, max_length=500)
    channel: str = "public"


class VoteRequest(BaseModel):
    target_id: str


class NightActionRequest(BaseModel):
    action_type: str
    target_id: str


class CoRequest(BaseModel):
    claimed_role: str


class OkResponse(BaseModel):
    ok: bool = True
