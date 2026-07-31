"""Wire-protocol pydantic models for the REST/WS API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CreateGameRequest(BaseModel):
    # Not "あなた": that string is broadcast verbatim into every OTHER
    # player's prompt (chat log, roster labels), where it reads as the
    # second-person pronoun rather than a name.
    human_name: str = Field(default="ゲスト", max_length=32)
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
