"""Wire-protocol pydantic models for the REST/WS API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

# Message ids are short ("m12", "w3"). Unbounded ids and quotes were stored in
# the chat log and re-sent in every /view poll: a 5 MB quote made every
# 2.5-second poll 5 MB.
MessageId = Annotated[str, Field(max_length=32)]


class CreateGameRequest(BaseModel):
    human_name: str = Field(default="あなた", max_length=32)
    seed: int | None = None

    @field_validator("human_name")
    @classmethod
    def _single_line_name(cls, value: str) -> str:
        # The name is interpolated into every AI prompt; a newline in it could
        # start a line that reads like another player's message.
        return " ".join(value.split()) or "あなた"


class CreateGameResponse(BaseModel):
    session_id: str
    human_player_id: str
    player_names: dict[str, str]


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1500)
    channel: Literal["public", "wolf", "freemason"] = "public"
    reply_to: MessageId | None = None
    quote: str | None = Field(default=None, max_length=300)
    references: list[MessageId] = Field(default_factory=list, max_length=10)


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
