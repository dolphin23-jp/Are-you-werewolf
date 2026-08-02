"""REST routes for game session lifecycle and player actions.

Every mutating action accepts an optional `player_id` query parameter,
defaulting to the session's human player. This lets a single human-vs-AI
game be driven purely via curl/Swagger during development (by explicitly
passing each of the 17 player_ids) even before any AI is wired up (M2), and
is also how the AI coordinator (M3+) or a debug UI could act on behalf of
any seat if ever needed. In normal frontend use the field is simply omitted.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.ai.coordinator import AICoordinator
from app.ai.provider.factory import LLMProviderConfigError, build_llm_provider
from app.ai.reasoning import PublicFactLedger
from app.ai.reasoning.claims import build_claim_drafts, register_claim_drafts
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import DiscussionOutput
from app.api import orchestrator
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CoRequest,
    CreateGameRequest,
    CreateGameResponse,
    DiscussionControlRequest,
    NightActionRequest,
    OkResponse,
    VoteRequest,
)
from app.api.ws_hub import SessionWSHub
from app.config import get_settings
from app.engine.game import GameController, GameError, PlayerSpec
from app.engine.phases import Phase
from app.eval.transcript import TranscriptRecorder
from app.sessions.models import GameSession
from app.sessions.store import get_session_store

router = APIRouter(prefix="/api/games", tags=["games"])

AI_NAME_POOL = [
    "アカリ",
    "ハルト",
    "ユイ",
    "ソウタ",
    "ミオ",
    "レン",
    "サクラ",
    "カイト",
    "ノゾミ",
    "リク",
    "ツムギ",
    "ダイキ",
    "ホノカ",
    "シオン",
    "アオイ",
    "ケント",
]


def _new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _get_session(session_id: str) -> GameSession:
    session = get_session_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="game not found")
    session.touch()
    return session


def _resolve_player_id(session: GameSession, player_id: str | None) -> str:
    return player_id or session.human_id


def _run(fn: object, *args: object, **kwargs: object) -> Any:
    try:
        return fn(*args, **kwargs)  # type: ignore[operator]
    except GameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("", response_model=CreateGameResponse)
def create_game(req: CreateGameRequest) -> CreateGameResponse:
    settings = get_settings()
    session_id = _new_session_id()
    human_id = "p0"
    specs = [PlayerSpec(player_id=human_id, name=req.human_name, is_human=True)]
    ai_ids: list[str] = []
    for i in range(1, 17):
        pid = f"p{i}"
        ai_ids.append(pid)
        specs.append(PlayerSpec(player_id=pid, name=AI_NAME_POOL[i - 1], is_human=False))

    seed = req.seed if req.seed is not None else settings.werewolf_rng_seed
    try:
        controller = GameController(session_id=session_id, player_specs=specs, seed=seed)
    except GameError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ws_hub = SessionWSHub()
    controller.events.subscribe(ws_hub.on_event)

    try:
        provider = build_llm_provider(settings, seed=seed)
        transcript_recorder = TranscriptRecorder()
        reasoning = (
            ReasoningRuntime(controller.state, ai_ids, seed=seed)
            if settings.werewolf_reasoning_engine == "v2"
            else None
        )
        coordinator: AICoordinator | None = AICoordinator(
            controller.state,
            ai_ids,
            provider,
            seed=seed,
            recorder=transcript_recorder,
            reasoning=reasoning,
            discussion_segment_size=settings.werewolf_discussion_segment_size,
            pacing_scale=(
                0.0
                if settings.werewolf_llm_provider == "mock"
                else settings.werewolf_ai_pacing_scale
            ),
        )
    except LLMProviderConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    session = GameSession(
        session_id=session_id,
        controller=controller,
        human_id=human_id,
        ai_player_ids=ai_ids,
        ws_hub=ws_hub,
        coordinator=coordinator,
        transcript_recorder=transcript_recorder,
    )
    get_session_store().create(session)
    return CreateGameResponse(
        session_id=session_id,
        human_player_id=human_id,
        player_names={s.player_id: s.name for s in specs},
    )


@router.get("/{session_id}/transcript")
def get_transcript(session_id: str) -> dict[str, Any]:
    """Return the complete analysis log, but only after secrets can no longer affect play."""
    session = _get_session(session_id)
    if session.controller.state.phase != Phase.GAME_OVER:
        raise HTTPException(status_code=409, detail="解析用ログはゲーム終了後に取得できます")
    if session.transcript_recorder is None:
        raise HTTPException(status_code=404, detail="このゲームには解析用ログがありません")
    transcript = session.transcript_recorder.finalize(session.controller.get_debug_view())
    return transcript.to_dict()


@router.post("/{session_id}/start", response_model=OkResponse)
async def start_game(session_id: str) -> OkResponse:
    session = _get_session(session_id)
    _run(session.controller.start_game)
    await orchestrator.after_night_phase_entered(session)
    return OkResponse()


@router.get("/{session_id}/view")
def get_view(session_id: str, player_id: str | None = Query(default=None)) -> dict[str, Any]:
    session = _get_session(session_id)
    viewer = _resolve_player_id(session, player_id)
    view = session.controller.get_player_view(viewer)
    round_state = session.discussion_round
    settings = get_settings()
    awaiting = bool(
        round_state and round_state.awaiting_human and viewer == session.human_id
    )
    view["awaiting_your_speech"] = awaiting
    view["discussion_paused"] = bool(
        session.discussion_paused or session.discussion_pause_requested
    )
    view["discussion_progress"] = {
        "spoken": len(round_state.outputs) if round_state else 0,
        "total": round_state.max_total if round_state else 0,
    }
    # The countdown and its identifier only mean anything to the seat being waited
    # on, so they follow `awaiting_your_speech` rather than the raw round state.
    view["speech_wait_remaining_seconds"] = 0
    # Identifies one specific wait. The client auto-passes at most once per token,
    # so a wait that lingers at zero cannot turn into a stream of pass requests.
    view["speech_wait_token"] = None
    if awaiting and round_state is not None and round_state.awaiting_since is not None:
        elapsed = time.time() - round_state.awaiting_since
        view["speech_wait_remaining_seconds"] = max(
            0, int(settings.werewolf_discussion_wait_seconds - elapsed + 0.999)
        )
        view["speech_wait_token"] = f"{round_state.day}:{round_state.awaiting_since}"
    return view


@router.get("/{session_id}/debug")
def get_debug(session_id: str) -> dict[str, Any]:
    session = _get_session(session_id)
    return session.controller.get_debug_view()


@router.post("/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: str, req: ChatRequest, player_id: str | None = Query(default=None)
) -> ChatResponse:
    session = _get_session(session_id)
    author = _resolve_player_id(session, player_id)
    message_id = _run(
        session.controller.chat,
        author,
        req.content,
        req.channel,
        req.reply_to,
        req.quote,
        req.references,
    )
    if req.channel == "public":
        state = session.controller.state
        if session.coordinator is not None:
            session.coordinator.register_public_claim(
                session.controller,
                author,
                DiscussionOutput(public_message=req.content),
                message_id,
            )
            # Corrections are checked once, in code, for the whole table.
            session.coordinator.note_human_message(state, author, req.content)
        else:
            # Same conversion the coordinator uses, so a human claim becomes the
            # same kind of event whether or not AI players are in the game.
            drafts = build_claim_drafts(
                DiscussionOutput(public_message=req.content),
                PublicFactLedger(state),
                speaker_id=author,
            )
            _run(register_claim_drafts, session.controller, author, drafts, message_id)
        await orchestrator.after_human_chat(session)
    else:
        await orchestrator.after_human_private_chat(session, req.channel)
    return ChatResponse(message_id=message_id)


@router.post("/{session_id}/vote", response_model=OkResponse)
async def vote(
    session_id: str, req: VoteRequest, player_id: str | None = Query(default=None)
) -> OkResponse:
    session = _get_session(session_id)
    voter = _resolve_player_id(session, player_id)
    _run(session.controller.vote, voter, req.target_id)
    await orchestrator.after_human_vote(session)
    return OkResponse()


@router.post("/{session_id}/night-action", response_model=OkResponse)
async def night_action(
    session_id: str, req: NightActionRequest, player_id: str | None = Query(default=None)
) -> OkResponse:
    session = _get_session(session_id)
    actor = _resolve_player_id(session, player_id)
    _run(session.controller.submit_night_action, actor, req.action_type, req.target_id)
    await orchestrator.after_human_night_action(session)
    return OkResponse()


@router.post("/{session_id}/co", response_model=OkResponse)
async def co(
    session_id: str, req: CoRequest, player_id: str | None = Query(default=None)
) -> OkResponse:
    session = _get_session(session_id)
    claimant = _resolve_player_id(session, player_id)
    _run(session.controller.co, claimant, req.claimed_role)
    return OkResponse()


@router.post("/{session_id}/end-discussion", response_model=OkResponse)
async def end_discussion(session_id: str) -> OkResponse:
    session = _get_session(session_id)
    _run(session.controller.end_discussion)
    await orchestrator.after_voting_phase_entered(session)
    return OkResponse()


@router.post("/{session_id}/start-night", response_model=OkResponse)
async def start_night(session_id: str) -> OkResponse:
    session = _get_session(session_id)
    _run(session.controller.start_night)
    await orchestrator.after_night_phase_entered(session)
    return OkResponse()


@router.post("/{session_id}/resolve-night", response_model=OkResponse)
async def resolve_night(session_id: str) -> OkResponse:
    session = _get_session(session_id)
    _run(session.controller.resolve_night)
    return OkResponse()


@router.post("/{session_id}/resolve-votes", response_model=OkResponse)
async def resolve_votes(session_id: str) -> OkResponse:
    session = _get_session(session_id)
    _run(session.controller.resolve_votes)
    return OkResponse()


@router.post("/{session_id}/start-discussion", response_model=OkResponse)
async def start_discussion(session_id: str) -> OkResponse:
    session = _get_session(session_id)
    _run(session.controller.start_discussion)
    await orchestrator.after_discussion_phase_entered(session)
    return OkResponse()


@router.post("/{session_id}/pass-turn", response_model=OkResponse)
async def pass_turn(session_id: str) -> OkResponse:
    session = _get_session(session_id)
    if session.coordinator is not None:
        session.coordinator.resume_after_human(session)
        await session.coordinator.advance_discussion(session)
    return OkResponse()


@router.post("/{session_id}/discussion-control", response_model=OkResponse)
async def discussion_control(session_id: str, req: DiscussionControlRequest) -> OkResponse:
    session = _get_session(session_id)
    if req.action == "pause":
        session.discussion_pause_requested = True
        task = session.discussion_advance_task
        if task is None or task.done():
            session.discussion_pause_requested = False
            session.discussion_paused = True
        return OkResponse()
    if req.action not in ("resume", "step"):
        raise HTTPException(status_code=400, detail="unknown discussion control action")
    session.discussion_pause_requested = False
    session.discussion_paused = False
    session.discussion_step_budget = 1 if req.action == "step" else None
    if session.coordinator is not None:
        session.coordinator.resume_after_human(session)
        await session.coordinator.advance_discussion(session)
    return OkResponse()
