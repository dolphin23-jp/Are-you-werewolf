"""The second live Claude game (v3, seed 1, 2026-11-16), replayed offline.

The owner's review of that game (`artifacts/live-v3-claude-sonnet55-seeds1-3-
2026-11-16/seed1-review-causes.md`) asks what each seat saw when it spoke,
voted or acted. The prompts were not recorded in that game, so this module
replays the committed transcript through the real `GameController`,
`AICoordinator` and `ContextBuilder` -- the path a live game takes -- and
captures the exact v3 prompts at chosen moments. Every line, ballot and night
action comes from the record; no provider is called and nothing reaches the
network.

A moment is captured *before* the recorded utterance it names is applied, so
the prompt is the one that seat was answering. Captures are cached per process:
the whole game replays once (about ten seconds) however many tests read it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.ai.coordinator import AICoordinator
from app.ai.engine_mode import build_reasoning_runtime, model_decides
from app.ai.provider.scenario import ScenarioProvider
from app.ai.schemas import DiscussionOutput, PublicResultClaim
from app.engine.game import GameController, PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import PendingQuestion

TRANSCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "live-v3-claude-sonnet55-seeds1-3-2026-11-16"
    / "seed1"
    / "transcript-seed1.json"
)

_NIGHT_ACTIONS = {
    RoleName.SEER: "divine",
    RoleName.HUNTER: "guard",
    RoleName.WEREWOLF: "attack",
}


@dataclass(frozen=True)
class Moment:
    """The prompt `player_id` got for its `nth` utterance of `kind` on `day`.

    `kind` is the record's: discussion, vote, wolf_chat, freemason_chat or
    night_action. A discussion moment is built for the stage the record names.
    """

    tag: str
    day: int
    kind: str
    player_id: str
    nth: int = 0


@dataclass(frozen=True)
class CapturedPrompt:
    system: str
    user: str

    @property
    def text(self) -> str:
        return f"{self.system}\n\n{self.user}"


# The moments the seed-1 review names. Seats: p0 マコト(共有), p1 アカリ(霊媒),
# p2 ハルト(狼), p4 ソウタ(狐), p8 カイト(狼), p12 ダイキ(狼・偽占い),
# p13 ホノカ(共有), p15 アオイ(狂人・偽占い), p16 ケント(占い).
MOMENTS: tuple[Moment, ...] = (
    Moment("d1-m5-akari-initial", 1, "discussion", "p1", 0),
    Moment("d1-m14-souta-initial", 1, "discussion", "p4", 0),
    Moment("d1-m16-honoka-confirm", 1, "discussion", "p13", 0),
    Moment("d1-vote-tsumugi", 1, "vote", "p11", 0),
    Moment("n1-mason-makoto", 1, "freemason_chat", "p0", 0),
    Moment("n1-wolf-kaito", 1, "wolf_chat", "p8", 0),
    Moment("n1-guard-mio", 1, "night_action", "p5", 0),
    Moment("d2-m31-akari-immediate", 2, "discussion", "p1", 0),
    Moment("d2-m33-daiki-immediate", 2, "discussion", "p12", 0),
    Moment("d2-m34-aoi-immediate", 2, "discussion", "p15", 0),
    Moment("d2-m35-makoto-initial", 2, "discussion", "p0", 0),
    Moment("d2-vote-haruto", 2, "vote", "p2", 0),
    Moment("d2-vote-makoto", 2, "vote", "p0", 0),
    Moment("d2-vote-kento", 2, "vote", "p16", 0),
    Moment("d2-runoff-makoto", 2, "vote", "p0", 1),
    Moment("n2-attack-kaito", 2, "night_action", "p8", 0),
    Moment("d3-m45-akari-immediate", 3, "discussion", "p1", 0),
    Moment("d4-m61-haruto-initial", 4, "discussion", "p2", 0),
    Moment("d4-vote-kaito", 4, "vote", "p8", 0),
    Moment("d4-vote-honoka", 4, "vote", "p13", 0),
    Moment("d5-m81-kento-rebuttal", 5, "discussion", "p16", 1),
    Moment("d5-vote-kento", 5, "vote", "p16", 0),
)


@lru_cache(maxsize=1)
def load_transcript() -> dict[str, Any]:
    with TRANSCRIPT_PATH.open(encoding="utf-8") as stream:
        data: dict[str, Any] = json.load(stream)
    return data


@lru_cache(maxsize=1)
def captured_prompts() -> dict[str, CapturedPrompt]:
    """Every prompt in `MOMENTS`, by tag, from one replay of the game."""
    return replay(MOMENTS)


def replay(moments: Iterable[Moment]) -> dict[str, CapturedPrompt]:
    return _Replay(load_transcript(), tuple(moments)).run()


class _Replay:
    def __init__(self, record: dict[str, Any], moments: tuple[Moment, ...]) -> None:
        self._record = record
        self._moments = {(m.day, m.kind, m.player_id, m.nth): m for m in moments}
        names = record["names"]
        specs = [
            PlayerSpec(player_id=pid, name=names[pid], is_human=False) for pid in names
        ]
        self.controller = GameController(session_id="seed1-replay", player_specs=specs, seed=1)
        ids = [spec.player_id for spec in specs]
        state = self.controller.state
        self.coordinator = AICoordinator(
            state,
            ids,
            ScenarioProvider(seed=1),
            seed=1,
            reasoning=build_reasoning_runtime("v3", state, ids, seed=1),
            model_decides=model_decides("v3"),
            pacing_scale=0.0,
        )
        self.context = self.coordinator._context
        self.controller.start_game()
        roles = {pid: player.role.value for pid, player in state.players.items()}
        if roles != record["roles"]:
            raise AssertionError("the seed no longer deals the recorded roles")
        # The questions the record kept, by the message that asked them.
        self._questions: dict[str, list[PendingQuestion]] = {}
        for raw_list in record["final_state"]["pending_questions"].values():
            for raw in raw_list:
                self._questions.setdefault(raw["source_message_id"], []).append(
                    PendingQuestion(
                        asker=raw["asker"],
                        target=raw["target"],
                        question=raw["question"],
                        source_message_id=raw["source_message_id"],
                        day=raw["day"],
                        topic=raw.get("topic", ""),
                    )
                )
        self._captured: dict[str, CapturedPrompt] = {}

    @property
    def state(self) -> Any:
        return self.controller.state

    def run(self) -> dict[str, CapturedPrompt]:
        seen: dict[tuple[int, str, str], int] = {}
        for utterance in self._record["utterances"]:
            kind, pid, day = utterance["kind"], utterance["player_id"], utterance["day"]
            if kind in ("vote_change", "summary"):
                continue
            self._enter_phase_for(kind)
            nth = seen.get((day, kind, pid), 0)
            seen[(day, kind, pid)] = nth + 1
            moment = self._moments.get((day, kind, pid, nth))
            if moment is not None:
                self._captured[moment.tag] = self._prompt(moment, utterance)
            self._apply(utterance)
        missing = {m.tag for m in self._moments.values()} - set(self._captured)
        if missing:
            raise AssertionError(f"moments not reached: {sorted(missing)}")
        return self._captured

    # -- phases --

    def _enter_phase_for(self, kind: str) -> None:
        state = self.state
        if kind == "discussion" and state.phase == Phase.NIGHT:
            self.controller.resolve_night()
        if kind == "discussion" and state.phase == Phase.DAWN:
            self.controller.start_discussion()
        if kind == "vote" and state.phase == Phase.DISCUSSION:
            self.controller.end_discussion()
        if kind in ("night_action", "wolf_chat", "freemason_chat") and (
            state.phase == Phase.VOTE_RESULT
        ):
            asyncio.run(self.coordinator._generate_day_summary(self.controller, state))
            self.controller.start_night()

    # -- applying the record --

    def _apply(self, utterance: dict[str, Any]) -> None:
        kind, pid = utterance["kind"], utterance["player_id"]
        state = self.state
        if kind == "discussion":
            shown = [q for q in state.pending_questions.get(pid, []) if not q.served]
            memo = utterance["reasoning_memo"]
            stage = utterance["stage"] or "initial_view"
            if memo:
                # Stored before the line is posted, as `_speak` does.
                self.context.set_reasoning_memo(pid, memo, day=state.day, stage=stage)
            results = [PublicResultClaim(**raw) for raw in utterance["public_results"] or []]
            output = DiscussionOutput(
                public_message=utterance["text"],
                public_claim_role=utterance["public_claim_role"],
                contains_co_claim=bool(utterance["public_claim_role"]),
                public_results=results,
                reply_to=utterance["reply_to"],
            )
            message_id = self.controller.chat(
                pid, utterance["text"], "public", utterance["reply_to"]
            )
            for question in shown:
                question.served = True
            for question in self._questions.get(message_id, []):
                state.pending_questions.setdefault(question.target, []).append(question)
            self.context.record_key_point(state.day, message_id, pid, "", stage=stage)
            self.coordinator.register_public_claim(self.controller, pid, output, message_id)
            if memo and self.coordinator.reasoning is not None:
                self.coordinator.reasoning.record_stated_target(
                    pid, memo.get("execution_target"), state.day
                )
        elif kind in ("wolf_chat", "freemason_chat"):
            channel = "wolf" if kind == "wolf_chat" else "freemason"
            self.controller.chat(pid, utterance["text"], channel)
        elif kind == "vote":
            self.controller.vote(pid, utterance["target"])
            if len(state.pending_votes) == len(state.alive_ids()):
                self.controller.resolve_votes()
        elif kind == "night_action":
            action = _NIGHT_ACTIONS[state.players[pid].role]
            self.controller.submit_night_action(pid, action, utterance["target"])
            if utterance["day"] == 0:
                self.controller.resolve_night()

    # -- prompts, built the way the coordinator builds them --

    def _prompt(self, moment: Moment, utterance: dict[str, Any]) -> CapturedPrompt:
        state = self.state
        pid = moment.player_id
        coordinator = self.coordinator
        memo = coordinator._memo_for(state, pid)
        if moment.kind == "discussion":
            stage = utterance["stage"] or "initial_view"
            assert coordinator.reasoning is not None
            decision = coordinator.reasoning.discussion_decision(
                state,
                pid,
                pending_question=bool(coordinator._open_questions(pid)),
                under_pressure=stage.startswith(("rebuttal", "minority_review")),
            )
            system, messages = self.context.build_discussion_context(
                state,
                pid,
                stage,
                board_memo=memo,
                required_lines=coordinator._required_lines(state, pid, decision),
            )
        elif moment.kind == "vote":
            assert coordinator.reasoning is not None
            stated = coordinator.reasoning.stated_target(pid, state.day)
            system, messages = self.context.build_vote_context(
                state,
                pid,
                list(state.votable_ids(pid)),
                board_memo=memo,
                stated_target=stated if stated and state.players[stated].alive else None,
            )
        elif moment.kind == "freemason_chat":
            system, messages = self.context.build_freemason_chat_context(
                state, pid, board_memo=memo
            )
        elif moment.kind == "wolf_chat":
            system, messages = self.context.build_wolf_chat_context(state, pid, board_memo=memo)
        else:
            action = _NIGHT_ACTIONS[state.players[pid].role]
            if action == "attack":
                candidates = [
                    other
                    for other in state.alive_ids()
                    if state.players[other].role != RoleName.WEREWOLF
                ]
            else:
                candidates = [other for other in state.alive_ids() if other != pid]
            system, messages = self.context.build_night_action_context(
                state, pid, action, candidates, board_memo=memo
            )
        return CapturedPrompt(system=system, user="\n\n".join(m.content for m in messages))
