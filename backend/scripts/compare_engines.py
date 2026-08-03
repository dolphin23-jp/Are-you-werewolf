"""Run the same seeded game under both reasoning engines and compare.

The v2 claim is that moving decisions into code cuts requests without costing
reasoning quality. That is a measurement, not an opinion, so this script plays
identical boards both ways and prints the numbers the roadmap asks for --
requests, HTTP calls, tokens, and the correctness counters that must stay at
zero.

Mock provider only: the comparison is about call counts and decisions, and a
live endpoint would add cost and noise without changing either.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any

from app.ai.coordinator import AICoordinator
from app.ai.metrics import MetricsCollector
from app.ai.provider.mock import MockProvider
from app.ai.reasoning.facts import PublicFactLedger
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import DiscussionOutput
from app.engine.game import GameController, PlayerSpec
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.engine.state import DeathCause
from app.eval.reasoning_analyzer import ReasoningTranscriptAnalyzer
from app.eval.transcript import TranscriptRecorder

MAX_LOOPS = 150
HUMAN_ID = "p0"


@dataclass
class EngineRun:
    engine: str
    seed: int
    llm_requests: int = 0
    http_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dead_target_selections: int = 0
    public_fact_flips: int = 0
    vote_history_misreads: int = 0
    vote_plan_mismatches: int = 0
    votes_cast: int = 0
    days: int = 0
    top_candidate_concentration: list[float] = field(default_factory=list)
    opinion_spread: list[float] = field(default_factory=list)
    target_distribution_by_profile: dict[str, dict[str, int]] = field(default_factory=dict)
    reasoning_metrics: dict[str, Any] = field(default_factory=dict)
    winner: str | None = None
    executions: list[str] = field(default_factory=list)
    role_survival_days: dict[str, list[int]] = field(default_factory=dict)
    public_utterances: int = 0
    reasoning_quality: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "seed": self.seed,
            "llm_requests": self.llm_requests,
            "http_requests": self.http_requests,
            "tokens": self.prompt_tokens + self.completion_tokens,
            "dead_target_selections": self.dead_target_selections,
            "public_fact_flips": self.public_fact_flips,
            "vote_history_misreads": self.vote_history_misreads,
            "vote_plan_mismatch_rate": round(
                self.vote_plan_mismatches / max(self.votes_cast, 1), 3
            ),
            "days": self.days,
            "winner": self.winner,
            "executions": self.executions,
            "role_survival_days": self.role_survival_days,
            "public_utterances": self.public_utterances,
            "reasoning_quality": self.reasoning_quality,
            "mean_top_candidate_concentration": _mean(self.top_candidate_concentration),
            "mean_opinion_spread": _mean(self.opinion_spread),
            "target_distribution_by_profile": self.target_distribution_by_profile,
            **self.reasoning_metrics,
        }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _specs() -> list[PlayerSpec]:
    return [PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0)) for i in range(17)]


async def play(engine: str, seed: int) -> EngineRun:
    metrics = MetricsCollector()
    provider = MockProvider(seed=seed, metrics=metrics)
    controller = GameController(session_id=f"cmp-{engine}", player_specs=_specs(), seed=seed)
    # p0 is the human seat. Listing it as an AI too made the coordinator wait
    # for a turn the human stub was also supposed to take, and put the human's
    # own sentences into the count of messages the AIs "considered".
    ai_ids = [f"p{i}" for i in range(1, 17)]
    reasoning = (
        ReasoningRuntime(controller.state, ai_ids, seed=seed, metrics=metrics)
        if engine == "v2"
        else None
    )
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        controller.state,
        ai_ids,
        provider,
        seed=seed,
        recorder=recorder,
        reasoning=reasoning,
        pacing_scale=0.0,
    )
    session = _Session(controller, coordinator)

    controller.start_game()
    run = EngineRun(engine=engine, seed=seed)
    rng = random.Random(seed)

    # The human seat is played by a seeded stub. Both engines get the identical
    # sequence, so any difference in the numbers is the engine's.
    for _ in range(MAX_LOOPS):
        state = controller.state
        if state.phase is Phase.GAME_OVER:
            break
        if state.phase is Phase.NIGHT:
            _human_night(controller, rng)
            await coordinator.run_night_phase(session)
        elif state.phase is Phase.DAWN:
            controller.start_discussion()
            run.days += 1
        elif state.phase is Phase.DISCUSSION:
            _human_speak(controller, coordinator, rng)
            await coordinator.run_discussion_round(session)
            controller.end_discussion()
        elif state.phase in (Phase.VOTING, Phase.RUNOFF):
            if reasoning is not None:
                reasoning.refresh(state)
                run.opinion_spread.append(reasoning.opinion_spread())
                run.target_distribution_by_profile = reasoning.distribution_by_profile()
            human = state.players[HUMAN_ID]
            if human.alive:
                candidates = state.votable_ids(HUMAN_ID)
                if candidates:
                    controller.vote(HUMAN_ID, rng.choice(candidates))
            await coordinator.generate_all_votes(session)
        elif state.phase is Phase.VOTE_RESULT:
            controller.start_night()
        else:
            break

    if reasoning is not None:
        reasoning.flush_metrics()
        run.reasoning_metrics = dict(reasoning.metrics())
        run.reasoning_metrics.pop("target_distribution_by_profile", None)
    summary = metrics.summary()
    run.llm_requests = summary.get("total_calls", 0)
    run.http_requests = summary.get("http_requests", 0)
    tokens = summary.get("tokens") or {}
    run.prompt_tokens = tokens.get("prompt", 0) or 0
    run.completion_tokens = tokens.get("completion", 0) or 0

    codes = coordinator.validation.codes()
    run.dead_target_selections = sum(
        code.endswith("_dead") or code.endswith("target_invalid") for code in codes
    )
    run.public_fact_flips = codes.count("result_polarity_conflict")
    run.vote_history_misreads = codes.count("vote_history_misread")
    run.vote_plan_mismatches = len(coordinator.validation.vote_plan_mismatches)
    run.votes_cast = len(controller.state.vote_records)
    run.top_candidate_concentration = _concentration(controller)
    run.winner = controller.state.winner.value if controller.state.winner else None
    run.executions = [
        record.player_id
        for record in controller.state.death_records
        if record.cause is DeathCause.EXECUTED
    ]
    for player in controller.state.players.values():
        survived = player.death_day if player.death_day is not None else controller.state.day
        run.role_survival_days.setdefault(player.role.value, []).append(survived)
    recorder.transcript.metrics = {
        **summary,
        **(reasoning.metrics() if reasoning is not None else {}),
    }
    transcript = recorder.finalize(controller.get_debug_view())
    run.public_utterances = len(transcript.by_kind("discussion"))
    run.reasoning_quality = ReasoningTranscriptAnalyzer().analyze(transcript).to_dict()
    return run


def _human_speak(
    controller: GameController, coordinator: AICoordinator, rng: random.Random
) -> None:
    """One human turn per day, through the same path the chat route uses.

    The old stub said "よろしくお願いします" straight into `controller.chat`,
    which meant every human-facing metric in this report was measured against a
    sentence containing no claim, delivered by a route that never told the
    reasoning layer it had happened. Both halves are fixed here: the message
    carries something the record can check, and it goes through
    `register_public_claim` / `note_human_message` exactly as the API does.
    """
    state = controller.state
    if not state.players[HUMAN_ID].alive:
        return
    text = _human_line(controller, rng)
    message_id = controller.chat(HUMAN_ID, text, "public")
    coordinator.register_public_claim(
        controller, HUMAN_ID, DiscussionOutput(public_message=text), message_id
    )
    coordinator.note_human_message(state, HUMAN_ID, text, message_id)


def _human_line(controller: GameController, rng: random.Random) -> str:
    """A rotating script: an accusation, a correction of the record, advice.

    The correction is built from the ledger so it is genuinely checkable -- it
    denies a ballot that did not happen and asserts the one that did, which is
    the case a confirmed correction is supposed to cover.
    """
    state = controller.state
    ledger = PublicFactLedger(state)
    others = [pid for pid in state.alive_ids() if pid != HUMAN_ID]
    if not others:
        return "様子を見ます。"
    recorded = ledger.vote_of(HUMAN_ID, state.day - 1) if state.day > 1 else None
    if recorded is not None:
        wrong = next((pid for pid in others if pid != recorded.target_id), recorded.target_id)
        if wrong != recorded.target_id:
            return (
                f"{state.day - 1}日目、私は{wrong}には投票していません。"
                f"{recorded.target_id}へ投票しました。"
            )
    if state.day % 2 == 0:
        return f"{rng.choice(others)}が怪しいと思います。"
    return "占い師を残すべきです。まだ吊る必要はありません。"


def _human_night(controller: GameController, rng: random.Random) -> None:
    state = controller.state
    human = state.players[HUMAN_ID]
    if not human.alive:
        return
    candidates = [pid for pid in state.alive_ids() if pid != HUMAN_ID]
    if not candidates:
        return
    if human.role is RoleName.SEER:
        controller.submit_night_action(HUMAN_ID, "divine", rng.choice(candidates))
    elif state.day > 0 and human.role is RoleName.HUNTER:
        controller.submit_night_action(HUMAN_ID, "guard", rng.choice(candidates))
    elif state.day > 0 and human.role is RoleName.WEREWOLF and controller.alpha_wolf_id == HUMAN_ID:
        prey = [pid for pid in candidates if state.players[pid].role is not RoleName.WEREWOLF]
        if prey:
            controller.submit_night_action(HUMAN_ID, "attack", rng.choice(prey))


def _concentration(controller: GameController) -> list[float]:
    per_day: dict[int, dict[str, int]] = {}
    for vote in controller.state.vote_records:
        bucket = per_day.setdefault(vote.day, {})
        bucket[vote.target_id] = bucket.get(vote.target_id, 0) + 1
    return [max(counts.values()) / sum(counts.values()) for counts in per_day.values() if counts]


class _Session:
    """Minimal stand-in for the API session the coordinator expects."""

    def __init__(self, controller: GameController, coordinator: AICoordinator) -> None:
        self.controller = controller
        self.coordinator = coordinator
        self.human_id = "p0"
        self.discussion_lock = asyncio.Lock()
        self.discussion_round: Any = None
        self.discussion_paused = False
        self.discussion_pause_requested = False
        self.discussion_step_budget: int | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for engine in ("legacy", "v2"):
            rows.append(asyncio.run(play(engine, seed)).as_dict())

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    legacy = [row for row in rows if row["engine"] == "legacy"]
    v2 = [row for row in rows if row["engine"] == "v2"]
    print(f"{'metric':<34}{'legacy':>12}{'v2':>12}{'change':>12}")
    for key in ("llm_requests", "http_requests", "tokens"):
        before = sum(row[key] for row in legacy)
        after = sum(row[key] for row in v2)
        delta = f"{(after - before) / before * 100:+.0f}%" if before else "n/a"
        print(f"{key:<34}{before:>12}{after:>12}{delta:>12}")
    for key in (
        "dead_target_selections",
        "public_fact_flips",
        "vote_history_misreads",
    ):
        before = sum(row[key] for row in legacy)
        after = sum(row[key] for row in v2)
        print(f"{key:<34}{before:>12}{after:>12}{'':>12}")
    for key in (
        "mean_top_candidate_concentration",
        "mean_opinion_spread",
        # A stated candidate the ballot did not name. Not a correctness counter:
        # the board moves between speaking and voting, and changing your mind is
        # legal. It is here because a rate near zero and a rate near one mean
        # different things and neither is visible from the others.
        "vote_plan_mismatch_rate",
    ):
        before = _mean([row[key] for row in legacy])
        after = _mean([row[key] for row in v2])
        print(f"{key:<34}{before:>12}{after:>12}{'':>12}")
    for key in sorted(set(v2[0]) - set(legacy[0])):
        value = v2[0][key]
        if isinstance(value, int | float):
            print(f"{key:<34}{'n/a':>12}{_mean([row[key] for row in v2]):>12}{'':>12}")


if __name__ == "__main__":
    main()
