#!/usr/bin/env python3
"""Play one real-provider game under the v2 engine and report what it did.

Everything else in this repository is measured against `MockProvider`, which is
the right default -- tests must cost nothing and never touch a network. But a
mock never argues badly, never misquotes a ballot and never invents a schedule,
so "v2 reasons well" cannot be concluded from mock runs alone. This is the
script that goes and looks.

Manual on purpose. It spends real requests, so it is never run by CI and never
plays a real game from a test:

    WEREWOLF_LLM_PROVIDER=anthropic WEREWOLF_ANTHROPIC_API_KEY=... \\
        python scripts/live_reasoning_check.py --seed 11 --engine v3 --max-cost 15

    WEREWOLF_LLM_PROVIDER=luna LUNA_API_KEY=... \\
        python scripts/live_reasoning_check.py --seed 11

`--max-cost` (USD) is required for `anthropic`, as in `scripts/evaluate.py`.
A game the cap stops exits with code 3, but its report is still printed and
its partial transcript still written: that part of the game was paid for.

Run `scripts/check_llm.py` first -- one call to find a bad key beats a few
hundred. What this reports beyond the mock comparison:

* the correctness counters, which must be zero against a live model too;
* how often a model tried to substitute its own execution target and was
  overridden -- the number that says whether the enforcement is load-bearing;
* published accounts that contradicted the public calendar, which mocks do not
  generate at all;
* the transcript path, so a human can read the game and judge the parts no
  counter captures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.coordinator import AICoordinator  # noqa: E402
from app.ai.engine_mode import build_reasoning_runtime, model_decides  # noqa: E402
from app.ai.metrics import MetricsCollector  # noqa: E402
from app.ai.provider.base import LLMProvider  # noqa: E402
from app.ai.provider.budget import (  # noqa: E402
    BudgetedProvider,
    BudgetExceeded,
    EvaluationBudget,
)
from app.ai.provider.factory import build_llm_provider  # noqa: E402
from app.ai.reasoning.timeline import find_timeline_conflicts  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.engine.game import GameController, PlayerSpec  # noqa: E402
from app.engine.phases import Phase  # noqa: E402
from app.engine.roles import RoleName  # noqa: E402
from app.eval.report import BUDGET_STOP_BANNER  # noqa: E402
from app.eval.transcript import TranscriptRecorder  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    EXIT_BUDGET_STOPPED,
    _stop_stray_tasks,
    model_label,
)

MAX_LOOPS = 150
HUMAN_ID = "p0"
AI_IDS = [f"p{i}" for i in range(1, 17)]


class _Session:
    def __init__(self, controller: GameController, coordinator: AICoordinator) -> None:
        self.controller = controller
        self.coordinator = coordinator
        self.human_id = HUMAN_ID
        self.discussion_lock = asyncio.Lock()
        self.discussion_round: Any = None
        self.discussion_paused = False
        self.discussion_pause_requested = False
        self.discussion_step_budget: int | None = None


def _specs() -> list[PlayerSpec]:
    return [PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0)) for i in range(17)]


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


async def run(
    seed: int,
    transcript_path: Path | None,
    engine: str = "v2",
    request_budget: EvaluationBudget | None = None,
) -> dict[str, Any]:
    """Play the checked game and report it.

    The spending cap ending the game is a result, not an error: the report
    comes back with `status: budget_exhausted` and `budget_stop`, with its
    metrics, and the partial transcript is on disk. That game was paid for,
    and a stop that kept nothing could be neither read nor audited
    (approach-reset §11.6, run #11). The other seats' calls still in flight
    are cancelled and settled first, so the budget is final by the time the
    caller reads it.

    Any other failure still propagates, but only after the partial
    transcript, with the metrics so far, has been written.
    """
    tasks_before = asyncio.all_tasks()
    started = perf_counter()
    settings = get_settings()
    if settings.werewolf_llm_provider == "mock":
        raise SystemExit(
            "この確認は実プロバイダ用です。WEREWOLF_LLM_PROVIDER=anthropic"
            "(または休眠中の luna)を設定してください。"
        )
    metrics = MetricsCollector()
    inner = build_llm_provider(settings, seed=seed, metrics=metrics)
    provider: LLMProvider = (
        BudgetedProvider(inner, request_budget) if request_budget is not None else inner
    )
    controller = GameController(
        session_id=f"live-{seed}-{engine}", player_specs=_specs(), seed=seed
    )
    reasoning = build_reasoning_runtime(
        engine, controller.state, AI_IDS, seed=seed, metrics=metrics
    )
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        controller.state,
        AI_IDS,
        provider,
        seed=seed,
        recorder=recorder,
        reasoning=reasoning,
        model_decides=model_decides(engine),
        pacing_scale=0.0,
    )
    # The budget proxy is plumbing; the transcript names the provider measured.
    recorder.transcript.provider = type(inner).__name__
    session = _Session(controller, coordinator)
    rng = random.Random(seed)
    conflicts: list[str] = []

    def measured() -> dict[str, Any]:
        """The metrics so far, also stored on the transcript."""
        if reasoning is not None:
            reasoning.flush_metrics()  # publishes deltas, so safe to repeat
        metrics.total_game_wall_time = perf_counter() - started
        metrics.game_days = controller.state.day
        metrics.public_utterances = len(recorder.transcript.by_kind("discussion"))
        summary = metrics.summary()
        recorder.transcript.metrics = {
            **summary,
            **(reasoning.metrics() if reasoning is not None else {}),
        }
        return summary

    def write_transcript(path: Path) -> None:
        transcript = recorder.finalize(controller.get_debug_view())
        path.write_text(
            json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    budget_stop: str | None = None
    try:
        controller.start_game()
        for _ in range(MAX_LOOPS):
            state = controller.state
            if state.phase is Phase.GAME_OVER:
                break
            if state.phase is Phase.NIGHT:
                _human_night(controller, rng)
                await coordinator.run_night_phase(session)
            elif state.phase is Phase.DAWN:
                controller.start_discussion()
            elif state.phase is Phase.DISCUSSION:
                await coordinator.run_discussion_round(session)
                if reasoning is not None:
                    reasoning.refresh(state)
                    conflicts.extend(
                        conflict.explanation
                        for conflict in find_timeline_conflicts(reasoning.observations)
                    )
                controller.end_discussion()
            elif state.phase in (Phase.VOTING, Phase.RUNOFF):
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
    except BudgetExceeded as exc:
        budget_stop = str(exc)
        # The refused call leaves its `asyncio.gather` while the other seats'
        # calls are still in flight, each holding a reservation. Settle them
        # here, or the caller writes a budget that keeps growing afterwards.
        await _stop_stray_tasks(tasks_before)
    except BaseException:
        # Ctrl-C, a bug, a provider error: the game is lost, but what it said
        # and measured so far is kept. Synchronous, so it also runs while
        # this task is being cancelled.
        if transcript_path is not None:
            try:
                measured()
                write_transcript(transcript_path)
            except Exception as write_error:  # never hide what ended the game
                print(f"途中経過を書けませんでした: {write_error}", file=sys.stderr)
        raise

    metric_summary = measured()
    codes = coordinator.validation.codes()
    report: dict[str, Any] = {
        "seed": seed,
        "engine": engine,
        "game_id": controller.state.session_id,
        "provider": settings.werewolf_llm_provider,
        "model": model_label(provider, settings),
        "days": controller.state.day,
        "finished": controller.state.phase is Phase.GAME_OVER,
        "llm_requests": metric_summary.get("total_calls", 0),
        "http_requests": metric_summary.get("http_requests", 0),
        "tokens": metric_summary.get("tokens", {}),
        "operational_metrics": metric_summary,
        # Must be zero against a live model exactly as against a mock.
        "dead_target_selections": sum(
            code.endswith("_dead") or code.endswith("target_invalid") for code in codes
        ),
        "public_fact_flips": codes.count("result_polarity_conflict"),
        "vote_history_misreads": codes.count("vote_history_misread"),
        # Does the enforcement carry weight, or was the model going to agree?
        "vote_plan_mismatches": len(coordinator.validation.vote_plan_mismatches),
        "timeline_conflicts": sorted(set(conflicts)),
        **(reasoning.metrics() if reasoning is not None else {}),
    }
    if budget_stop is not None:
        report["status"] = "budget_exhausted"
        report["budget_stop"] = budget_stop
    if transcript_path is not None:
        write_transcript(transcript_path)
        report["transcript"] = str(transcript_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--engine", choices=("legacy", "v2", "v3"), default="v2")
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="どこへ解析用ログを書くか。指定しなければ書かない。",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="推定費用の上限(USD)。anthropic では必須。超えそうなら送らずに止めて終了コード3",
    )
    args = parser.parse_args(argv)
    budget = budget_for(args, parser, get_settings())
    try:
        report = asyncio.run(run(args.seed, args.transcript, args.engine, budget))
    except BudgetExceeded as exc:
        print(f"{BUDGET_STOP_BANNER}: {exc}", file=sys.stderr)
        return EXIT_BUDGET_STOPPED
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report.get("budget_stop"):
        # The partial game is printed above and its transcript kept.
        print(f"{BUDGET_STOP_BANNER}: {report['budget_stop']}", file=sys.stderr)
        return EXIT_BUDGET_STOPPED
    return 0


def budget_for(
    args: argparse.Namespace, parser: argparse.ArgumentParser, settings: Settings
) -> EvaluationBudget | None:
    """The cap for a single checked game; required where the provider prices itself."""
    if args.max_cost is not None and not args.max_cost > 0:
        parser.error("--max-cost は0より大きいUSDの金額にしてください。")
    if settings.werewolf_llm_provider == "anthropic" and args.max_cost is None:
        parser.error("anthropic で動かすときは --max-cost(USD)が必須です。例: --max-cost 15")
    if args.max_cost is None:
        return None
    if settings.werewolf_llm_provider == "luna":
        parser.error(
            "luna では --max-cost を計算できません(単価が無い)。"
            "単価つきの上限は scripts/live_ab_reasoning_check.py を使ってください。"
        )
    return EvaluationBudget(max_requests=10_000_000, max_estimated_cost=args.max_cost)


if __name__ == "__main__":
    raise SystemExit(main())
