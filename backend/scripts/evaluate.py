#!/usr/bin/env python3
"""Plays N full games and reports on AI quality and running cost.

    # ハーネスの動作確認(通信なし・費用ゼロ)
    python scripts/evaluate.py --games 2 --provider mock

    # 実際の Claude で評価(--max-cost は必須。USD)
    python scripts/evaluate.py --games 1 --provider anthropic --max-cost 15

    # 休眠中の選択肢: gpt-6-luna(単価を渡すと金額換算)
    python scripts/evaluate.py --games 3 --provider luna --judge \
        --price-in 0.15 --price-out 0.60

Writes to --out (default `eval-out/`):
  report.md              集計レポート
  summary.json           機械可読な集計(試合ごとの数値つき)
  transcript-seedN.md    目視確認用の対戦記録
  transcript-seedN.json  生データ

While a game is running the log prints each utterance, a marker per phase with
the call health so far, and a "生成中" line after two quiet minutes. A partial
transcript-seedN.md/.json is rewritten at every phase, so a run that is
cancelled or times out still leaves what the table said.

Spending cap: with `--max-cost` one `EvaluationBudget` is shared by every game
and the judge. When the next request could cross it, that request is not sent:
the run stops, writes the partial transcript and a report that says it was
stopped (`費用上限に達したため中断`), and exits with code 3. `--max-cost` is
required for `anthropic`, whose cost the provider prices itself.

Each game keeps its own `MetricsCollector`, so summary.json can show one game's
failures or cost next to another's instead of only their pooled total.

Cost warning: one 17-player game is on the order of a few hundred LLM
calls, so start with --games 1 against a real endpoint and read the
reported token counts before scaling up.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.coordinator import AICoordinator  # noqa: E402
from app.ai.engine_mode import (  # noqa: E402
    build_reasoning_runtime,
    model_decides,
    uses_reasoning_runtime,
)
from app.ai.metrics import CallRecord, MetricsCollector  # noqa: E402
from app.ai.provider.base import LLMProvider  # noqa: E402
from app.ai.provider.budget import (  # noqa: E402
    BudgetedProvider,
    BudgetExceeded,
    EvaluationBudget,
)
from app.ai.provider.factory import build_llm_provider  # noqa: E402
from app.config import Settings, settings_for_provider  # noqa: E402
from app.engine.game import GameController, PlayerSpec  # noqa: E402
from app.engine.phases import Phase  # noqa: E402
from app.eval.analyzers import AnalysisResult, analyze  # noqa: E402
from app.eval.call_log import CallLogEntry, CallLogProvider, write_call_log  # noqa: E402
from app.eval.judge import judge_transcript  # noqa: E402
from app.eval.progress import (  # noqa: E402
    HEARTBEAT_SECONDS,
    LiveTranscriptRecorder,
    RunProgress,
    write_transcript_files,
)
from app.eval.reasoning_analyzer import ReasoningTranscriptAnalyzer  # noqa: E402
from app.eval.release_gate import HARD_FAILURE_FIELDS  # noqa: E402
from app.eval.report import (  # noqa: E402
    BUDGET_STOP_BANNER,
    BudgetStopPhase,
    render_report,
)
from app.eval.transcript import GameTranscript, TranscriptRecorder  # noqa: E402
from app.eval.transcript_render import team_label  # noqa: E402

HUMAN_ID = "p0"
MAX_LOOPS = 200

PROVIDERS = ("mock", "scenario", "luna", "anthropic")
_OFFLINE_PROVIDERS = frozenset({"mock", "scenario"})

# Exit status when the spending cap stopped the run. Distinct from argparse's
# 2 (bad arguments) and from a crash's 1, so a workflow can tell "stopped on
# purpose, partial output written" from "broken".
EXIT_BUDGET_STOPPED = 3

# The cap is in dollars. The request count still has to be bounded by
# something, but it is not what is being limited, so it is set out of reach.
_UNLIMITED_REQUESTS = 10_000_000

AI_NAMES = [
    "マコト",
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


def _make_specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=AI_NAMES[i], is_human=False) for i in range(17)
    ]


@dataclass
class GameRun:
    """One game and everything measured about it.

    `budget_stop` is the `BudgetExceeded` message when the spending cap ended
    this game early; the transcript then holds what was played until then.
    """

    seed: int
    transcript: GameTranscript
    metrics: MetricsCollector
    model: str
    engine: str
    finished: bool
    wall_time_seconds: float
    validation: dict[str, int]
    budget_stop: str | None = None
    # Every model call of the game, with its prompt and answer (`app.eval.call_log`).
    call_log: list[CallLogEntry] = field(default_factory=list)


def model_label(provider: object | None, settings: Settings) -> str:
    """What answered: the provider's own `model` when it has one (the
    Anthropic provider, a budget proxy around it), else the setting that
    configures it. The offline doubles have no model."""
    model = getattr(provider, "model", None)
    if isinstance(model, str) and model:
        return model
    name = settings.werewolf_llm_provider
    if name == "anthropic":
        return settings.werewolf_anthropic_model
    if name == "luna":
        return settings.luna_model
    return "(mock)"


def _validation_counts(coordinator: AICoordinator) -> dict[str, int]:
    """The repairs `scripts/live_reasoning_check.py` reports, from the same log.

    Each is a mistake the model made that the code caught before it reached
    the table -- the public record stayed right, but the count says how much
    weight the enforcement is carrying.
    """
    codes = coordinator.validation.codes()
    return {
        "dead_target_selections": sum(
            code.endswith("_dead") or code.endswith("target_invalid") for code in codes
        ),
        "public_fact_flips": codes.count("result_polarity_conflict"),
        "vote_history_misreads": codes.count("vote_history_misread"),
        "vote_plan_mismatches": len(coordinator.validation.vote_plan_mismatches),
    }


async def _stop_stray_tasks(started_with: set[asyncio.Task[Any]]) -> None:
    """Cancel and await what the game left running.

    `BudgetExceeded` leaves its `asyncio.gather` while the sibling calls (the
    other seats' votes, say) keep going. The budget would refuse their next
    request anyway, but one already in flight could still settle after the
    report has read the budget, or post to a table that is being written out.
    """
    current = asyncio.current_task()
    stray = [task for task in asyncio.all_tasks() if task is not current]
    stray = [task for task in stray if task not in started_with]
    for task in stray:
        task.cancel()
    await asyncio.gather(*stray, return_exceptions=True)


async def play_game(
    seed: int,
    settings: Settings,
    metrics: MetricsCollector,
    *,
    emit: Callable[[str], None] | None = None,
    out_dir: Path | None = None,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    budget: EvaluationBudget | None = None,
) -> GameRun:
    """Plays one all-AI game. `emit` turns on the live log; `out_dir` also keeps a
    partial transcript there. Neither changes how the game is played.

    With `budget`, every call goes through a `BudgetedProvider` on it. A
    `BudgetExceeded` ends the game where it is and comes back as
    `GameRun.budget_stop`; any other error propagates as before.
    """
    tasks_before = asyncio.all_tasks()
    started = perf_counter()
    inner = build_llm_provider(settings, seed=seed, metrics=metrics)
    budgeted: LLMProvider = BudgetedProvider(inner, budget) if budget is not None else inner
    specs = _make_specs()
    names = {spec.player_id: spec.name for spec in specs}
    controller = GameController(session_id=f"eval-{seed}", player_specs=specs, seed=seed)
    # Outermost, so a call the budget refused is logged too. It only observes.
    call_log = CallLogProvider(
        budgeted, context=lambda: (controller.state.day, controller.state.phase.value)
    )
    provider: LLMProvider = call_log
    ai_ids = [s.player_id for s in specs]
    live = LiveTranscriptRecorder(emit) if emit is not None else None
    recorder = live if live is not None else TranscriptRecorder()
    # The engine flag was never read here, so every "AI評価" run -- including
    # the GitHub Action -- measured the legacy engine whatever the deployment
    # was configured to play. Resolve it the same way the API does.
    engine = settings.werewolf_reasoning_engine
    reasoning = build_reasoning_runtime(
        engine, controller.state, ai_ids, seed=seed, metrics=metrics
    )
    coordinator = AICoordinator(
        controller.state,
        ai_ids,
        provider,
        seed=seed,
        recorder=recorder,
        reasoning=reasoning,
        model_decides=model_decides(engine),
        pacing_scale=0.0,
    )
    # The budget proxy is harness plumbing, not the provider under test, and
    # the report keys its "this was a mock" warning off this name.
    recorder.transcript.provider = type(inner).__name__
    session = SimpleNamespace(
        controller=controller,
        human_id=HUMAN_ID,
        coordinator=coordinator,
        discussion_lock=asyncio.Lock(),
    )
    def save_call_log() -> None:
        if out_dir is not None:
            write_call_log(call_log.entries, list(metrics.records), names, out_dir, seed)

    progress = (
        RunProgress(
            live,
            metrics,
            seed=seed,
            out_dir=out_dir,
            heartbeat_seconds=heartbeat_seconds,
            on_save=save_call_log,
        )
        if live is not None
        else None
    )
    heartbeat = asyncio.create_task(progress.heartbeat()) if progress is not None else None
    budget_stop: str | None = None
    elapsed = 0.0
    try:
        controller.start_game()
        for _ in range(MAX_LOOPS):
            phase = controller.state.phase
            if progress is not None:
                progress.phase(controller.state.day, phase, controller.get_debug_view)
            if phase == Phase.GAME_OVER:
                break
            if phase == Phase.NIGHT:
                await coordinator.run_night_phase(session)
            elif phase == Phase.DAWN:
                controller.start_discussion()
            elif phase == Phase.DISCUSSION:
                await coordinator.run_discussion_round(session)
                controller.end_discussion()
            elif phase in (Phase.VOTING, Phase.RUNOFF):
                await coordinator.generate_all_votes(session)
            elif phase == Phase.VOTE_RESULT:
                controller.start_night()
            else:
                raise RuntimeError(f"unexpected phase {phase}")
    except BudgetExceeded as exc:
        budget_stop = str(exc)
        if live is not None:
            live.log(f"── {BUDGET_STOP_BANNER} ── {exc}")
        await _stop_stray_tasks(tasks_before)
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        if reasoning is not None:
            reasoning.flush_metrics()
        # Accumulated, not assigned, so a collector shared by several games
        # still pools them -- the same rule as `MetricsCollector.merged`.
        elapsed = perf_counter() - started
        metrics.total_game_wall_time += elapsed
        metrics.game_days += controller.state.day
        metrics.public_utterances += len(recorder.transcript.by_kind("discussion"))
        if progress is not None and controller.state.phase != Phase.GAME_OVER:
            progress.interrupted(controller.get_debug_view)

    finished = controller.state.phase == Phase.GAME_OVER
    return GameRun(
        seed=seed,
        transcript=recorder.finalize(controller.get_debug_view()),
        metrics=metrics,
        model=model_label(provider, settings),
        engine=engine,
        finished=finished,
        wall_time_seconds=elapsed,
        validation=_validation_counts(coordinator),
        budget_stop=budget_stop,
        call_log=list(call_log.entries),
    )


async def play_one_game(
    seed: int,
    settings: Settings,
    metrics: MetricsCollector,
    *,
    emit: Callable[[str], None] | None = None,
    out_dir: Path | None = None,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> GameTranscript:
    """`play_game` for callers that only want the transcript."""
    run = await play_game(
        seed,
        settings,
        metrics,
        emit=emit,
        out_dir=out_dir,
        heartbeat_seconds=heartbeat_seconds,
    )
    return run.transcript


def hard_correctness(run: GameRun) -> tuple[dict[str, int] | None, dict[str, Any] | None]:
    """`(hard_failures, reasoning_quality)` for one game; `(None, None)` for legacy.

    The same computation the live qualification uses: the reasoning-quality
    report is recomputed from the transcript's audit records
    (`ReasoningTranscriptAnalyzer`, as `scripts/live_ab_reasoning_check.py`
    does on the transcript `scripts/live_reasoning_check.py` writes), and the
    hard failures are exactly the release gate's `HARD_FAILURE_FIELDS`. legacy
    keeps no audit records, so a zero there would mean "not measured", not
    "passed" -- it is reported as None instead.
    """
    if not uses_reasoning_runtime(run.engine):
        return None, None
    quality = ReasoningTranscriptAnalyzer().analyze(run.transcript)
    return {field: int(getattr(quality, field)) for field in HARD_FAILURE_FIELDS}, dict(
        quality.to_dict()
    )


def describe_efforts(records: Sequence[CallRecord], schema: str) -> str | None:
    """The effort a schema was sent with: one level, or each level with its count."""
    counts = Counter(r.effort for r in records if r.schema == schema and r.effort)
    if not counts:
        return None
    if len(counts) == 1:
        return next(iter(counts))
    return " / ".join(f"{effort} {count}" for effort, count in counts.most_common())


def metrics_summary(
    collector: MetricsCollector, price_in: float, price_out: float
) -> dict[str, Any]:
    """`collector.summary()` plus the effort each schema was sent with."""
    summary = collector.summary(price_in, price_out)
    records = list(collector.records)
    for schema, data in (summary.get("by_schema") or {}).items():
        data["effort"] = describe_efforts(records, schema)
    return summary


def per_game_entry(run: GameRun, price_in: float, price_out: float) -> dict[str, Any]:
    hard, quality = hard_correctness(run)
    return {
        "seed": run.seed,
        "winner": run.transcript.final_state.get("winner"),
        "finished": run.finished,
        "wall_time_seconds": round(run.wall_time_seconds, 3),
        "public_utterances": run.metrics.public_utterances,
        "days": run.metrics.game_days,
        "metrics": metrics_summary(run.metrics, price_in, price_out),
        "hard_failures": hard,
        "reasoning_quality": quality,
        "validation": run.validation,
        "budget_stop": run.budget_stop,
    }


def commit_sha() -> str | None:
    """The commit the run measured: Actions' `GITHUB_SHA`, else the checkout's
    HEAD. None when neither is available -- a missing SHA must not fail a run
    whose games have already been paid for."""
    sha = os.environ.get("GITHUB_SHA", "").strip()
    if sha:
        return sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--games", type=int, default=1)
    # No default: passing one explicitly to Settings would override
    # WEREWOLF_LLM_PROVIDER from the environment, so a CLI *default* would
    # silently beat an env var the operator deliberately set. Settings
    # supplies the real default (mock) when this is omitted.
    parser.add_argument("--provider", choices=PROVIDERS, default=None)
    parser.add_argument("--seed", type=int, default=1, help="最初のゲームのシード")
    parser.add_argument("--out", type=Path, default=Path("eval-out"))
    parser.add_argument("--judge", action="store_true", help="日本語/人格をLLMで採点する(追加費用)")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="判定に使うモデル(既定: 本体と同じ)。anthropic なら WEREWOLF_ANTHROPIC_MODEL、"
        "luna なら LUNA_MODEL を判定のときだけ置き換える",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="推定費用の上限(USD)。対戦と判定の全呼び出しで共有し、超えそうなら送らずに"
        "止めて終了コード3。anthropic では必須",
    )
    parser.add_argument("--price-in", type=float, default=0.0, help="入力100万トークンあたり単価")
    parser.add_argument("--price-out", type=float, default=0.0, help="出力100万トークンあたり単価")
    return parser


def resolve_settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Settings:
    """Settings for the run, refusing combinations whose spending is unbounded."""
    if args.games < 1:
        parser.error("--games は1以上にしてください。")
    settings = settings_for_provider(args.provider)
    provider = settings.werewolf_llm_provider
    if args.max_cost is not None and not args.max_cost > 0:
        parser.error("--max-cost は0より大きいUSDの金額にしてください。")
    # Resolved, not just the flag: WEREWOLF_LLM_PROVIDER=anthropic in the
    # environment spends exactly as much as --provider anthropic does.
    if provider == "anthropic" and args.max_cost is None:
        parser.error(
            "anthropic で評価するときは --max-cost(USD)が必須です。"
            "例: --max-cost 15(1戦の費用はまだ計測していません。"
            "docs/evaluation.md を参照)"
        )
    if (
        provider == "luna"
        and args.max_cost is not None
        and not (args.price_in > 0 or args.price_out > 0)
    ):
        parser.error(
            "luna で --max-cost を効かせるには --price-in / --price-out が必要です"
            "(単価が無いと費用を計算できず、上限が効きません)。"
        )
    return settings


def judge_settings(args: argparse.Namespace, settings: Settings) -> Settings:
    """The game's settings with the judge's model swapped in, when one is given."""
    if not args.judge_model:
        return settings
    overrides: dict[str, Any] = {"werewolf_llm_provider": settings.werewolf_llm_provider}
    if settings.werewolf_llm_provider == "anthropic":
        overrides["werewolf_anthropic_model"] = args.judge_model
    elif settings.werewolf_llm_provider == "luna":
        overrides["luna_model"] = args.judge_model
    else:
        return settings  # the offline doubles have no model to swap
    return Settings(**overrides)


def build_budget(args: argparse.Namespace) -> EvaluationBudget | None:
    if args.max_cost is None:
        return None
    priced = args.price_in > 0 or args.price_out > 0
    return EvaluationBudget(
        max_requests=_UNLIMITED_REQUESTS,
        max_estimated_cost=args.max_cost,
        # luna prices its usage with these; anthropic prices from its own table
        # and settles exact costs, which take precedence.
        input_price_per_million=args.price_in if priced else None,
        output_price_per_million=args.price_out if priced else None,
    )


def _budget_line(budget: EvaluationBudget | None) -> str:
    if budget is None or budget.max_estimated_cost is None:
        return ""
    spent = budget.estimated_cost
    spent_text = f"${spent:.4f}" if spent is not None else "不明"
    return f" 費用 {spent_text} / 上限 ${budget.max_estimated_cost:.2f}"


def winner_line(winner: object) -> str:
    """The live log's line for a finished game, the faction named as the record names it."""
    return f"    勝者: {team_label(winner) if winner else '(未決着)'}"


def _print_line(line: str) -> None:
    # Flushed on purpose: stdout is a pipe under Actions, so Python would hold
    # these back in a block buffer and the live log would stay empty.
    print(line, flush=True)


async def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = resolve_settings(args, parser)
    provider_name = settings.werewolf_llm_provider
    print(f"==> LLM プロバイダ: {provider_name}", flush=True)
    print(f"==> モデル: {model_label(None, settings)}", flush=True)
    print(f"==> 推理エンジン: {settings.werewolf_reasoning_engine}", flush=True)
    if provider_name == "anthropic":
        print(f"==> 思考量(effort): {settings.anthropic_effort_by_kind}", flush=True)
    budget = build_budget(args)
    if budget is not None:
        print(f"==> 費用上限: ${args.max_cost:.2f}(対戦と判定で共有)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    runs: list[GameRun] = []
    budget_stop: str | None = None
    # Which part of the run the cap stopped; the report words the stop by it.
    budget_stop_phase: BudgetStopPhase = "game"
    for offset in range(args.games):
        seed = args.seed + offset
        print(f"==> game {offset + 1}/{args.games} (seed={seed})", flush=True)
        run = await play_game(
            seed, settings, MetricsCollector(), emit=_print_line, out_dir=args.out, budget=budget
        )
        runs.append(run)
        write_transcript_files(run.transcript, args.out, seed, partial=not run.finished)
        write_call_log(
            run.call_log, list(run.metrics.records), run.transcript.names, args.out, seed
        )
        winner = run.transcript.final_state.get("winner")
        print(f"{winner_line(winner)}{_budget_line(budget)}", flush=True)
        if run.budget_stop is not None:
            budget_stop = run.budget_stop
            print(f"==> {BUDGET_STOP_BANNER}: {budget_stop}", flush=True)
            break

    games: list[tuple[GameTranscript, AnalysisResult]] = [
        (run.transcript, analyze(run.transcript)) for run in runs
    ]
    for (_transcript, analysis), run in zip(games, runs, strict=True):
        print(f"    seed={run.seed} findings={len(analysis.findings)}", flush=True)

    judge_summary: dict[str, Any] | None = None
    judge_detail: dict[str, Any] = {}
    judge_metrics: dict[str, Any] | None = None
    if args.judge and runs and budget_stop is None:
        print("==> LLM判定を実行中", flush=True)
        # Judge calls get their own collector, so the game's cost and latency
        # are not polluted by evaluation overhead -- but they share the budget,
        # because the cap is on what the whole run spends.
        judge_collector = MetricsCollector()
        judge_provider: LLMProvider = build_llm_provider(
            judge_settings(args, settings), seed=args.seed, metrics=judge_collector
        )
        if budget is not None:
            judge_provider = BudgetedProvider(judge_provider, budget)
        try:
            result = await judge_transcript(runs[0].transcript, judge_provider)
        except BudgetExceeded as exc:
            budget_stop = str(exc)
            budget_stop_phase = "judge"
            print(f"==> {BUDGET_STOP_BANNER}(判定中): {budget_stop}", flush=True)
        else:
            judge_summary = result.summary
            judge_detail = result.per_player
        judge_metrics = metrics_summary(judge_collector, args.price_in, args.price_out)

    aggregate = MetricsCollector.merged(run.metrics for run in runs)
    overall = metrics_summary(aggregate, args.price_in, args.price_out)
    model = runs[0].model if runs else model_label(None, settings)
    per_game = [per_game_entry(run, args.price_in, args.price_out) for run in runs]
    commit = commit_sha()
    effort_by_kind = (
        settings.anthropic_effort_by_kind if provider_name == "anthropic" else None
    )
    snapshot = budget.snapshot() if budget is not None else None

    report = render_report(
        games=games,
        metrics_summary=overall,
        judge_summary=judge_summary,
        provider=games[0][0].provider if games else "unknown",
        model=model,
        engine=settings.werewolf_reasoning_engine,
        effort_by_kind=effort_by_kind,
        commit=commit,
        per_game=per_game,
        budget=snapshot,
        budget_stop=budget_stop,
        budget_stop_phase=budget_stop_phase,
    )
    (args.out / "report.md").write_text(report, encoding="utf-8")
    summary: dict[str, Any] = {
        "provider": provider_name,
        "model": model,
        "reasoning_engine": settings.werewolf_reasoning_engine,
        "commit": commit,
        "games": args.games,
        "games_played": len(runs),
        "stopped_by_budget": budget_stop is not None,
        "budget_stop": budget_stop,
        # "game": a game was cut short; "judge": every game finished and only
        # the LLM judge afterwards reached the cap.
        "budget_stop_phase": budget_stop_phase if budget_stop is not None else None,
        "budget": snapshot,
        "metrics": overall,
        "per_game": per_game,
        "judge": {"summary": judge_summary, "per_player": judge_detail, "metrics": judge_metrics},
        "findings": [
            {
                "seed": t.seed,
                "winner": t.final_state.get("winner"),
                "counts": {check: a.count(check) for check in {f.check for f in a.findings}},
                "stats": a.stats,
            }
            for t, a in games
        ],
    }
    if provider_name == "anthropic":
        summary["settings"] = {
            "anthropic_effort_by_kind": settings.anthropic_effort_by_kind,
            "anthropic_fallbacks": settings.werewolf_anthropic_fallbacks,
            "anthropic_max_tokens": settings.werewolf_anthropic_max_tokens,
        }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(report)
    print(f"==> 出力先: {args.out}/")
    return EXIT_BUDGET_STOPPED if budget_stop is not None else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
