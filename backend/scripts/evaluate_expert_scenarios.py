#!/usr/bin/env python3
"""Evaluate models on the reviewed real-game cutoff scenarios.

Examples:

    # Zero-cost harness smoke test
    python scripts/evaluate_expert_scenarios.py --provider baseline

    # The real provider (Claude). --max-cost (USD) is required.
    python scripts/evaluate_expert_scenarios.py --provider anthropic --max-cost 5

    # The dormant OpenAI-compatible model (luna)
    python scripts/evaluate_expert_scenarios.py --provider luna

    # Score answers produced by any external model
    python scripts/evaluate_expert_scenarios.py --answers-dir external-answers

External answer files are named ``<scenario_id>.json`` and must validate as
``ExpertScenarioAnswer``. Prompts are always written to ``<out>/prompts`` so
another model can be evaluated without adding a provider adapter first.

With `--max-cost`, every answer call shares one budget. When the cap is
reached the remaining scenarios are left unanswered (scored as invalid), the
report says so first (`費用上限に達したため中断`), and the exit code is 3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.metrics import MetricsCollector  # noqa: E402
from app.ai.provider.base import LLMProvider  # noqa: E402
from app.ai.provider.budget import (  # noqa: E402
    BudgetedProvider,
    BudgetExceeded,
    EvaluationBudget,
)
from app.ai.provider.factory import build_llm_provider  # noqa: E402
from app.ai.provider.json_instruction import JsonInstructionProvider  # noqa: E402
from app.config import Settings  # noqa: E402
from app.eval.expert_scenarios import (  # noqa: E402
    BaselineAnswerProvider,
    ExpertScenarioAnswer,
    ExpertScenarioCase,
    LLMAnswerProvider,
    load_cases,
    load_external_answer,
    render_model_prompt,
    render_report,
    score_answer,
    summarize_scores,
)
from app.eval.report import BUDGET_STOP_BANNER  # noqa: E402
from scripts.evaluate import EXIT_BUDGET_STOPPED, build_budget, model_label  # noqa: E402

PROVIDERS = ("baseline", "luna", "anthropic")
_LLM_PROVIDERS = frozenset({"luna", "anthropic"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_cost_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--price-in", type=float, default=0.0)
    parser.add_argument("--price-out", type=float, default=0.0)
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="推定費用の上限(USD)。全回答で共有し、超えそうなら送らずに止めて終了コード3。"
        "anthropic では必須",
    )


def check_cost_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Refuse a run whose spending would be unbounded or whose cap could not fire."""
    if args.answers_dir is not None:
        return  # scoring existing answers calls no API
    if args.max_cost is not None and not args.max_cost > 0:
        parser.error("--max-cost は0より大きいUSDの金額にしてください。")
    if args.provider == "anthropic" and args.max_cost is None:
        parser.error("anthropic で評価するときは --max-cost(USD)が必須です。例: --max-cost 5")
    if (
        args.provider == "luna"
        and args.max_cost is not None
        and not (args.price_in > 0 or args.price_out > 0)
    ):
        parser.error(
            "luna で --max-cost を効かせるには --price-in / --price-out が必要です"
            "(単価が無いと費用を計算できず、上限が効きません)。"
        )


def build_answer_llm(
    provider_name: str, metrics: MetricsCollector, budget: EvaluationBudget | None
) -> tuple[LLMProvider, str]:
    """The game's own provider for `provider_name`, budgeted, and its model label.

    The schema goes into the prompt as text for every provider: luna needs it
    to know the field names in JSON-object mode, and giving Claude the same
    prompt (its grammar enforces the schema anyway) keeps the two comparable.
    """
    settings = Settings(werewolf_llm_provider=provider_name)
    inner = build_llm_provider(settings, metrics=metrics)
    label = model_label(inner, settings)
    llm: LLMProvider = BudgetedProvider(inner, budget) if budget is not None else inner
    return JsonInstructionProvider(llm), label


async def answer_all[CaseT, AnswerT](
    cases: Sequence[CaseT],
    answer: Callable[[CaseT], Awaitable[AnswerT | None]],
    scenario_id: Callable[[CaseT], str],
    concurrency: int,
) -> tuple[dict[str, AnswerT | None], str | None]:
    """Every case's answer, and the `BudgetExceeded` message if the cap ended it.

    A capped call leaves that case unanswered instead of failing the run: the
    answers already paid for are scored and reported, the rest are refused
    by the budget before anything is sent.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))
    stops: list[str] = []

    async def run_one(case: CaseT) -> tuple[str, AnswerT | None]:
        async with semaphore:
            try:
                return scenario_id(case), await answer(case)
            except BudgetExceeded as exc:
                stops.append(str(exc))
                return scenario_id(case), None

    pairs = await asyncio.gather(*(run_one(case) for case in cases))
    return dict(pairs), (stops[0] if stops else None)


def budget_banner(stop: str) -> str:
    return (
        f"> **{BUDGET_STOP_BANNER}。** 上限に届く呼び出しは送らずに止めました。"
        "回答の無いシナリオは無効として採点されています。\n>\n"
        f"> `{stop}`\n\n"
    )


async def _generate_answers(
    cases: list[ExpertScenarioCase],
    *,
    provider_name: str,
    metrics: MetricsCollector,
    concurrency: int,
    budget: EvaluationBudget | None = None,
) -> tuple[dict[str, ExpertScenarioAnswer | None], str, str | None]:
    """`(answers, model_label, budget_stop)`."""
    if provider_name == "baseline":
        baseline = BaselineAnswerProvider()
        answers, stop = await answer_all(
            cases, baseline.answer, lambda case: case.scenario_id, concurrency
        )
        return answers, "baseline-v1", stop
    llm, label = build_answer_llm(provider_name, metrics, budget)
    provider = LLMAnswerProvider(llm)
    answers, stop = await answer_all(
        cases, provider.answer, lambda case: case.scenario_id, concurrency
    )
    return answers, label, stop


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", choices=PROVIDERS, default="baseline")
    parser.add_argument(
        "--answers-dir",
        type=Path,
        default=None,
        help="既存回答を採点する。指定時はAPIを呼ばず --provider を無視する。",
    )
    parser.add_argument(
        "--scenarios-root",
        type=Path,
        default=_repo_root() / "data" / "expert_scenarios" / "reviewed",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=None,
        help="対象scenario_id。複数回指定可。省略時は全reviewed JSON。",
    )
    parser.add_argument("--seed", type=int, default=1, help="候補順の再現用シード")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("expert-eval-out"))
    add_cost_arguments(parser)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="平均overallがこの値未満なら終了コード1。比較ゲート用。",
    )
    args = parser.parse_args(argv)
    check_cost_arguments(args, parser)

    scenario_ids = set(args.scenario_id) if args.scenario_id else None
    cases = load_cases(args.scenarios_root, seed=args.seed, scenario_ids=scenario_ids)
    args.out.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.out / "prompts"
    answers_out_dir = args.out / "answers"
    scores_dir = args.out / "scores"
    prompts_dir.mkdir(exist_ok=True)
    answers_out_dir.mkdir(exist_ok=True)
    scores_dir.mkdir(exist_ok=True)

    for case in cases:
        (prompts_dir / f"{case.scenario_id}.json").write_text(
            render_model_prompt(case), encoding="utf-8"
        )

    metrics = MetricsCollector()
    budget = build_budget(args) if args.answers_dir is None else None
    budget_stop: str | None = None
    if args.answers_dir is not None:
        answers: dict[str, ExpertScenarioAnswer | None] = {}
        for case in cases:
            path = args.answers_dir / f"{case.scenario_id}.json"
            answers[case.scenario_id] = load_external_answer(path) if path.exists() else None
        provider_label = "external-answers"
        model = "external"
    else:
        answers, model, budget_stop = await _generate_answers(
            cases,
            provider_name=args.provider,
            metrics=metrics,
            concurrency=args.concurrency,
            budget=budget,
        )
        provider_label = args.provider

    scores = []
    for case in cases:
        answer = answers[case.scenario_id]
        if answer is not None:
            (answers_out_dir / f"{case.scenario_id}.json").write_text(
                answer.model_dump_json(indent=2), encoding="utf-8"
            )
        score = score_answer(case, answer)
        scores.append(score)
        (scores_dir / f"{case.scenario_id}.json").write_text(
            json.dumps(score.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{case.scenario_id}: overall={score.overall_score:.3f} "
            f"action={score.action_exact:.0f} valid={score.answer_valid}",
            flush=True,
        )

    summary = summarize_scores(scores)
    metrics_summary: dict[str, Any] | None = None
    if args.answers_dir is None and args.provider in _LLM_PROVIDERS:
        metrics_summary = metrics.summary(args.price_in, args.price_out)

    report = render_report(
        provider=provider_label,
        model=model,
        scores=scores,
        summary=summary,
        metrics_summary=metrics_summary,
    )
    if budget_stop is not None:
        report = budget_banner(budget_stop) + report
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "provider": provider_label,
                "model": model,
                "seed": args.seed,
                "summary": summary.to_dict(),
                "metrics": metrics_summary,
                "budget": budget.snapshot() if budget is not None else None,
                "stopped_by_budget": budget_stop is not None,
                "budget_stop": budget_stop,
                "scores": [score.to_dict() for score in scores],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(report)
    print(f"==> 出力先: {args.out}/")

    if budget_stop is not None:
        print(f"STOPPED: {BUDGET_STOP_BANNER}: {budget_stop}", file=sys.stderr)
        return EXIT_BUDGET_STOPPED

    if summary.valid_answer_rate == 0.0:
        print(
            "FAIL: valid model answers were not obtained; this run did not measure reasoning.",
            file=sys.stderr,
        )
        if metrics_summary is not None:
            for item in metrics_summary.get("errors", []):
                print(
                    f"  {item.get('count', 0)}x {item.get('error', 'unknown error')}",
                    file=sys.stderr,
                )
        return 2

    if args.fail_under is not None and summary.mean_overall_score < args.fail_under:
        print(
            f"FAIL: overall {summary.mean_overall_score:.3f} < {args.fail_under:.3f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
