#!/usr/bin/env python3
"""Evaluate models with the phase-aware expert cutoff benchmark v2.

    python scripts/evaluate_expert_scenarios_v2.py --provider baseline
    python scripts/evaluate_expert_scenarios_v2.py --provider anthropic --max-cost 5

Providers, the spending cap and its exit code 3 work exactly as in
`scripts/evaluate_expert_scenarios.py`, whose helpers this reuses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.metrics import MetricsCollector  # noqa: E402
from app.ai.provider.budget import EvaluationBudget  # noqa: E402
from app.eval.expert_scenarios_v2 import (  # noqa: E402
    BaselineV2AnswerProvider,
    ExpertScenarioV2Answer,
    ExpertScenarioV2Case,
    LLMV2AnswerProvider,
    load_external_v2_answer,
    load_v2_cases,
    render_v2_model_prompt,
    render_v2_report,
    score_v2_answer,
    summarize_v2_scores,
)
from app.eval.report import BUDGET_STOP_BANNER  # noqa: E402
from scripts.evaluate import EXIT_BUDGET_STOPPED, build_budget  # noqa: E402
from scripts.evaluate_expert_scenarios import (  # noqa: E402
    PROVIDERS,
    add_cost_arguments,
    answer_all,
    budget_banner,
    build_answer_llm,
    check_cost_arguments,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


async def _generate_answers(
    cases: list[ExpertScenarioV2Case],
    *,
    provider_name: str,
    metrics: MetricsCollector,
    concurrency: int,
    budget: EvaluationBudget | None = None,
) -> tuple[dict[str, ExpertScenarioV2Answer | None], str, str | None]:
    """`(answers, model_label, budget_stop)`."""
    if provider_name == "baseline":
        baseline = BaselineV2AnswerProvider()
        answers, stop = await answer_all(
            cases, baseline.answer, lambda case: case.scenario_id, concurrency
        )
        return answers, "baseline-v2", stop
    llm, label = build_answer_llm(provider_name, metrics, budget)
    provider = LLMV2AnswerProvider(llm)
    answers, stop = await answer_all(
        cases, provider.answer, lambda case: case.scenario_id, concurrency
    )
    return answers, label, stop


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", choices=PROVIDERS, default="baseline")
    parser.add_argument("--answers-dir", type=Path, default=None)
    parser.add_argument(
        "--scenarios-root",
        type=Path,
        default=_repo_root() / "data" / "expert_scenarios" / "reviewed",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=_repo_root() / "data" / "expert_scenarios" / "v2_annotations.json",
    )
    parser.add_argument("--scenario-id", action="append", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("expert-eval-v2-out"))
    add_cost_arguments(parser)
    parser.add_argument("--fail-under", type=float, default=None)
    args = parser.parse_args(argv)
    check_cost_arguments(args, parser)

    scenario_ids = set(args.scenario_id) if args.scenario_id else None
    cases = load_v2_cases(
        args.scenarios_root,
        annotations_path=args.annotations,
        seed=args.seed,
        scenario_ids=scenario_ids,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    prompts_dir = args.out / "prompts"
    answers_out_dir = args.out / "answers"
    scores_dir = args.out / "scores"
    prompts_dir.mkdir(exist_ok=True)
    answers_out_dir.mkdir(exist_ok=True)
    scores_dir.mkdir(exist_ok=True)

    for case in cases:
        (prompts_dir / f"{case.scenario_id}.json").write_text(
            render_v2_model_prompt(case), encoding="utf-8"
        )

    metrics = MetricsCollector()
    budget = build_budget(args) if args.answers_dir is None else None
    budget_stop: str | None = None
    if args.answers_dir is not None:
        answers: dict[str, ExpertScenarioV2Answer | None] = {}
        for case in cases:
            path = args.answers_dir / f"{case.scenario_id}.json"
            answers[case.scenario_id] = (
                load_external_v2_answer(path) if path.exists() else None
            )
        provider_label = "external-answers"
        model = "external-v2"
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
        score = score_v2_answer(case, answer)
        scores.append(score)
        (scores_dir / f"{case.scenario_id}.json").write_text(
            json.dumps(score.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{case.scenario_id}: overall={score.overall_score:.3f} "
            f"plan={score.phase_choice_exact:.2f} valid={score.answer_valid}",
            flush=True,
        )

    summary = summarize_v2_scores(scores)
    metrics_summary: dict[str, Any] | None = None
    if args.answers_dir is None and args.provider != "baseline":
        metrics_summary = metrics.summary(args.price_in, args.price_out)

    report = render_v2_report(
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
                "benchmark_version": "v2",
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
