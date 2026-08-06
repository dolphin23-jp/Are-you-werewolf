#!/usr/bin/env python3
"""Evaluate models with the phase-aware expert cutoff benchmark v2."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.metrics import MetricsCollector  # noqa: E402
from app.ai.provider.factory import build_llm_provider  # noqa: E402
from app.ai.provider.json_instruction import JsonInstructionProvider  # noqa: E402
from app.config import Settings  # noqa: E402
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


async def _generate_answers(
    cases: list[ExpertScenarioV2Case],
    *,
    provider_name: str,
    metrics: MetricsCollector,
    concurrency: int,
) -> dict[str, ExpertScenarioV2Answer | None]:
    if provider_name == "baseline":
        provider = BaselineV2AnswerProvider()
    else:
        settings = Settings(werewolf_llm_provider="luna")
        llm_provider = JsonInstructionProvider(build_llm_provider(settings, metrics=metrics))
        provider = LLMV2AnswerProvider(llm_provider)

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(
        case: ExpertScenarioV2Case,
    ) -> tuple[str, ExpertScenarioV2Answer | None]:
        async with semaphore:
            return case.scenario_id, await provider.answer(case)

    pairs = await asyncio.gather(*(run_one(case) for case in cases))
    return dict(pairs)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["baseline", "luna"], default="baseline")
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
    parser.add_argument("--price-in", type=float, default=0.0)
    parser.add_argument("--price-out", type=float, default=0.0)
    parser.add_argument("--fail-under", type=float, default=None)
    args = parser.parse_args()

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
    if args.answers_dir is not None:
        answers: dict[str, ExpertScenarioV2Answer | None] = {}
        for case in cases:
            path = args.answers_dir / f"{case.scenario_id}.json"
            answers[case.scenario_id] = (
                load_external_v2_answer(path) if path.exists() else None
            )
        provider_label = "external-answers"
        model_label = "external-v2"
    else:
        answers = await _generate_answers(
            cases,
            provider_name=args.provider,
            metrics=metrics,
            concurrency=args.concurrency,
        )
        provider_label = args.provider
        settings = Settings()
        model_label = settings.luna_model if args.provider == "luna" else "baseline-v2"

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
    if args.answers_dir is None and args.provider == "luna":
        metrics_summary = metrics.summary(args.price_in, args.price_out)

    report = render_v2_report(
        provider=provider_label,
        model=model_label,
        scores=scores,
        summary=summary,
        metrics_summary=metrics_summary,
    )
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "benchmark_version": "v2",
                "provider": provider_label,
                "model": model_label,
                "seed": args.seed,
                "summary": summary.to_dict(),
                "metrics": metrics_summary,
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
