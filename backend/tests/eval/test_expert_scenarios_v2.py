from __future__ import annotations

from pathlib import Path

import pytest

from app.eval.expert_scenarios_v2 import (
    ActionAssessment,
    BaselineV2AnswerProvider,
    ExpertScenarioV2Answer,
    PhaseChoice,
    WorldJudgment,
    load_v2_cases,
    render_v2_model_prompt,
    score_v2_answer,
    summarize_v2_scores,
)


def _paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root / "data" / "expert_scenarios" / "reviewed",
        repo_root / "data" / "expert_scenarios" / "v2_annotations.json",
    )


def _cases(seed: int = 1):
    root, annotations = _paths()
    return load_v2_cases(root, annotations_path=annotations, seed=seed)


def _perfect_answer(case) -> ExpertScenarioV2Answer:
    judgments = []
    for world in case.worlds:
        contradiction = case.gold_world_contradictions.get(world.world_id)
        judgments.append(
            WorldJudgment(
                world_id=world.world_id,
                status=(
                    "impossible"
                    if world.world_id in case.gold_impossible_world_ids
                    else "possible"
                ),
                contradiction_fact_ids=(
                    sorted(contradiction.fact_ids) if contradiction is not None else []
                ),
                contradiction_rule_ids=(
                    sorted(contradiction.rule_ids) if contradiction is not None else []
                ),
                rationale="gold-compatible fixture",
            )
        )
    assessments = [
        ActionAssessment(
            action_id=action.action_id,
            rating=case.gold_action_ratings[action.action_id],
            loss_condition=case.gold_action_loss_conditions[action.action_id],
            cited_fact_ids=[],
            rationale="gold-compatible fixture",
        )
        for action in case.actions
    ]
    choices = [
        PhaseChoice(
            phase=slot.phase,
            actor_id=slot.actor_id,
            selected_action_id=slot.action_id,
            rationale="gold-compatible fixture",
        )
        for slot in case.gold_plan
    ]
    return ExpertScenarioV2Answer(
        world_judgments=judgments,
        main_world_ids=sorted(case.gold_main_world_ids),
        alternative_world_ids=sorted(case.gold_alternative_world_ids),
        action_assessments=assessments,
        phase_choices=choices,
        next_observation="next public result",
        confidence=case.gold_confidence,
        rationale="perfect structured answer",
    )


def test_repository_bundle_loads_all_eight_v2_annotations() -> None:
    cases = _cases()
    assert len(cases) == 8
    assert {case.scenario_id for case in cases} >= {
        "ruru-349517-d7-lwco-controlled-night",
        "ruru-352698-d6-lw-hold-cross-divination",
    }


def test_v2_prompt_is_spoiler_safe_and_phase_aware() -> None:
    case = next(
        item
        for item in _cases()
        if item.scenario_id == "ruru-352698-d6-lw-hold-cross-divination"
    )
    prompt = render_v2_model_prompt(case)

    assert '"task_version": "expert-scenario-phase-plan-v2"' in prompt
    assert "gold_action_ratings" not in prompt
    assert "loss_condition" not in prompt
    assert "涙の数だけ" in prompt
    assert "night_divination" in prompt
    assert {slot.phase for slot in case.gold_plan} == {"day", "night_divination"}
    assert len(case.gold_plan) == 3


def test_perfect_v2_answer_scores_one() -> None:
    case = _cases()[0]
    score = score_v2_answer(case, _perfect_answer(case))

    assert score.overall_score == pytest.approx(1.0)
    assert score.world_status_accuracy == 1.0
    assert score.action_rating_accuracy == 1.0
    assert score.phase_choice_exact == 1.0
    assert score.consistency_violation_count == 0


def test_missing_hard_impossibility_is_strongly_penalized() -> None:
    case = next(
        item for item in _cases() if item.scenario_id == "ruru-349517-d3-corona-death"
    )
    answer = _perfect_answer(case)
    impossible_id = next(iter(case.gold_impossible_world_ids))
    answer.world_judgments = [
        item.model_copy(
            update={
                "status": "possible",
                "contradiction_fact_ids": [],
                "contradiction_rule_ids": [],
            }
        )
        if item.world_id == impossible_id
        else item
        for item in answer.world_judgments
    ]

    score = score_v2_answer(case, answer)

    assert score.impossible_world_recall == 0.0
    assert score.world_status_accuracy < 1.0
    assert score.overall_score < 0.9


def test_self_contradictory_plan_is_detected() -> None:
    case = next(
        item
        for item in _cases()
        if item.scenario_id == "ruru-352698-d6-lw-hold-cross-divination"
    )
    answer = _perfect_answer(case)
    day_slot = next(slot for slot in case.gold_plan if slot.phase == "day")
    answer.action_assessments = [
        item.model_copy(update={"rating": "catastrophic", "loss_condition": "forced loss"})
        if item.action_id == day_slot.action_id
        else item
        for item in answer.action_assessments
    ]

    score = score_v2_answer(case, answer)

    assert score.consistency_violation_count >= 1
    assert score.consistency_score < 1.0
    assert score.action_rating_accuracy < 1.0


@pytest.mark.asyncio
async def test_v2_baseline_is_schema_valid() -> None:
    case = _cases()[0]
    answer = await BaselineV2AnswerProvider().answer(case)
    score = score_v2_answer(case, answer)

    assert score.answer_valid
    assert score.world_coverage == 1.0
    assert score.action_coverage == 1.0


def test_v2_summary_aggregates() -> None:
    case = _cases()[0]
    summary = summarize_v2_scores([score_v2_answer(case, _perfect_answer(case))])

    assert summary.scenario_count == 1
    assert summary.mean_overall_score == pytest.approx(1.0)
    assert summary.mean_phase_choice_exact == 1.0
