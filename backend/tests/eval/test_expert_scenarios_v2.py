from __future__ import annotations

from pathlib import Path

import pytest

from app.eval.expert_scenarios import load_cases, render_model_prompt
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


def test_no_prompt_carries_the_labels_or_ids_it_is_scored_on() -> None:
    """Only possible worlds had `required_assumptions`, so their presence alone
    classified every world; the scenario, log and cutoff ids name the gold plan
    and the real game. None of it may reach the model."""
    root, _ = _paths()
    v1_cases = load_cases(root, seed=1)
    v2_cases = _cases()
    assert v1_cases and v2_cases
    for case, prompt in [(c, render_model_prompt(c)) for c in v1_cases] + [
        (c, render_v2_model_prompt(c)) for c in v2_cases
    ]:
        assert "required_assumptions" not in prompt
        for identifier in (case.scenario_id, case.log_id, case.cutoff_event_id):
            assert identifier not in prompt, identifier


def test_every_perfect_answer_scores_one_without_violations() -> None:
    for case in _cases():
        score = score_v2_answer(case, _perfect_answer(case))
        assert score.overall_score == pytest.approx(1.0), case.scenario_id
        assert score.consistency_violation_count == 0, case.scenario_id


def _board_case():
    return next(
        item for item in _cases() if item.scenario_id == "ruru-349517-d2-2-0-public-board"
    )


def test_citing_a_fact_for_a_rule_only_contradiction_is_not_penalized() -> None:
    """The gold for this world names only a rule; the prompt asks to cite both
    kinds, and citing one fact used to drop the fact F1 from 1.0 to 0.0."""
    case = _board_case()
    answer = _perfect_answer(case)
    some_fact = sorted(case.fact_ids)[0]
    answer.world_judgments = [
        item.model_copy(update={"contradiction_fact_ids": [some_fact]})
        if item.world_id == "w-reino-and-seer-both-first"
        else item
        for item in answer.world_judgments
    ]

    score = score_v2_answer(case, answer)

    assert score.contradiction_fact_f1 == 1.0
    assert score.consistency_violation_count == 0


def test_consistency_follows_the_answers_own_status_not_the_gold() -> None:
    case = _board_case()
    possible_id = next(
        world.world_id
        for world in case.worlds
        if world.world_id not in case.gold_impossible_world_ids
    )
    some_fact = sorted(case.fact_ids)[0]

    def judged_impossible(citations: list[str]) -> int:
        answer = _perfect_answer(case)
        answer.world_judgments = [
            item.model_copy(update={"status": "impossible", "contradiction_fact_ids": citations})
            if item.world_id == possible_id
            else item
            for item in answer.world_judgments
        ]
        # Keep the answer's own weighting consistent with its own verdict.
        answer.main_world_ids = [w for w in answer.main_world_ids if w != possible_id]
        answer.alternative_world_ids = [
            w for w in answer.alternative_world_ids if w != possible_id
        ]
        return score_v2_answer(case, answer).consistency_violation_count

    # Wrong but supported: scored by status accuracy, not as inconsistency.
    assert judged_impossible([some_fact]) == 0
    # Unsupported impossibility is the inconsistent one.
    assert judged_impossible([]) == 1


def test_one_wrong_slot_choice_is_one_violation() -> None:
    case = next(
        item
        for item in _cases()
        if item.scenario_id == "ruru-352698-d6-lw-hold-cross-divination"
    )
    answer = _perfect_answer(case)
    day_slot = next(slot for slot in case.gold_plan if slot.phase == "day")
    night_action = next(slot.action_id for slot in case.gold_plan if slot.phase != "day")
    answer.phase_choices = [
        item.model_copy(update={"selected_action_id": night_action})
        if (item.phase, item.actor_id) == (day_slot.phase, day_slot.actor_id)
        else item
        for item in answer.phase_choices
    ]

    assert score_v2_answer(case, answer).consistency_violation_count == 1


def test_omissions_are_not_scored_as_success() -> None:
    case = _cases()[0]
    empty = ExpertScenarioV2Answer(
        world_judgments=[],
        main_world_ids=[],
        alternative_world_ids=[],
        action_assessments=[],
        phase_choices=[],
        next_observation="",
        confidence="low",
        rationale="",
    )
    score = score_v2_answer(case, empty)
    assert not score.answer_valid
    assert score.catastrophic_action_avoidance == 0.0
    assert score.consistency_score == 0.0

    partial = _perfect_answer(case)
    partial.phase_choices = partial.phase_choices[1:]
    score = score_v2_answer(case, partial)
    assert score.answer_valid
    assert score.catastrophic_action_avoidance < 1.0
    assert score.consistency_violation_count == 1
