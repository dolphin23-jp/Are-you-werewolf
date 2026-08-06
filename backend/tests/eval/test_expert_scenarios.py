from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.expert_scenarios import (
    BaselineAnswerProvider,
    ExpertScenarioAnswer,
    load_case,
    load_cases,
    render_model_prompt,
    score_answer,
    summarize_scores,
)


def _scenario() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "scenario_id": "fixture-cutoff",
        "log_id": "fixture",
        "cutoff_event_id": "fixture.d6.cutoff",
        "perspective": {"kind": "public", "actor_id": None, "notes": "public"},
        "public_facts": [
            {
                "fact_id": "f1",
                "statement": "Two wolves are publicly dead.",
                "source_event_ids": ["e1"],
                "confidence": "deterministic",
                "alternatives": [],
            },
            {
                "fact_id": "f2",
                "statement": "Candidate L is an LW candidate.",
                "source_event_ids": ["e2"],
                "confidence": "reviewed_extraction",
                "alternatives": [],
            },
        ],
        "private_facts": [
            {
                "fact_id": "secret",
                "statement": "This must not appear in a public prompt.",
                "source_event_ids": ["s1"],
                "confidence": "deterministic",
                "alternatives": [],
            }
        ],
        "possible_worlds": [
            {
                "world_id": "w-main",
                "summary": "L is LW and fox may live.",
                "required_assumptions": ["medium true"],
            },
            {
                "world_id": "w-alt",
                "summary": "A different claimant is true.",
                "required_assumptions": ["claimant B true"],
            },
        ],
        "impossible_worlds": [
            {
                "world_id": "w-impossible",
                "summary": "A result is both black and white.",
                "contradiction": "same target cannot be both",
                "unsat_core": ["e1"],
            }
        ],
        "conditional_deductions": [],
        "expert_assessment": {
            "main_world_ids": ["w-main"],
            "alternative_world_ids": ["w-alt"],
            "confidence": "high",
            "disagreement_expected": False,
            "reason": "gold explanation must not leak",
        },
        "weighting_factors": [],
        "recommended_action": {
            "action_type": "execute",
            "target_id": "safe-grey",
            "reason": "buy a night",
            "alternatives": [
                {
                    "action_type": "execute",
                    "target_id": "L",
                    "reason": "LWなら即狐勝利になる。",
                }
            ],
        },
        "common_mistakes": ["do not execute LW"],
        "counterexamples": [],
        "ai_correction": {
            "initial_assessment": "execute L",
            "expert_correction": "hold L",
            "error_categories": ["faction_objective"],
            "why_the_ai_failed": "ignored fox",
            "generalized_prevention_rule": "separate wolf belief from action",
            "rule_counterexamples": [],
        },
        "review": {
            "status": "expert_reviewed",
            "reviewer": "fixture",
            "reviewed_at": "2026-08-06T00:00:00+09:00",
            "training_ready": False,
        },
    }


def _write_scenario(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(_scenario(), ensure_ascii=False), encoding="utf-8")
    return path


def test_prompt_is_spoiler_safe_and_actions_are_opaque(tmp_path: Path) -> None:
    case = load_case(_write_scenario(tmp_path), seed=7)
    prompt = render_model_prompt(case)

    assert "This must not appear" not in prompt
    assert "gold explanation must not leak" not in prompt
    assert "buy a night" not in prompt
    assert "expert_correction" not in prompt
    assert case.gold_action_id in case.action_ids
    assert all(action.action_id.startswith("a") for action in case.actions)


def test_perfect_answer_scores_one(tmp_path: Path) -> None:
    case = load_case(_write_scenario(tmp_path), seed=1)
    answer = ExpertScenarioAnswer(
        possible_world_ids=sorted(case.gold_possible_world_ids),
        impossible_world_ids=sorted(case.gold_impossible_world_ids),
        main_world_ids=sorted(case.gold_main_world_ids),
        alternative_world_ids=sorted(case.gold_alternative_world_ids),
        recommended_action_id=case.gold_action_id,
        alternative_action_ids=sorted(case.action_ids - {case.gold_action_id}),
        catastrophic_action_ids=sorted(case.gold_catastrophic_action_ids),
        cited_fact_ids=sorted(case.fact_ids),
        next_observation="night result",
        confidence=case.gold_confidence,
        rationale="separate LW belief and today's execution",
    )

    score = score_answer(case, answer)

    assert score.overall_score == pytest.approx(1.0)
    assert score.action_exact == 1.0
    assert score.catastrophic_action_f1 == 1.0
    assert score.structured_reference_leakage_count == 0


def test_bad_answer_is_penalized_and_selects_catastrophic_action(tmp_path: Path) -> None:
    case = load_case(_write_scenario(tmp_path), seed=1)
    catastrophic = next(iter(case.gold_catastrophic_action_ids))
    answer = ExpertScenarioAnswer(
        possible_world_ids=[
            *case.gold_possible_world_ids,
            *case.gold_impossible_world_ids,
            "future",
        ],
        impossible_world_ids=[],
        main_world_ids=["future"],
        alternative_world_ids=[],
        recommended_action_id=catastrophic,
        alternative_action_ids=["unknown-action"],
        catastrophic_action_ids=[],
        cited_fact_ids=["future-fact"],
        next_observation="later role table",
        confidence="low",
        rationale="uses unavailable truth",
    )

    score = score_answer(case, answer)

    assert score.action_exact == 0.0
    assert score.catastrophic_action_avoidance == 0.0
    assert score.structured_reference_leakage_count == 3
    assert score.overall_score < 0.3


@pytest.mark.asyncio
async def test_baseline_provider_is_schema_valid(tmp_path: Path) -> None:
    case = load_case(_write_scenario(tmp_path), seed=2)
    answer = await BaselineAnswerProvider().answer(case)
    score = score_answer(case, answer)

    assert score.answer_valid
    assert answer.recommended_action_id in case.action_ids


def test_repository_reviewed_bundle_loads_at_least_eight_scenarios() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cases = load_cases(repo_root / "data" / "expert_scenarios" / "reviewed", seed=1)
    scenario_ids = {case.scenario_id for case in cases}

    assert len(cases) >= 8
    assert "ruru-349517-d7-lwco-controlled-night" in scenario_ids
    assert "ruru-352698-d6-lw-hold-cross-divination" in scenario_ids


def test_summary_aggregates_scores(tmp_path: Path) -> None:
    case = load_case(_write_scenario(tmp_path), seed=3)
    answer = ExpertScenarioAnswer(
        possible_world_ids=sorted(case.gold_possible_world_ids),
        impossible_world_ids=sorted(case.gold_impossible_world_ids),
        main_world_ids=sorted(case.gold_main_world_ids),
        alternative_world_ids=sorted(case.gold_alternative_world_ids),
        recommended_action_id=case.gold_action_id,
        alternative_action_ids=[],
        catastrophic_action_ids=sorted(case.gold_catastrophic_action_ids),
        cited_fact_ids=sorted(case.fact_ids),
        next_observation="result",
        confidence=case.gold_confidence,
        rationale="perfect",
    )
    summary = summarize_scores([score_answer(case, answer)])

    assert summary.scenario_count == 1
    assert summary.mean_overall_score == pytest.approx(1.0)
    assert summary.action_accuracy == 1.0
