"""Phase-aware v2 benchmark for expert-reviewed werewolf cutoffs.

v1 remains the stable single-action closed-set benchmark. v2 adds:

- explicit basic-rule candidates and per-world contradiction evidence,
- phase/actor-scoped actions instead of one mixed action list,
- optimal/acceptable/dominated/catastrophic action ratings,
- multi-step day/night plans,
- deterministic internal-consistency checks.

The prompt still excludes expert weighting, gold ratings, gold plans, later truth,
review metadata, and post-game corrections.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.ai.provider.base import LLMProvider, Message

Confidence = Literal["low", "medium", "high"]
WorldStatus = Literal["possible", "impossible"]
ActionRating = Literal["optimal", "acceptable", "dominated", "catastrophic"]
Phase = Literal["day", "night_divination", "night_guard", "night_attack"]


class WorldJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str
    status: WorldStatus
    contradiction_fact_ids: list[str] = Field(default_factory=list)
    contradiction_rule_ids: list[str] = Field(default_factory=list)
    rationale: str


class ActionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    rating: ActionRating
    loss_condition: str | None = None
    cited_fact_ids: list[str] = Field(default_factory=list)
    rationale: str


class PhaseChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: Phase
    actor_id: str | None = None
    selected_action_id: str
    rationale: str


class ExpertScenarioV2Answer(BaseModel):
    """Structured v2 answer with world, action, and phase-plan judgments."""

    model_config = ConfigDict(extra="forbid")

    world_judgments: list[WorldJudgment]
    main_world_ids: list[str] = Field(default_factory=list)
    alternative_world_ids: list[str] = Field(default_factory=list)
    action_assessments: list[ActionAssessment]
    phase_choices: list[PhaseChoice]
    next_observation: str
    confidence: Confidence
    rationale: str


class V2AnswerProvider(Protocol):
    async def answer(self, case: ExpertScenarioV2Case) -> ExpertScenarioV2Answer | None: ...


@dataclass(frozen=True)
class RuleCandidate:
    rule_id: str
    statement: str


@dataclass(frozen=True)
class WorldCandidate:
    world_id: str
    summary: str
    required_assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    phase: Phase
    actor_id: str | None
    action_type: str
    target_id: str | None


@dataclass(frozen=True)
class PlanSlot:
    phase: Phase
    actor_id: str | None
    action_id: str


@dataclass(frozen=True)
class ContradictionGold:
    fact_ids: frozenset[str]
    rule_ids: frozenset[str]


@dataclass(frozen=True)
class ExpertScenarioV2Case:
    scenario_id: str
    log_id: str
    cutoff_event_id: str
    perspective: dict[str, Any]
    facts: tuple[dict[str, Any], ...]
    rules: tuple[RuleCandidate, ...]
    worlds: tuple[WorldCandidate, ...]
    actions: tuple[ActionCandidate, ...]
    gold_possible_world_ids: frozenset[str]
    gold_impossible_world_ids: frozenset[str]
    gold_world_contradictions: dict[str, ContradictionGold]
    gold_main_world_ids: frozenset[str]
    gold_alternative_world_ids: frozenset[str]
    gold_action_ratings: dict[str, ActionRating]
    gold_action_loss_conditions: dict[str, str | None]
    gold_plan: tuple[PlanSlot, ...]
    gold_confidence: Confidence
    source_path: str

    @property
    def world_ids(self) -> frozenset[str]:
        return frozenset(item.world_id for item in self.worlds)

    @property
    def action_ids(self) -> frozenset[str]:
        return frozenset(item.action_id for item in self.actions)

    @property
    def fact_ids(self) -> frozenset[str]:
        return frozenset(str(item["fact_id"]) for item in self.facts)

    @property
    def rule_ids(self) -> frozenset[str]:
        return frozenset(item.rule_id for item in self.rules)

    def action_by_id(self) -> dict[str, ActionCandidate]:
        return {item.action_id: item for item in self.actions}

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "task_version": "expert-scenario-phase-plan-v2",
            "scenario_id": self.scenario_id,
            "log_id": self.log_id,
            "cutoff_event_id": self.cutoff_event_id,
            "perspective": self.perspective,
            "observed_facts": list(self.facts),
            "basic_rules": [asdict(item) for item in self.rules],
            "world_candidates": [
                {
                    "world_id": item.world_id,
                    "summary": item.summary,
                    "required_assumptions": list(item.required_assumptions),
                }
                for item in self.worlds
            ],
            "action_candidates": [asdict(item) for item in self.actions],
            "instructions": {
                "worlds": (
                    "Judge every world exactly once. Mark impossible only when supplied facts "
                    "and basic rules create a contradiction; cite both IDs."
                ),
                "weighting": (
                    "Choose main and alternative worlds only from worlds you judged possible. "
                    "Additional assumptions and tactical cost should lower, not erase, a world."
                ),
                "actions": (
                    "Assess every action exactly once as optimal, acceptable, dominated, or "
                    "catastrophic. Catastrophic means an immediate or forced faction loss, not "
                    "merely a weak or information-losing move."
                ),
                "plan": (
                    "Choose one action for every distinct (phase, actor_id) slot represented by "
                    "the candidates. Day and night actions are complementary, not mutually "
                    "exclusive."
                ),
                "consistency": (
                    "Do not select an action that your own assessment calls dominated or "
                    "catastrophic. Keep actor, phase, target, and loss condition consistent."
                ),
                "perspective": "Use only this payload and never later truth.",
            },
        }


@dataclass(frozen=True)
class ScenarioV2Score:
    scenario_id: str
    answer_valid: bool
    world_status_accuracy: float
    impossible_world_recall: float
    contradiction_fact_f1: float
    contradiction_rule_f1: float
    world_coverage: float
    main_world_jaccard: float
    alternative_world_jaccard: float
    action_rating_accuracy: float
    action_rating_ordinal_score: float
    action_coverage: float
    phase_choice_exact: float
    phase_choice_utility: float
    catastrophic_action_f1: float
    catastrophic_action_avoidance: float
    consistency_score: float
    consistency_violation_count: int
    confidence_score: float
    structured_reference_leakage_count: int
    unknown_world_ids: tuple[str, ...]
    unknown_action_ids: tuple[str, ...]
    unknown_fact_ids: tuple[str, ...]
    unknown_rule_ids: tuple[str, ...]
    missing_world_ids: tuple[str, ...]
    missing_action_ids: tuple[str, ...]
    missing_plan_slots: tuple[str, ...]
    overall_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkV2Summary:
    scenario_count: int
    valid_answer_rate: float
    mean_overall_score: float
    mean_hard_logic_score: float
    mean_weighting_score: float
    mean_action_rating_accuracy: float
    mean_phase_choice_exact: float
    mean_phase_choice_utility: float
    catastrophic_action_avoidance: float
    mean_consistency_score: float
    total_structured_reference_leakage_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SYSTEM_PROMPT = """You are evaluating one 17A werewolf cutoff with a phase-aware plan.
Use only the supplied facts and rules. Separate hard contradiction, soft belief,
action quality, and temporal planning. A bad move is not automatically
catastrophic. Day execution and night abilities can both belong to the same
optimal plan. Return only the requested structured JSON object."""

_RATING_ORDER: dict[ActionRating, int] = {
    "catastrophic": 0,
    "dominated": 1,
    "acceptable": 2,
    "optimal": 3,
}
_RATING_UTILITY: dict[ActionRating, float] = {
    "catastrophic": 0.0,
    "dominated": 0.2,
    "acceptable": 0.6,
    "optimal": 1.0,
}


def _stable_rng(seed: int, scenario_id: str) -> random.Random:
    digest = hashlib.sha256(f"v2:{seed}:{scenario_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _require_list(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def _signature(raw: dict[str, Any]) -> tuple[str, str | None, str, str | None]:
    return (
        str(raw["phase"]),
        raw.get("actor_id"),
        str(raw["action_type"]),
        raw.get("target_id"),
    )


def _source_signature(raw: dict[str, Any]) -> tuple[str, str | None]:
    return str(raw["action_type"]), raw.get("target_id")


def discover_scenario_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def load_v2_annotations(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "2.0":
        raise ValueError(f"invalid v2 annotation bundle: {path}")
    _require_list(raw, "rules")
    _require_mapping(raw, "scenarios")
    return raw


def load_v2_case(
    path: Path,
    *,
    annotations: dict[str, Any],
    seed: int = 0,
) -> ExpertScenarioV2Case:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"scenario must be an object: {path}")
    scenario_id = str(raw["scenario_id"])
    scenario_annotations = _require_mapping(annotations, "scenarios").get(scenario_id)
    if not isinstance(scenario_annotations, dict):
        raise ValueError(f"v2 annotations missing scenario {scenario_id}")

    perspective = _require_mapping(raw, "perspective")
    facts: list[dict[str, Any]] = []
    for item in _require_list(raw, "public_facts"):
        if not isinstance(item, dict):
            raise ValueError(f"public fact must be an object: {path}")
        facts.append(
            {
                "fact_id": str(item["fact_id"]),
                "visibility": "public",
                "statement": str(item["statement"]),
            }
        )
    if perspective.get("kind") != "public":
        for item in _require_list(raw, "private_facts"):
            if not isinstance(item, dict):
                raise ValueError(f"private fact must be an object: {path}")
            facts.append(
                {
                    "fact_id": str(item["fact_id"]),
                    "visibility": "private",
                    "statement": str(item["statement"]),
                }
            )
    fact_ids = {str(item["fact_id"]) for item in facts}

    rules = tuple(
        RuleCandidate(rule_id=str(item["rule_id"]), statement=str(item["statement"]))
        for item in _require_list(annotations, "rules")
        if isinstance(item, dict)
    )
    rule_ids = {item.rule_id for item in rules}

    worlds: list[WorldCandidate] = []
    possible_ids: set[str] = set()
    impossible_ids: set[str] = set()
    for item in _require_list(raw, "possible_worlds"):
        if not isinstance(item, dict):
            raise ValueError(f"possible world must be an object: {path}")
        world_id = str(item["world_id"])
        possible_ids.add(world_id)
        worlds.append(
            WorldCandidate(
                world_id=world_id,
                summary=str(item["summary"]),
                required_assumptions=tuple(
                    str(value) for value in item.get("required_assumptions", [])
                ),
            )
        )
    for item in _require_list(raw, "impossible_worlds"):
        if not isinstance(item, dict):
            raise ValueError(f"impossible world must be an object: {path}")
        world_id = str(item["world_id"])
        impossible_ids.add(world_id)
        worlds.append(
            WorldCandidate(world_id=world_id, summary=str(item["summary"]), required_assumptions=())
        )

    contradiction_raw = _require_mapping(scenario_annotations, "world_contradictions")
    if set(contradiction_raw) != impossible_ids:
        raise ValueError(
            f"v2 contradiction worlds do not match source impossible worlds for {scenario_id}"
        )
    contradictions: dict[str, ContradictionGold] = {}
    for world_id, item in contradiction_raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"contradiction annotation must be an object: {world_id}")
        gold_facts = frozenset(str(value) for value in item.get("fact_ids", []))
        gold_rules = frozenset(str(value) for value in item.get("rule_ids", []))
        if not gold_facts <= fact_ids:
            raise ValueError(
                f"unknown contradiction fact IDs for {scenario_id}: {gold_facts - fact_ids}"
            )
        if not gold_rules <= rule_ids:
            raise ValueError(
                f"unknown contradiction rule IDs for {scenario_id}: {gold_rules - rule_ids}"
            )
        contradictions[str(world_id)] = ContradictionGold(gold_facts, gold_rules)

    action_specs = _require_list(scenario_annotations, "actions")
    seen_signatures: set[tuple[str, str | None, str, str | None]] = set()
    normalized_actions: list[dict[str, Any]] = []
    for item in action_specs:
        if not isinstance(item, dict):
            raise ValueError(f"v2 action annotation must be an object: {scenario_id}")
        signature = _signature(item)
        if signature in seen_signatures:
            raise ValueError(f"duplicate v2 action signature for {scenario_id}: {signature}")
        seen_signatures.add(signature)
        rating = str(item["rating"])
        if rating not in _RATING_ORDER:
            raise ValueError(f"invalid action rating {rating!r}: {scenario_id}")
        normalized_actions.append(dict(item))

    source_actions: list[dict[str, Any]] = []
    recommended = _require_mapping(raw, "recommended_action")
    source_actions.append(recommended)
    alternatives = recommended.get("alternatives", [])
    if not isinstance(alternatives, list):
        raise ValueError(f"recommended_action.alternatives must be an array: {path}")
    source_actions.extend(item for item in alternatives if isinstance(item, dict))
    annotated_source_signatures = {
        (str(item["action_type"]), item.get("target_id")) for item in normalized_actions
    }
    missing_source_actions = {
        _source_signature(item) for item in source_actions
    } - annotated_source_signatures
    if missing_source_actions:
        raise ValueError(
            f"v2 annotations miss source actions for {scenario_id}: {missing_source_actions}"
        )

    rng = _stable_rng(seed, scenario_id)
    rng.shuffle(worlds)
    rng.shuffle(normalized_actions)
    actions: list[ActionCandidate] = []
    action_id_by_signature: dict[tuple[str, str | None, str, str | None], str] = {}
    ratings: dict[str, ActionRating] = {}
    loss_conditions: dict[str, str | None] = {}
    for index, item in enumerate(normalized_actions):
        action_id = f"a{index}"
        signature = _signature(item)
        action_id_by_signature[signature] = action_id
        phase = str(item["phase"])
        if phase not in {"day", "night_divination", "night_guard", "night_attack"}:
            raise ValueError(f"invalid phase {phase!r}: {scenario_id}")
        rating = str(item["rating"])
        actions.append(
            ActionCandidate(
                action_id=action_id,
                phase=phase,  # type: ignore[arg-type]
                actor_id=item.get("actor_id"),
                action_type=str(item["action_type"]),
                target_id=item.get("target_id"),
            )
        )
        ratings[action_id] = rating  # type: ignore[assignment]
        loss_conditions[action_id] = item.get("loss_condition")

    plan: list[PlanSlot] = []
    seen_slots: set[tuple[str, str | None]] = set()
    for item in _require_list(scenario_annotations, "optimal_plan"):
        if not isinstance(item, dict):
            raise ValueError(f"optimal plan item must be an object: {scenario_id}")
        signature = _signature(item)
        action_id = action_id_by_signature.get(signature)
        if action_id is None:
            raise ValueError(
                f"optimal plan action missing from candidates: {scenario_id} {signature}"
            )
        slot = (str(item["phase"]), item.get("actor_id"))
        if slot in seen_slots:
            raise ValueError(f"duplicate optimal plan slot for {scenario_id}: {slot}")
        seen_slots.add(slot)
        if ratings[action_id] != "optimal":
            raise ValueError(f"gold plan action must be optimal: {scenario_id} {signature}")
        plan.append(
            PlanSlot(
                phase=str(item["phase"]),  # type: ignore[arg-type]
                actor_id=item.get("actor_id"),
                action_id=action_id,
            )
        )

    assessment = _require_mapping(raw, "expert_assessment")
    confidence = str(assessment["confidence"])
    if confidence not in {"low", "medium", "high"}:
        raise ValueError(f"invalid confidence {confidence!r}: {scenario_id}")

    return ExpertScenarioV2Case(
        scenario_id=scenario_id,
        log_id=str(raw["log_id"]),
        cutoff_event_id=str(raw["cutoff_event_id"]),
        perspective=dict(perspective),
        facts=tuple(facts),
        rules=rules,
        worlds=tuple(worlds),
        actions=tuple(actions),
        gold_possible_world_ids=frozenset(possible_ids),
        gold_impossible_world_ids=frozenset(impossible_ids),
        gold_world_contradictions=contradictions,
        gold_main_world_ids=frozenset(str(value) for value in assessment["main_world_ids"]),
        gold_alternative_world_ids=frozenset(
            str(value) for value in assessment["alternative_world_ids"]
        ),
        gold_action_ratings=ratings,
        gold_action_loss_conditions=loss_conditions,
        gold_plan=tuple(plan),
        gold_confidence=confidence,  # type: ignore[arg-type]
        source_path=str(path),
    )


def load_v2_cases(
    root: Path,
    *,
    annotations_path: Path,
    seed: int = 0,
    scenario_ids: set[str] | None = None,
) -> list[ExpertScenarioV2Case]:
    annotations = load_v2_annotations(annotations_path)
    cases = [
        load_v2_case(path, annotations=annotations, seed=seed)
        for path in discover_scenario_files(root)
    ]
    if scenario_ids is not None:
        cases = [case for case in cases if case.scenario_id in scenario_ids]
        missing = scenario_ids - {case.scenario_id for case in cases}
        if missing:
            raise ValueError(f"unknown scenario IDs: {sorted(missing)}")
    if not cases:
        raise ValueError(f"no scenarios found under {root}")
    return cases


def render_v2_model_prompt(case: ExpertScenarioV2Case) -> str:
    return json.dumps(case.prompt_payload(), ensure_ascii=False, indent=2)


class LLMV2AnswerProvider:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_tokens: int = 4200,
        temperature: float = 0.2,
    ) -> None:
        self._provider = provider
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def answer(self, case: ExpertScenarioV2Case) -> ExpertScenarioV2Answer | None:
        return await self._provider.generate_structured(
            system=_SYSTEM_PROMPT,
            messages=[Message(role="user", content=render_v2_model_prompt(case))],
            response_schema=ExpertScenarioV2Answer,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )


class BaselineV2AnswerProvider:
    """Free schema-valid baseline that intentionally avoids expert information."""

    async def answer(self, case: ExpertScenarioV2Case) -> ExpertScenarioV2Answer:
        world_judgments = [
            WorldJudgment(
                world_id=item.world_id,
                status="possible",
                contradiction_fact_ids=[],
                contradiction_rule_ids=[],
                rationale="Conservative baseline keeps the candidate possible.",
            )
            for item in case.worlds
        ]
        action_assessments = [
            ActionAssessment(
                action_id=item.action_id,
                rating="acceptable",
                loss_condition=None,
                cited_fact_ids=[],
                rationale="Neutral baseline action assessment.",
            )
            for item in case.actions
        ]
        choices: list[PhaseChoice] = []
        seen: set[tuple[Phase, str | None]] = set()
        for item in case.actions:
            slot = (item.phase, item.actor_id)
            if slot in seen:
                continue
            seen.add(slot)
            choices.append(
                PhaseChoice(
                    phase=item.phase,
                    actor_id=item.actor_id,
                    selected_action_id=item.action_id,
                    rationale="Select the first shuffled action in each slot.",
                )
            )
        possible = [item.world_id for item in case.worlds]
        return ExpertScenarioV2Answer(
            world_judgments=world_judgments,
            main_world_ids=possible[:1],
            alternative_world_ids=possible[1:2],
            action_assessments=action_assessments,
            phase_choices=choices,
            next_observation="Observe the next public result.",
            confidence="low",
            rationale="Conservative phase-aware baseline.",
        )


def load_external_v2_answer(path: Path) -> ExpertScenarioV2Answer:
    return ExpertScenarioV2Answer.model_validate_json(path.read_text(encoding="utf-8"))


def _prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 1.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _jaccard(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    union = predicted | gold
    return len(predicted & gold) / len(union) if union else 1.0


def _confidence_score(predicted: Confidence, gold: Confidence) -> float:
    order = {"low": 0, "medium": 1, "high": 2}
    return 1.0 - abs(order[predicted] - order[gold]) / 2.0


def _slot_label(phase: str, actor_id: str | None) -> str:
    return f"{phase}:{actor_id or '-'}"


def score_v2_answer(
    case: ExpertScenarioV2Case,
    answer: ExpertScenarioV2Answer | None,
) -> ScenarioV2Score:
    if answer is None:
        return ScenarioV2Score(
            scenario_id=case.scenario_id,
            answer_valid=False,
            world_status_accuracy=0.0,
            impossible_world_recall=0.0,
            contradiction_fact_f1=0.0,
            contradiction_rule_f1=0.0,
            world_coverage=0.0,
            main_world_jaccard=0.0,
            alternative_world_jaccard=0.0,
            action_rating_accuracy=0.0,
            action_rating_ordinal_score=0.0,
            action_coverage=0.0,
            phase_choice_exact=0.0,
            phase_choice_utility=0.0,
            catastrophic_action_f1=0.0,
            catastrophic_action_avoidance=0.0,
            consistency_score=0.0,
            consistency_violation_count=0,
            confidence_score=0.0,
            structured_reference_leakage_count=0,
            unknown_world_ids=(),
            unknown_action_ids=(),
            unknown_fact_ids=(),
            unknown_rule_ids=(),
            missing_world_ids=tuple(sorted(case.world_ids)),
            missing_action_ids=tuple(sorted(case.action_ids)),
            missing_plan_slots=tuple(
                sorted(_slot_label(item.phase, item.actor_id) for item in case.gold_plan)
            ),
            overall_score=0.0,
        )

    violations = 0
    world_by_id: dict[str, WorldJudgment] = {}
    duplicate_worlds: set[str] = set()
    for item in answer.world_judgments:
        if item.world_id in world_by_id:
            duplicate_worlds.add(item.world_id)
        world_by_id[item.world_id] = item
    violations += len(duplicate_worlds)

    action_assessment_by_id: dict[str, ActionAssessment] = {}
    duplicate_actions: set[str] = set()
    for item in answer.action_assessments:
        if item.action_id in action_assessment_by_id:
            duplicate_actions.add(item.action_id)
        action_assessment_by_id[item.action_id] = item
    violations += len(duplicate_actions)

    phase_choice_by_slot: dict[tuple[str, str | None], PhaseChoice] = {}
    duplicate_slots: set[tuple[str, str | None]] = set()
    for item in answer.phase_choices:
        slot = (item.phase, item.actor_id)
        if slot in phase_choice_by_slot:
            duplicate_slots.add(slot)
        phase_choice_by_slot[slot] = item
    violations += len(duplicate_slots)

    unknown_world_ids = (
        set(world_by_id) | set(answer.main_world_ids) | set(answer.alternative_world_ids)
    ) - case.world_ids
    selected_action_ids = {item.selected_action_id for item in answer.phase_choices}
    unknown_action_ids = (set(action_assessment_by_id) | selected_action_ids) - case.action_ids

    cited_fact_ids: set[str] = set()
    cited_rule_ids: set[str] = set()
    for item in answer.world_judgments:
        cited_fact_ids.update(item.contradiction_fact_ids)
        cited_rule_ids.update(item.contradiction_rule_ids)
    for item in answer.action_assessments:
        cited_fact_ids.update(item.cited_fact_ids)
    unknown_fact_ids = cited_fact_ids - case.fact_ids
    unknown_rule_ids = cited_rule_ids - case.rule_ids
    leakage_count = (
        len(unknown_world_ids)
        + len(unknown_action_ids)
        + len(unknown_fact_ids)
        + len(unknown_rule_ids)
    )

    known_world_judgments = {
        world_id: item for world_id, item in world_by_id.items() if world_id in case.world_ids
    }
    missing_world_ids = case.world_ids - set(known_world_judgments)
    world_coverage = len(known_world_judgments) / len(case.world_ids) if case.world_ids else 1.0
    status_correct = 0
    predicted_impossible: set[str] = set()
    predicted_fact_pairs: set[str] = set()
    gold_fact_pairs: set[str] = set()
    predicted_rule_pairs: set[str] = set()
    gold_rule_pairs: set[str] = set()
    for world_id in case.world_ids:
        item = known_world_judgments.get(world_id)
        gold_status = "impossible" if world_id in case.gold_impossible_world_ids else "possible"
        if item is not None and item.status == gold_status:
            status_correct += 1
        if item is not None and item.status == "impossible":
            predicted_impossible.add(world_id)
            predicted_fact_pairs.update(
                f"{world_id}|{fact_id}" for fact_id in item.contradiction_fact_ids
            )
            predicted_rule_pairs.update(
                f"{world_id}|{rule_id}" for rule_id in item.contradiction_rule_ids
            )
        if gold_status == "possible" and item is not None:
            if item.contradiction_fact_ids or item.contradiction_rule_ids:
                violations += 1
        gold = case.gold_world_contradictions.get(world_id)
        if gold is not None:
            gold_fact_pairs.update(f"{world_id}|{fact_id}" for fact_id in gold.fact_ids)
            gold_rule_pairs.update(f"{world_id}|{rule_id}" for rule_id in gold.rule_ids)
    world_status_accuracy = status_correct / len(case.world_ids) if case.world_ids else 1.0
    _, impossible_recall, _ = _prf(predicted_impossible, set(case.gold_impossible_world_ids))
    _, _, contradiction_fact_f1 = _prf(predicted_fact_pairs, gold_fact_pairs)
    _, _, contradiction_rule_f1 = _prf(predicted_rule_pairs, gold_rule_pairs)

    own_impossible = {
        world_id for world_id, item in known_world_judgments.items() if item.status == "impossible"
    }
    if (set(answer.main_world_ids) | set(answer.alternative_world_ids)) & own_impossible:
        violations += 1
    if set(answer.main_world_ids) & set(answer.alternative_world_ids):
        violations += 1

    known_action_assessments = {
        action_id: item
        for action_id, item in action_assessment_by_id.items()
        if action_id in case.action_ids
    }
    missing_action_ids = case.action_ids - set(known_action_assessments)
    action_coverage = (
        len(known_action_assessments) / len(case.action_ids) if case.action_ids else 1.0
    )
    exact_ratings = 0
    ordinal_total = 0.0
    predicted_catastrophic: set[str] = set()
    gold_catastrophic = {
        action_id
        for action_id, rating in case.gold_action_ratings.items()
        if rating == "catastrophic"
    }
    for action_id in case.action_ids:
        item = known_action_assessments.get(action_id)
        if item is None:
            continue
        gold_rating = case.gold_action_ratings[action_id]
        exact_ratings += int(item.rating == gold_rating)
        ordinal_total += 1.0 - abs(_RATING_ORDER[item.rating] - _RATING_ORDER[gold_rating]) / 3.0
        if item.rating == "catastrophic":
            predicted_catastrophic.add(action_id)
            if not item.loss_condition:
                violations += 1
    action_rating_accuracy = exact_ratings / len(case.action_ids) if case.action_ids else 1.0
    action_rating_ordinal_score = ordinal_total / len(case.action_ids) if case.action_ids else 1.0
    _, _, catastrophic_f1 = _prf(predicted_catastrophic, gold_catastrophic)

    action_by_id = case.action_by_id()
    gold_plan_by_slot = {(item.phase, item.actor_id): item.action_id for item in case.gold_plan}
    exact_plan = 0
    utility_total = 0.0
    catastrophic_selected = 0
    missing_plan_slots: set[str] = set()
    for slot, gold_action_id in gold_plan_by_slot.items():
        choice = phase_choice_by_slot.get(slot)
        if choice is None:
            missing_plan_slots.add(_slot_label(*slot))
            continue
        selected = action_by_id.get(choice.selected_action_id)
        if selected is None:
            continue
        if (selected.phase, selected.actor_id) != slot:
            violations += 1
            continue
        exact_plan += int(choice.selected_action_id == gold_action_id)
        gold_rating = case.gold_action_ratings[choice.selected_action_id]
        utility_total += _RATING_UTILITY[gold_rating]
        catastrophic_selected += int(gold_rating == "catastrophic")
        own_assessment = known_action_assessments.get(choice.selected_action_id)
        if own_assessment is not None and own_assessment.rating in {"dominated", "catastrophic"}:
            violations += 1
    plan_count = len(gold_plan_by_slot)
    phase_choice_exact = exact_plan / plan_count if plan_count else 1.0
    phase_choice_utility = utility_total / plan_count if plan_count else 1.0
    catastrophic_avoidance = 1.0 - catastrophic_selected / plan_count if plan_count else 1.0

    expected_slots = set(gold_plan_by_slot)
    for slot, choice in phase_choice_by_slot.items():
        if slot not in expected_slots:
            violations += 1
        selected = action_by_id.get(choice.selected_action_id)
        if selected is not None and (selected.phase, selected.actor_id) != slot:
            violations += 1

    consistency_score = max(0.0, 1.0 - 0.15 * violations)
    confidence_score = _confidence_score(answer.confidence, case.gold_confidence)
    hard_logic_score = (
        0.35 * world_status_accuracy
        + 0.30 * impossible_recall
        + 0.10 * contradiction_fact_f1
        + 0.10 * contradiction_rule_f1
        + 0.15 * world_coverage
    )
    weighting_score = (
        _jaccard(set(answer.main_world_ids), set(case.gold_main_world_ids))
        + _jaccard(set(answer.alternative_world_ids), set(case.gold_alternative_world_ids))
    ) / 2.0
    action_rating_score = (action_rating_accuracy + action_rating_ordinal_score) / 2.0
    plan_score = (phase_choice_exact + phase_choice_utility) / 2.0
    safety_score = (catastrophic_f1 + catastrophic_avoidance) / 2.0
    overall = (
        0.30 * hard_logic_score
        + 0.15 * weighting_score
        + 0.15 * action_rating_score
        + 0.25 * plan_score
        + 0.10 * safety_score
        + 0.05 * consistency_score
    )
    if leakage_count:
        overall *= max(0.0, 1.0 - min(0.5, 0.05 * leakage_count))

    return ScenarioV2Score(
        scenario_id=case.scenario_id,
        answer_valid=True,
        world_status_accuracy=world_status_accuracy,
        impossible_world_recall=impossible_recall,
        contradiction_fact_f1=contradiction_fact_f1,
        contradiction_rule_f1=contradiction_rule_f1,
        world_coverage=world_coverage,
        main_world_jaccard=_jaccard(set(answer.main_world_ids), set(case.gold_main_world_ids)),
        alternative_world_jaccard=_jaccard(
            set(answer.alternative_world_ids), set(case.gold_alternative_world_ids)
        ),
        action_rating_accuracy=action_rating_accuracy,
        action_rating_ordinal_score=action_rating_ordinal_score,
        action_coverage=action_coverage,
        phase_choice_exact=phase_choice_exact,
        phase_choice_utility=phase_choice_utility,
        catastrophic_action_f1=catastrophic_f1,
        catastrophic_action_avoidance=catastrophic_avoidance,
        consistency_score=consistency_score,
        consistency_violation_count=violations,
        confidence_score=confidence_score,
        structured_reference_leakage_count=leakage_count,
        unknown_world_ids=tuple(sorted(unknown_world_ids)),
        unknown_action_ids=tuple(sorted(unknown_action_ids)),
        unknown_fact_ids=tuple(sorted(unknown_fact_ids)),
        unknown_rule_ids=tuple(sorted(unknown_rule_ids)),
        missing_world_ids=tuple(sorted(missing_world_ids)),
        missing_action_ids=tuple(sorted(missing_action_ids)),
        missing_plan_slots=tuple(sorted(missing_plan_slots)),
        overall_score=overall,
    )


def summarize_v2_scores(scores: list[ScenarioV2Score]) -> BenchmarkV2Summary:
    if not scores:
        raise ValueError("at least one score is required")
    count = len(scores)
    hard_logic = [
        0.35 * item.world_status_accuracy
        + 0.30 * item.impossible_world_recall
        + 0.10 * item.contradiction_fact_f1
        + 0.10 * item.contradiction_rule_f1
        + 0.15 * item.world_coverage
        for item in scores
    ]
    weighting = [
        (item.main_world_jaccard + item.alternative_world_jaccard) / 2.0 for item in scores
    ]
    return BenchmarkV2Summary(
        scenario_count=count,
        valid_answer_rate=sum(item.answer_valid for item in scores) / count,
        mean_overall_score=sum(item.overall_score for item in scores) / count,
        mean_hard_logic_score=sum(hard_logic) / count,
        mean_weighting_score=sum(weighting) / count,
        mean_action_rating_accuracy=sum(item.action_rating_accuracy for item in scores) / count,
        mean_phase_choice_exact=sum(item.phase_choice_exact for item in scores) / count,
        mean_phase_choice_utility=sum(item.phase_choice_utility for item in scores) / count,
        catastrophic_action_avoidance=(
            sum(item.catastrophic_action_avoidance for item in scores) / count
        ),
        mean_consistency_score=sum(item.consistency_score for item in scores) / count,
        total_structured_reference_leakage_count=sum(
            item.structured_reference_leakage_count for item in scores
        ),
    )


def render_v2_report(
    *,
    provider: str,
    model: str,
    scores: list[ScenarioV2Score],
    summary: BenchmarkV2Summary,
    metrics_summary: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Expert cutoff benchmark v2",
        "",
        f"- Provider: `{provider}`",
        f"- Model: `{model}`",
        f"- Scenarios: {summary.scenario_count}",
        f"- Valid answer rate: {summary.valid_answer_rate:.1%}",
        f"- Overall score: {summary.mean_overall_score:.3f}",
        f"- Hard-logic score: {summary.mean_hard_logic_score:.3f}",
        f"- Weighting score: {summary.mean_weighting_score:.3f}",
        f"- Action-rating accuracy: {summary.mean_action_rating_accuracy:.1%}",
        f"- Phase-plan exact: {summary.mean_phase_choice_exact:.1%}",
        f"- Phase-plan utility: {summary.mean_phase_choice_utility:.1%}",
        (
            "- Catastrophic-action avoidance: "
            f"{summary.catastrophic_action_avoidance:.1%}"
        ),
        f"- Internal consistency: {summary.mean_consistency_score:.1%}",
        (
            "- Structured unknown-reference count: "
            f"{summary.total_structured_reference_leakage_count}"
        ),
        "",
        "## Per-scenario scores",
        "",
        "| Scenario | Overall | Logic | Rating | Plan exact/utility | Safety | Consistency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in scores:
        logic = (
            0.35 * item.world_status_accuracy
            + 0.30 * item.impossible_world_recall
            + 0.10 * item.contradiction_fact_f1
            + 0.10 * item.contradiction_rule_f1
            + 0.15 * item.world_coverage
        )
        lines.append(
            "| "
            f"`{item.scenario_id}` | {item.overall_score:.3f} | {logic:.2f} | "
            f"{item.action_rating_accuracy:.2f} | "
            f"{item.phase_choice_exact:.2f}/{item.phase_choice_utility:.2f} | "
            f"{item.catastrophic_action_avoidance:.2f} | {item.consistency_score:.2f} |"
        )
    lines.extend(
        [
            "",
            "## What v2 fixes",
            "",
            "- Day and night actions are scored as one multi-phase plan.",
            "- Every action is graded optimal/acceptable/dominated/catastrophic.",
            "- Impossible worlds require explicit fact and rule IDs.",
            "- Selecting an action the answer itself calls dominated/catastrophic is penalized.",
            "- v1 remains available for historical comparison.",
            "",
            "## Remaining limits",
            "",
            "- Facts, worlds, rules, and actions are still curated closed sets.",
            "- Free-text semantic correctness is not judged beyond structured consistency.",
            "- Full action regret and raw-log extraction remain future work.",
        ]
    )
    if metrics_summary is not None:
        lines.extend(
            [
                "",
                "## Provider metrics",
                "",
                "```json",
                json.dumps(metrics_summary, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"
