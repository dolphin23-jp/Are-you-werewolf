"""Closed-set evaluation for expert-reviewed real-game cutoffs.

The reviewed scenario JSON files contain both the playable cutoff and the
expert answer. This module deliberately exposes only a restricted task view to
the model:

- public facts (and actor-private facts only for non-public perspectives),
- unlabeled candidate worlds,
- opaque candidate actions.

The expert weighting, recommended action, corrections, review metadata and
provenance never enter the prompt. This first harness therefore measures
reasoning over curated facts. Raw-log fact extraction and full semantic
perspective-leakage detection remain blocked until canonical public/full event
streams exist.
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


class ExpertScenarioAnswer(BaseModel):
    """Structured answer contract shared by live models and external answers."""

    model_config = ConfigDict(extra="forbid")

    possible_world_ids: list[str] = Field(default_factory=list)
    impossible_world_ids: list[str] = Field(default_factory=list)
    main_world_ids: list[str] = Field(default_factory=list)
    alternative_world_ids: list[str] = Field(default_factory=list)
    recommended_action_id: str
    alternative_action_ids: list[str] = Field(default_factory=list)
    catastrophic_action_ids: list[str] = Field(default_factory=list)
    cited_fact_ids: list[str] = Field(default_factory=list)
    next_observation: str
    confidence: Confidence
    rationale: str


class AnswerProvider(Protocol):
    async def answer(self, case: ExpertScenarioCase) -> ExpertScenarioAnswer | None: ...


@dataclass(frozen=True)
class WorldCandidate:
    world_id: str
    summary: str
    required_assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    action_type: str
    target_id: str | None


@dataclass(frozen=True)
class ExpertScenarioCase:
    scenario_id: str
    log_id: str
    cutoff_event_id: str
    perspective: dict[str, Any]
    facts: tuple[dict[str, Any], ...]
    worlds: tuple[WorldCandidate, ...]
    actions: tuple[ActionCandidate, ...]
    gold_possible_world_ids: frozenset[str]
    gold_impossible_world_ids: frozenset[str]
    gold_main_world_ids: frozenset[str]
    gold_alternative_world_ids: frozenset[str]
    gold_action_id: str
    gold_catastrophic_action_ids: frozenset[str]
    gold_confidence: Confidence
    source_path: str

    @property
    def world_ids(self) -> frozenset[str]:
        return frozenset(world.world_id for world in self.worlds)

    @property
    def action_ids(self) -> frozenset[str]:
        return frozenset(action.action_id for action in self.actions)

    @property
    def fact_ids(self) -> frozenset[str]:
        return frozenset(str(fact["fact_id"]) for fact in self.facts)

    def prompt_payload(self) -> dict[str, Any]:
        """Return the spoiler-safe closed-set task presented to a model."""

        return {
            "task_version": "expert-scenario-closed-set-v1",
            "scenario_id": self.scenario_id,
            "log_id": self.log_id,
            "cutoff_event_id": self.cutoff_event_id,
            "perspective": self.perspective,
            "observed_facts": list(self.facts),
            "world_candidates": [
                {
                    "world_id": world.world_id,
                    "summary": world.summary,
                    "required_assumptions": list(world.required_assumptions),
                }
                for world in self.worlds
            ],
            "action_candidates": [asdict(action) for action in self.actions],
            "instructions": {
                "world_partition": (
                    "Classify every world candidate as possible or impossible. "
                    "Unlikely is not impossible without a rules contradiction."
                ),
                "belief": "Choose main and alternative worlds only from possible candidates.",
                "action": (
                    "Choose today's best action separately from role likelihood. "
                    "Avoid an immediate faction-loss action even if its target is likely hostile."
                ),
                "evidence": "Cite only fact_id values included above.",
                "perspective": (
                    "Use only the information in this payload; do not infer later truth."
                ),
            },
        }


@dataclass(frozen=True)
class ScenarioScore:
    scenario_id: str
    answer_valid: bool
    possible_precision: float
    possible_recall: float
    possible_f1: float
    impossible_precision: float
    impossible_recall: float
    impossible_f1: float
    classification_coverage: float
    partition_integrity: float
    main_world_jaccard: float
    alternative_world_jaccard: float
    action_exact: float
    catastrophic_action_f1: float
    catastrophic_action_avoidance: float
    confidence_score: float
    citation_validity: float
    structured_reference_leakage_count: int
    unknown_world_ids: tuple[str, ...]
    unknown_action_ids: tuple[str, ...]
    unknown_fact_ids: tuple[str, ...]
    unclassified_world_ids: tuple[str, ...]
    overlapping_world_ids: tuple[str, ...]
    overall_score: float
    selected_action_id: str | None
    gold_action_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSummary:
    scenario_count: int
    valid_answer_rate: float
    mean_overall_score: float
    mean_logic_score: float
    mean_weighting_score: float
    action_accuracy: float
    catastrophic_action_avoidance: float
    mean_confidence_score: float
    total_structured_reference_leakage_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CATASTROPHIC_MARKERS: tuple[tuple[str, str], ...] = (
    ("即狐勝利", "immediate_fox_win"),
    ("狐勝利", "fox_win"),
    ("即狼勝利", "immediate_wolf_win"),
    ("狼勝利", "wolf_win"),
    ("即敗北", "immediate_loss"),
    ("immediate fox win", "immediate_fox_win"),
    ("immediate wolf win", "immediate_wolf_win"),
    ("catastrophic", "catastrophic"),
)

_SYSTEM_PROMPT = """You are evaluating a single 17A werewolf cutoff.
Use only the supplied perspective and observed facts. Keep observation, hard
logic, soft weighting, and action choice separate. Treat unlikely worlds as
possible unless a rules contradiction makes them impossible. The most likely
wolf is not automatically today's best execution; account for fox/LW loss
conditions and information gained by the next night. Return only the requested
structured object."""


def _stable_rng(seed: int, scenario_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{scenario_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _action_signature(action: dict[str, Any]) -> tuple[str, str | None]:
    return str(action["action_type"]), action.get("target_id")


def _is_catastrophic(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker.lower() in lowered for marker, _ in _CATASTROPHIC_MARKERS)


def discover_scenario_files(root: Path) -> list[Path]:
    """Find reviewed scenario JSON files while ignoring README/metadata files."""

    return sorted(path for path in root.rglob("*.json") if path.is_file())


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


def load_case(path: Path, *, seed: int = 0) -> ExpertScenarioCase:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"scenario must be an object: {path}")

    scenario_id = str(raw["scenario_id"])
    perspective = _require_mapping(raw, "perspective")
    public_facts = _require_list(raw, "public_facts")
    private_facts = _require_list(raw, "private_facts")

    facts: list[dict[str, Any]] = []
    for fact in public_facts:
        if not isinstance(fact, dict):
            raise ValueError(f"public fact must be an object: {path}")
        facts.append(
            {
                "fact_id": str(fact["fact_id"]),
                "visibility": "public",
                "statement": str(fact["statement"]),
            }
        )
    if perspective.get("kind") != "public":
        for fact in private_facts:
            if not isinstance(fact, dict):
                raise ValueError(f"private fact must be an object: {path}")
            facts.append(
                {
                    "fact_id": str(fact["fact_id"]),
                    "visibility": "private",
                    "statement": str(fact["statement"]),
                }
            )

    possible_worlds = _require_list(raw, "possible_worlds")
    impossible_worlds = _require_list(raw, "impossible_worlds")
    worlds: list[WorldCandidate] = []
    possible_ids: set[str] = set()
    impossible_ids: set[str] = set()
    for world in possible_worlds:
        if not isinstance(world, dict):
            raise ValueError(f"possible world must be an object: {path}")
        world_id = str(world["world_id"])
        possible_ids.add(world_id)
        assumptions = world.get("required_assumptions", [])
        worlds.append(
            WorldCandidate(
                world_id=world_id,
                summary=str(world["summary"]),
                required_assumptions=tuple(str(item) for item in assumptions),
            )
        )
    for world in impossible_worlds:
        if not isinstance(world, dict):
            raise ValueError(f"impossible world must be an object: {path}")
        world_id = str(world["world_id"])
        impossible_ids.add(world_id)
        worlds.append(
            WorldCandidate(
                world_id=world_id,
                summary=str(world["summary"]),
                required_assumptions=(),
            )
        )
    if possible_ids & impossible_ids:
        overlap = sorted(possible_ids & impossible_ids)
        raise ValueError(f"world IDs occur in both possible/impossible: {overlap}")

    recommended = _require_mapping(raw, "recommended_action")
    alternative_actions = recommended.get("alternatives", [])
    if not isinstance(alternative_actions, list):
        raise ValueError(f"recommended_action.alternatives must be an array: {path}")

    action_by_signature: dict[tuple[str, str | None], dict[str, Any]] = {}
    recommended_signature = _action_signature(recommended)
    action_by_signature[recommended_signature] = recommended
    catastrophic_signatures: set[tuple[str, str | None]] = set()
    for action in alternative_actions:
        if not isinstance(action, dict):
            raise ValueError(f"alternative action must be an object: {path}")
        signature = _action_signature(action)
        action_by_signature.setdefault(signature, action)
        if _is_catastrophic(str(action.get("reason", ""))):
            catastrophic_signatures.add(signature)

    rng = _stable_rng(seed, scenario_id)
    rng.shuffle(worlds)
    action_items = list(action_by_signature.items())
    rng.shuffle(action_items)

    actions: list[ActionCandidate] = []
    action_id_by_signature: dict[tuple[str, str | None], str] = {}
    for index, (signature, action) in enumerate(action_items):
        action_id = f"a{index}"
        action_id_by_signature[signature] = action_id
        actions.append(
            ActionCandidate(
                action_id=action_id,
                action_type=str(action["action_type"]),
                target_id=action.get("target_id"),
            )
        )

    assessment = _require_mapping(raw, "expert_assessment")
    confidence = str(assessment["confidence"])
    if confidence not in {"low", "medium", "high"}:
        raise ValueError(f"invalid expert confidence {confidence!r}: {path}")

    return ExpertScenarioCase(
        scenario_id=scenario_id,
        log_id=str(raw["log_id"]),
        cutoff_event_id=str(raw["cutoff_event_id"]),
        perspective=dict(perspective),
        facts=tuple(facts),
        worlds=tuple(worlds),
        actions=tuple(actions),
        gold_possible_world_ids=frozenset(possible_ids),
        gold_impossible_world_ids=frozenset(impossible_ids),
        gold_main_world_ids=frozenset(str(item) for item in assessment["main_world_ids"]),
        gold_alternative_world_ids=frozenset(
            str(item) for item in assessment["alternative_world_ids"]
        ),
        gold_action_id=action_id_by_signature[recommended_signature],
        gold_catastrophic_action_ids=frozenset(
            action_id_by_signature[signature]
            for signature in catastrophic_signatures
            if signature in action_id_by_signature
        ),
        gold_confidence=confidence,  # type: ignore[arg-type]
        source_path=str(path),
    )


def load_cases(
    root: Path,
    *,
    seed: int = 0,
    scenario_ids: set[str] | None = None,
) -> list[ExpertScenarioCase]:
    cases = [load_case(path, seed=seed) for path in discover_scenario_files(root)]
    if scenario_ids is not None:
        cases = [case for case in cases if case.scenario_id in scenario_ids]
        missing = scenario_ids - {case.scenario_id for case in cases}
        if missing:
            raise ValueError(f"unknown scenario IDs: {sorted(missing)}")
    if not cases:
        raise ValueError(f"no scenarios found under {root}")
    return cases


def render_model_prompt(case: ExpertScenarioCase) -> str:
    return json.dumps(case.prompt_payload(), ensure_ascii=False, indent=2)


class LLMAnswerProvider:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_tokens: int = 2400,
        temperature: float = 0.2,
    ) -> None:
        self._provider = provider
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def answer(self, case: ExpertScenarioCase) -> ExpertScenarioAnswer | None:
        return await self._provider.generate_structured(
            system=_SYSTEM_PROMPT,
            messages=[Message(role="user", content=render_model_prompt(case))],
            response_schema=ExpertScenarioAnswer,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )


class BaselineAnswerProvider:
    """Zero-cost smoke-test baseline; intentionally weak but schema-valid."""

    async def answer(self, case: ExpertScenarioCase) -> ExpertScenarioAnswer:
        worlds = [world.world_id for world in case.worlds]
        actions = [action.action_id for action in case.actions]
        facts = [str(fact["fact_id"]) for fact in case.facts]
        return ExpertScenarioAnswer(
            possible_world_ids=worlds,
            impossible_world_ids=[],
            main_world_ids=worlds[:1],
            alternative_world_ids=worlds[1:2],
            recommended_action_id=actions[0],
            alternative_action_ids=actions[1:2],
            catastrophic_action_ids=[],
            cited_fact_ids=facts[:1],
            next_observation="次の公開結果を確認する。",
            confidence="low",
            rationale="全候補を広く残す単純ベースライン。",
        )


def load_external_answer(path: Path) -> ExpertScenarioAnswer:
    return ExpertScenarioAnswer.model_validate_json(path.read_text(encoding="utf-8"))


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


def score_answer(
    case: ExpertScenarioCase,
    answer: ExpertScenarioAnswer | None,
) -> ScenarioScore:
    if answer is None:
        return ScenarioScore(
            scenario_id=case.scenario_id,
            answer_valid=False,
            possible_precision=0.0,
            possible_recall=0.0,
            possible_f1=0.0,
            impossible_precision=0.0,
            impossible_recall=0.0,
            impossible_f1=0.0,
            classification_coverage=0.0,
            partition_integrity=0.0,
            main_world_jaccard=0.0,
            alternative_world_jaccard=0.0,
            action_exact=0.0,
            catastrophic_action_f1=0.0,
            catastrophic_action_avoidance=0.0,
            confidence_score=0.0,
            citation_validity=0.0,
            structured_reference_leakage_count=0,
            unknown_world_ids=(),
            unknown_action_ids=(),
            unknown_fact_ids=(),
            unclassified_world_ids=tuple(sorted(case.world_ids)),
            overlapping_world_ids=(),
            overall_score=0.0,
            selected_action_id=None,
            gold_action_id=case.gold_action_id,
        )

    predicted_possible = set(answer.possible_world_ids)
    predicted_impossible = set(answer.impossible_world_ids)
    predicted_main = set(answer.main_world_ids)
    predicted_alternative = set(answer.alternative_world_ids)
    predicted_catastrophic = set(answer.catastrophic_action_ids)
    predicted_actions = {
        answer.recommended_action_id,
        *answer.alternative_action_ids,
        *answer.catastrophic_action_ids,
    }
    predicted_facts = set(answer.cited_fact_ids)

    unknown_world_ids = (
        predicted_possible | predicted_impossible | predicted_main | predicted_alternative
    ) - case.world_ids
    unknown_action_ids = predicted_actions - case.action_ids
    unknown_fact_ids = predicted_facts - case.fact_ids

    classified_known = (predicted_possible | predicted_impossible) & case.world_ids
    unclassified = case.world_ids - classified_known
    overlap = predicted_possible & predicted_impossible & case.world_ids
    coverage = len(classified_known) / len(case.world_ids) if case.world_ids else 1.0
    partition_integrity = 1.0 if not overlap and not unknown_world_ids else 0.0

    possible_precision, possible_recall, possible_f1 = _prf(
        predicted_possible, set(case.gold_possible_world_ids)
    )
    impossible_precision, impossible_recall, impossible_f1 = _prf(
        predicted_impossible, set(case.gold_impossible_world_ids)
    )
    _, _, catastrophic_f1 = _prf(
        predicted_catastrophic, set(case.gold_catastrophic_action_ids)
    )

    action_exact = float(answer.recommended_action_id == case.gold_action_id)
    selected_is_catastrophic = answer.recommended_action_id in case.gold_catastrophic_action_ids
    catastrophic_avoidance = float(not selected_is_catastrophic)

    citation_validity = (
        len(predicted_facts & case.fact_ids) / len(predicted_facts) if predicted_facts else 0.0
    )
    leakage_count = len(unknown_world_ids) + len(unknown_action_ids) + len(unknown_fact_ids)

    logic_score = (
        possible_f1 + impossible_f1 + coverage + partition_integrity + citation_validity
    ) / 5.0
    weighting_score = (
        _jaccard(predicted_main, set(case.gold_main_world_ids))
        + _jaccard(predicted_alternative, set(case.gold_alternative_world_ids))
    ) / 2.0
    safety_score = (catastrophic_f1 + catastrophic_avoidance) / 2.0
    confidence_score = _confidence_score(answer.confidence, case.gold_confidence)
    overall = (
        0.30 * logic_score
        + 0.20 * weighting_score
        + 0.35 * action_exact
        + 0.10 * safety_score
        + 0.05 * confidence_score
    )
    if leakage_count:
        overall *= max(0.0, 1.0 - min(0.5, 0.05 * leakage_count))

    return ScenarioScore(
        scenario_id=case.scenario_id,
        answer_valid=True,
        possible_precision=possible_precision,
        possible_recall=possible_recall,
        possible_f1=possible_f1,
        impossible_precision=impossible_precision,
        impossible_recall=impossible_recall,
        impossible_f1=impossible_f1,
        classification_coverage=coverage,
        partition_integrity=partition_integrity,
        main_world_jaccard=_jaccard(predicted_main, set(case.gold_main_world_ids)),
        alternative_world_jaccard=_jaccard(
            predicted_alternative, set(case.gold_alternative_world_ids)
        ),
        action_exact=action_exact,
        catastrophic_action_f1=catastrophic_f1,
        catastrophic_action_avoidance=catastrophic_avoidance,
        confidence_score=confidence_score,
        citation_validity=citation_validity,
        structured_reference_leakage_count=leakage_count,
        unknown_world_ids=tuple(sorted(unknown_world_ids)),
        unknown_action_ids=tuple(sorted(unknown_action_ids)),
        unknown_fact_ids=tuple(sorted(unknown_fact_ids)),
        unclassified_world_ids=tuple(sorted(unclassified)),
        overlapping_world_ids=tuple(sorted(overlap)),
        overall_score=overall,
        selected_action_id=answer.recommended_action_id,
        gold_action_id=case.gold_action_id,
    )


def summarize_scores(scores: list[ScenarioScore]) -> BenchmarkSummary:
    if not scores:
        raise ValueError("at least one score is required")

    count = len(scores)
    logic_scores = [
        (
            score.possible_f1
            + score.impossible_f1
            + score.classification_coverage
            + score.partition_integrity
            + score.citation_validity
        )
        / 5.0
        for score in scores
    ]
    weighting_scores = [
        (score.main_world_jaccard + score.alternative_world_jaccard) / 2.0
        for score in scores
    ]
    return BenchmarkSummary(
        scenario_count=count,
        valid_answer_rate=sum(score.answer_valid for score in scores) / count,
        mean_overall_score=sum(score.overall_score for score in scores) / count,
        mean_logic_score=sum(logic_scores) / count,
        mean_weighting_score=sum(weighting_scores) / count,
        action_accuracy=sum(score.action_exact for score in scores) / count,
        catastrophic_action_avoidance=(
            sum(score.catastrophic_action_avoidance for score in scores) / count
        ),
        mean_confidence_score=sum(score.confidence_score for score in scores) / count,
        total_structured_reference_leakage_count=sum(
            score.structured_reference_leakage_count for score in scores
        ),
    )


def render_report(
    *,
    provider: str,
    model: str,
    scores: list[ScenarioScore],
    summary: BenchmarkSummary,
    metrics_summary: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Expert cutoff benchmark",
        "",
        f"- Provider: `{provider}`",
        f"- Model: `{model}`",
        f"- Scenarios: {summary.scenario_count}",
        f"- Valid answer rate: {summary.valid_answer_rate:.1%}",
        f"- Overall score: {summary.mean_overall_score:.3f}",
        f"- Logic score: {summary.mean_logic_score:.3f}",
        f"- Weighting score: {summary.mean_weighting_score:.3f}",
        f"- Action accuracy: {summary.action_accuracy:.1%}",
        (
            "- Catastrophic-action avoidance: "
            f"{summary.catastrophic_action_avoidance:.1%}"
        ),
        (
            "- Structured unknown-reference count: "
            f"{summary.total_structured_reference_leakage_count}"
        ),
        "",
        "## Per-scenario scores",
        "",
        "| Scenario | Overall | Logic P/I F1 | Main | Action | Safety | Unknown refs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for score in scores:
        lines.append(
            "| "
            f"`{score.scenario_id}` | {score.overall_score:.3f} | "
            f"{score.possible_f1:.2f}/{score.impossible_f1:.2f} | "
            f"{score.main_world_jaccard:.2f} | {score.action_exact:.0f} | "
            f"{score.catastrophic_action_avoidance:.0f} | "
            f"{score.structured_reference_leakage_count} |"
        )

    lines.extend(
        [
            "",
            "## Limitations of v1",
            "",
            (
                "- The model receives curated observed facts, not raw chronological events; "
                "fact extraction is not measured yet."
            ),
            (
                "- Candidate worlds and actions are closed-set. Open-ended world recall is "
                "not measured yet."
            ),
            (
                "- Rationale and next-observation quality are preserved for review but not "
                "semantically scored."
            ),
            (
                "- Perspective leakage is measured only as unknown structured references, "
                "not by a full semantic judge."
            ),
            (
                "- Catastrophic actions are inferred from explicit loss wording in expert "
                "alternative reasons; canonical annotations should replace this heuristic."
            ),
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
