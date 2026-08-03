"""Human-readable qualification report and review checklist."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.eval.reasoning_analyzer import ReasoningQualityReport
from app.eval.release_gate import ReleaseGateResult

REVIEW_ITEMS = (
    "AIごとに発言内容の差があった",
    "差が単なる口調の違いではなかった",
    "人間の主張へ内容面で返答した",
    "事実訂正後に古い根拠を繰り返さなかった",
    "多数派への追従だけで議論が決まらなかった",
    "少数意見に具体的根拠があった",
    "人狼の発言が私的情報を漏らしていなかった",
    "能力者が結果を正しく公開した",
    "騙りの物語が一貫していた、または破綻後に反応した",
    "公開発言と投票が自然につながっていた",
)


@dataclass(frozen=True)
class HumanTranscriptReview:
    game_id: str
    reviewer: str
    reviewed_at: str
    answers: dict[str, bool | None] = field(default_factory=dict)
    notes: str = ""
    revision: int = 1

    @property
    def complete(self) -> bool:
        return bool(self.reviewer and self.reviewed_at) and all(
            self.answers.get(item) is not None for item in REVIEW_ITEMS
        )

    @classmethod
    def from_json(cls, path: Path) -> HumanTranscriptReview:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def empty_review(game_id: str) -> HumanTranscriptReview:
    return HumanTranscriptReview(
        game_id=game_id,
        reviewer="",
        reviewed_at="",
        answers={item: None for item in REVIEW_ITEMS},
    )


def render_release_report(
    quality: ReasoningQualityReport,
    gate: ReleaseGateResult,
    *,
    live_evaluation: bool,
) -> str:
    metrics = "\n".join(f"- {key}: {value}" for key, value in quality.to_dict().items())
    checklist = "\n".join(f"- [ ] {item}" for item in REVIEW_ITEMS)
    return (
        "# v2 Release Qualification\n\n"
        f"**ReleaseDecision: {gate.decision.value.upper()}**\n\n"
        f"Live model evaluation: {'executed' if live_evaluation else 'not executed'}\n\n"
        "## Recomputed metrics\n" + metrics + "\n\n"
        "## Human transcript review\n" + checklist + "\n\n### Reviewer notes\n\n"
    )
