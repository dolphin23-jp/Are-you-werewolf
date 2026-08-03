"""The decision a turn expresses, fixed before the words are generated.

Two reasoning systems were running side by side: the runtime decided the vote,
and the model's free-text memo decided what the discussion argued about. They
disagreed, so an AI could spend a day pushing one name and then vote for
another with nothing in between.

`DiscussionDecision` closes that. It is built from the belief engine before the
turn is generated and the model is asked only to say it -- tone, length, order
of argument, who to question. The conclusions are not negotiable.

Human speech goes the other way. `ArgumentEvent` turns one message into
structured claims *once*, in code, and each seat then weighs those claims
through its own traits and trust. One sentence therefore costs zero model calls
however many AIs are listening, and they still react differently -- which is the
difference between a table and a chorus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from app.ai.reasoning.belief.corrections import FactCorrection, parse_fact_corrections
from app.ai.reasoning.facts import PublicFactLedger, mentions_player


class SpeechGoal(StrEnum):
    PUBLISH_RESULT = "publish_result"
    CLAIM_ROLE = "claim_role"
    ANSWER_QUESTION = "answer_question"
    DEFEND = "defend"
    PRESS_CANDIDATE = "press_candidate"
    REASSESS = "reassess"
    RECOVER_STORY = "recover_story"
    OBSERVE = "observe"


@dataclass(frozen=True)
class BeliefChange:
    subject_id: str
    before: float
    after: float
    reason: str

    @property
    def direction(self) -> str:
        return "上げた" if self.after > self.before else "下げた"


@dataclass(frozen=True)
class DiscussionDecision:
    """What this turn concludes. The model renders it; it does not revise it.

    Everything here is derived from the belief engine, so the sentence the table
    hears and the ballot cast later come from the same place.
    """

    speaker_id: str
    execution_target: str | None
    alternative_target: str | None
    target_confidence_band: str
    supporting_evidence: tuple[str, ...] = ()
    counter_evidence: tuple[str, ...] = ()
    belief_changes: tuple[BeliefChange, ...] = ()
    strongest_countercase: str = ""
    public_story_status: str | None = None
    speech_goal: SpeechGoal = SpeechGoal.OBSERVE

    def render_brief(self) -> str:
        """The block handed to the model. Facts only, in fixed wording."""
        lines = [f"【この発言で述べる結論】発言目的: {self.speech_goal.value}"]
        if self.execution_target:
            lines.append(
                f"- 第一処刑候補: {self.execution_target}"
                f"（確度: {self.target_confidence_band}）"
            )
        else:
            lines.append("- 第一処刑候補: なし（まだ絞れていない）")
        if self.alternative_target:
            lines.append(f"- 第二候補: {self.alternative_target}")
        if self.supporting_evidence:
            lines.append("- 根拠: " + " / ".join(self.supporting_evidence))
        if self.counter_evidence:
            lines.append("- 反対材料: " + " / ".join(self.counter_evidence))
        for change in self.belief_changes:
            lines.append(
                f"- {change.subject_id}への疑いを{change.direction}: {change.reason}"
            )
        if self.strongest_countercase:
            lines.append(f"- 処刑しない最強の理由: {self.strongest_countercase}")
        if self.public_story_status:
            lines.append(f"- 自分の公開主張の状態: {self.public_story_status}")
        lines.append(
            "上の結論・根拠・数値は変更しないでください。口調、長さ、説明の順序、"
            "誰に質問するかだけがあなたの裁量です。"
        )
        return "\n".join(lines)


# -- human speech, parsed once --


class ConclusionType(StrEnum):
    FACT_CORRECTION = "fact_correction"
    ACCUSATION = "accusation"
    DEFENCE = "defence"
    STRATEGIC_CLAIM = "strategic_claim"
    CLOSED_WORLD_CHALLENGE = "closed_world_challenge"


@dataclass(frozen=True)
class ArgumentPremise:
    text: str
    references_player_id: str | None = None


@dataclass(frozen=True)
class StrategicClaim:
    """A claim about how the game should be played, not about what happened.

    Never auto-accepted. "We should keep the seer and hang a grey" is advice,
    and each seat weighs advice by who gave it and how much it already believes.
    """

    claim_type: str
    subject_id: str | None
    text: str


@dataclass(frozen=True)
class ArgumentEvent:
    argument_id: str
    speaker_id: str
    conclusion_type: ConclusionType
    conclusion_target_id: str | None
    source_message_id: str
    premises: tuple[ArgumentPremise, ...] = ()
    mentioned_players: tuple[str, ...] = ()
    factual_claims: tuple[FactCorrection, ...] = ()
    strategic_claims: tuple[StrategicClaim, ...] = field(default_factory=tuple)
    rhetorical_strength: float = 1.0


# Deliberately narrow patterns. A sentence these miss becomes ordinary
# conversation, which is the safe failure -- inventing an accusation out of a
# vague complaint is not.
_SPARE_RE = re.compile(r"(?:処刑|吊[るり])(?:する)?必要(?:は)?(?:ない|ありません)")
_BANDWAGON_RE = re.compile(r"(?:便乗|乗っかって|同調して|流されて)")
_KEEP_ROLE_RE = re.compile(r"(?:占い師?|霊媒師?|共有者?)を(?:残|守)")
_CLOSED_WORLD_RE = re.compile(
    r"(?:真(?:の)?)?(?:占い師?|霊媒師?)(?:は)?(?:どこ|誰|不在|いなく|いない)"
)
_SUSPECT_RE = re.compile(r"(?:が|は)(?:怪しい|あやしい|黒|狼)")
_EMPHASIS_RE = re.compile(r"(?:絶対|明らかに|間違いなく|確実に|必ず)")


def parse_argument(
    text: str, ledger: PublicFactLedger, speaker_id: str, source_message_id: str = ""
) -> ArgumentEvent | None:
    """Turn one human message into structured claims, once, in code.

    Runs a single time per message no matter how many AIs are listening. The
    alternative -- asking a model to read it per seat -- costs sixteen requests
    to produce sixteen slightly different readings of the same sentence.
    """
    mentioned = tuple(
        pid
        for pid in ledger.known_player_ids()
        if pid != speaker_id and mentions_player(text, pid, ledger.name_of(pid))
    )
    corrections = tuple(parse_fact_corrections(text, ledger, speaker_id))
    strategic: list[StrategicClaim] = []
    conclusion = ConclusionType.STRATEGIC_CLAIM
    target: str | None = mentioned[0] if mentioned else None

    if corrections:
        conclusion = ConclusionType.FACT_CORRECTION
        target = corrections[0].subject_id
    elif _SPARE_RE.search(text):
        conclusion = ConclusionType.DEFENCE
        strategic.append(
            StrategicClaim(claim_type="spare_target", subject_id=target, text=text)
        )
    elif _BANDWAGON_RE.search(text) and mentioned:
        conclusion = ConclusionType.ACCUSATION
        strategic.append(
            StrategicClaim(claim_type="bandwagon", subject_id=mentioned[0], text=text)
        )
    elif _CLOSED_WORLD_RE.search(text):
        conclusion = ConclusionType.CLOSED_WORLD_CHALLENGE
        strategic.append(
            StrategicClaim(claim_type="missing_true_role", subject_id=target, text=text)
        )
    elif _KEEP_ROLE_RE.search(text):
        strategic.append(
            StrategicClaim(claim_type="protect_power_role", subject_id=target, text=text)
        )
    elif _SUSPECT_RE.search(text) and mentioned:
        conclusion = ConclusionType.ACCUSATION
        target = mentioned[0]
    elif not mentioned:
        return None

    return ArgumentEvent(
        argument_id=f"arg:{speaker_id}:{source_message_id or len(text)}",
        speaker_id=speaker_id,
        conclusion_type=conclusion,
        conclusion_target_id=target,
        source_message_id=source_message_id,
        premises=tuple(
            ArgumentPremise(text=part.strip())
            for part in re.split(r"[。！？!?\n]", text)
            if part.strip()
        ),
        mentioned_players=mentioned,
        factual_claims=corrections,
        strategic_claims=tuple(strategic),
        rhetorical_strength=1.3 if _EMPHASIS_RE.search(text) else 1.0,
    )
