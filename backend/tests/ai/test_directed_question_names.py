"""A structured question counts only when the line the table heard names its target (4-A2).

Seed 1 of run 13 recorded 44 directed questions; 43 of the texts shown to
their targets as 「m61 で言われたこと」 were the model's paraphrase in the
structured field, not the line it posted, and some lines never named the
seat at all. So a question whose target the model-written text does not
name is dropped -- from the recorded targets, from the target's unanswered
list and from reply booking -- and counted. A line written by code carries
no question of the model's. The question a target is shown is the asker's
own sentence that names it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.ai.coordinator import AICoordinator
from app.ai.metrics import MetricsCollector
from app.ai.provider.base import Message, SchemaT
from app.ai.reasoning.runtime import ReasoningRuntime
from app.ai.schemas import (
    ChatDiscussionOutput,
    DirectedQuestion,
    DiscussionOutput,
    MorningIntentOutput,
)
from app.engine.phases import Phase
from app.engine.roles import RoleName
from app.eval.report import EVENT_LABELS
from app.eval.transcript import TranscriptRecorder
from tests.ai.test_turn_structure import _spy, _StagedProvider
from tests.conftest import make_controller

AI_IDS = [f"p{i}" for i in range(1, 17)]


class _Asking:
    """Says `line` and puts `questions` -- (target, structured text) -- in the field."""

    def __init__(self, line: str, questions: list[tuple[str, str]]) -> None:
        self.line = line
        self.questions = questions
        self._metrics = MetricsCollector()

    async def generate_structured(
        self, *, system: str, messages: list[Message], response_schema: type[SchemaT], **kwargs
    ):  # type: ignore[no-untyped-def]
        del system, messages, kwargs
        if response_schema is MorningIntentOutput:
            return MorningIntentOutput()
        output = DiscussionOutput(
            public_message=self.line,
            directed_questions=[
                DirectedQuestion(target_id=target, question=text)
                for target, text in self.questions
            ],
        )
        if response_schema is ChatDiscussionOutput:
            return ChatDiscussionOutput.from_discussion_output(output)
        return output


def _ask(line: str, questions: list[tuple[str, str]], speaker: str = "p1"):  # type: ignore[no-untyped-def]
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    state.day = 2
    provider = _Asking(line, questions)
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        state,
        AI_IDS,
        provider,
        seed=4,
        reasoning=ReasoningRuntime(state, AI_IDS, seed=4),
        model_decides=True,
        recorder=recorder,
    )
    output = asyncio.run(coordinator._speak(controller, state, speaker, "initial_view"))
    return state, recorder, provider._metrics, output


def test_a_question_to_a_seat_the_line_never_names_is_dropped_and_counted():
    state, recorder, metrics, output = _ask(
        "Player3さん、昨日の票の理由は？",
        [("p3", "昨日の投票理由を説明してください"), ("p5", "占い理由も聞きたい")],
    )

    assert output is not None
    assert [q.target_id for q in output.directed_questions] == ["p3"]
    spoken = recorder.transcript.utterances[-1]
    assert spoken.directed_question_targets == ["p3"]
    assert state.pending_questions.get("p5", []) == []
    assert [q.target for q in state.pending_questions["p3"]] == ["p3"]
    assert metrics.events.get("directed_question_target_unnamed") == 1


def test_the_question_a_target_is_shown_is_the_askers_own_sentence():
    state, _recorder, _metrics, _output = _ask(
        "灰を詰めたい。Player3さんは票が浮いてる。Player3さん、昨日の票の理由は？では。",
        [("p3", "昨日の投票理由を説明してください")],
    )

    (question,) = state.pending_questions["p3"]
    assert question.question == "Player3さん、昨日の票の理由は？"


def test_without_a_question_sentence_the_last_one_naming_the_target_is_shown():
    state, _recorder, _metrics, _output = _ask(
        "Player3さんは票が浮いてる。Player3さんの意見を聞かせて。",
        [("p3", "意見を聞かせてください")],
    )

    (question,) = state.pending_questions["p3"]
    assert question.question == "Player3さんの意見を聞かせて"


def test_a_name_inside_a_longer_name_does_not_name_the_target():
    """Player1 is not named by 「Player11」."""
    state, recorder, metrics, _output = _ask(
        "Player11さん、どう思う？", [("p1", "どう思う？")], speaker="p2"
    )

    assert recorder.transcript.utterances[-1].directed_question_targets == []
    assert state.pending_questions.get("p1", []) == []
    assert metrics.events.get("directed_question_target_unnamed") == 1


def test_a_line_written_by_code_carries_no_question():
    """A freemason confirming its partner speaks a code-written line, partner's name and all."""
    controller = make_controller(seed=4)
    state = controller.state
    state.phase = Phase.DISCUSSION
    leader, partner = [p.player_id for p in state.players_by_role(RoleName.FREEMASON)][:2]
    controller.co(leader, RoleName.FREEMASON.value, source_message_id="m1")
    controller.claim_freemason_partner(leader, partner, source_message_id="m1")
    provider = _Asking("よろしく。", [(leader, "共有の相方として何か言うことは？")])
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(state, AI_IDS, provider, seed=4, recorder=recorder)

    output = asyncio.run(coordinator._speak(controller, state, partner, "reaction"))

    assert output is not None
    assert state.players[leader].name in output.public_message
    assert output.directed_questions == []
    assert recorder.transcript.utterances[-1].directed_question_targets == []
    assert provider._metrics.events.get("directed_question_on_code_line") == 1
    assert provider._metrics.events.get("directed_question_target_unnamed") is None


def _summary(line: str) -> DiscussionOutput:
    return DiscussionOutput(
        public_message=line,
        directed_questions=[DirectedQuestion(target_id="p2", question="最後に反論はありますか？")],
        ready_to_vote=True,
    )


def _closing_turns(summary: DiscussionOutput) -> list[tuple[str, str]]:
    provider = _StagedProvider({"consensus_summary": summary})
    controller = make_controller(seed=4)
    controller.state.phase = Phase.DISCUSSION
    controller.co("p1", "freemason")
    coordinator = AICoordinator(controller.state, ["p1", "p2", "p3"], provider, seed=1)
    turns = _spy(coordinator, provider)
    session = SimpleNamespace(controller=controller, discussion_lock=asyncio.Lock())

    asyncio.run(coordinator.run_discussion_round(session))

    return turns[[stage for _seat, stage in turns].index("consensus_summary") :]


def test_a_question_the_line_does_not_name_books_no_reply():
    named = _closing_turns(_summary("仮指定です。Player2さん、最後に反論はありますか？"))
    unnamed = _closing_turns(_summary("仮指定です。最後に反論はありますか？"))

    assert ("p2", "reaction") in named
    assert ("p2", "reaction") not in unnamed


def test_each_dropped_question_has_a_report_label():
    assert EVENT_LABELS["directed_question_target_unnamed"]
    assert EVENT_LABELS["directed_question_on_code_line"]
