"""Each AI seat's skill is in the record, so a reviewer can read who saw what when.

The skill (熟練/普通/初心者) sets when a seat is shown a derived conclusion --
a fake claimant's contradiction, an endgame consequence. Without it in the
record, a novice not mentioning a 破綻 could not be told from a seat that
was shown it and ignored it.
"""

from __future__ import annotations

from app.ai.coordinator import AICoordinator
from app.ai.provider.mock import MockProvider
from app.ai.reasoning.runtime import ReasoningRuntime
from app.eval.report import render_transcript
from app.eval.transcript import GameTranscript, TranscriptRecorder
from tests.conftest import make_controller


def test_the_coordinator_records_each_ai_seats_skill_and_the_record_prints_it() -> None:
    controller = make_controller(seed=4)
    state = controller.state
    ai_ids = [pid for pid in state.players if pid != "p0"]
    reasoning = ReasoningRuntime(state, ai_ids, seed=4)
    recorder = TranscriptRecorder()
    AICoordinator(
        state, ai_ids, MockProvider(seed=1), seed=4, reasoning=reasoning, recorder=recorder
    )

    skills = recorder.transcript.skills
    assert skills == {pid: skill.value for pid, skill in reasoning.skills.items()}
    assert set(skills) == set(ai_ids)

    restored = GameTranscript.from_dict(recorder.transcript.to_dict())
    assert restored.skills == skills
    page = render_transcript(restored)
    assert "| 人格 | 腕前 | 騙り役 |" in page
    labels = {"expert": "熟練", "average": "普通", "novice": "初心者"}
    pid = ai_ids[0]
    assert f"| {labels[skills[pid]]} |" in page.split(f"`{pid}`", 1)[1].splitlines()[0]


def test_an_older_record_without_skills_prints_a_dash() -> None:
    data = GameTranscript(names={"p1": "ハルト"}, roles={"p1": "seer"}).to_dict()
    data.pop("skills")

    page = render_transcript(GameTranscript.from_dict(data))

    assert "| ハルト (`p1`) | 占い師 |" in page
    row = next(line for line in page.splitlines() if line.startswith("| ハルト"))
    assert row.count("| - |") >= 1
