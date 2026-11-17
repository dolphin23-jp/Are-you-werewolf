"""A question is shown to its target in the asker's own words (seed-1 review
4-A2, prompt side).

The question a seat is asked to answer used to be the asker's structured
paraphrase, presented as "what was said at m8". In the seed-1 game 43 of 44
such paraphrases differed from the message; ハルト's m8 said only that he
wanted to hear ホノカ's answer, and ホノカ was shown 「マコトの相方指名は事実？
共有COするなら今のうちにお願い。」, which nobody had said. The target is now
shown the sentence of the message that names it, as said. (Registering a
question whose message never names its target is the audit side's to drop.)
"""

from __future__ import annotations

from app.ai.context import ContextBuilder, DaySummaryManager
from app.ai.deception import FakeClaimGuard, WolfDeceptionAssignment
from app.ai.personalities import assign_personalities
from app.engine.roles import RoleName
from app.engine.state import PendingQuestion
from tests.ai.seed1_replay import captured_prompts
from tests.conftest import make_controller


def _questions(user: str) -> list[str]:
    layer = user.split("【あなたへの未回答の質問】\n", 1)[1]
    return [line for line in layer.splitlines() if line.startswith("[m")]


def test_honoka_reads_what_each_seat_actually_said_to_her():
    lines = _questions(captured_prompts()["d1-m16-honoka-confirm"].user)

    assert lines[:4] == [
        "[m2] マコト →あなた:「相方はホノカです」",
        "[m6] リク →あなた:「ホノカ、共有の相方か確認させてくれ」",
        "[m7] ミオ →あなた:「まずホノカさん、マコトさんの相方指名は事実ですか」",
        "[m8] ハルト →あなた:「ホノカの返事を先に聞きたいな」",
    ]
    # Two sentences of one message that name her are both quoted.
    assert "[m12] カイト →あなた:「ホノカの返答が全て」「ホノカ、早めに出てきて」" in lines
    user = captured_prompts()["d1-m16-honoka-confirm"].user
    # The paraphrases nobody said are gone.
    assert "共有COするなら今のうちにお願い" not in user
    assert "マコトが相方と名指ししているが" not in user


def _builder(state) -> ContextBuilder:  # type: ignore[no-untyped-def]
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    return ContextBuilder(
        personalities=assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="all_lurk",
            pattern_label="全潜伏",
            fake_role_by_player={},
            lurking_player_ids=wolf_ids,
        ),
        madman_fake_role=None,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
        engine="v3",
    )


def test_a_bare_call_takes_the_sentence_after_it():
    controller = make_controller(seed=4)
    state = controller.state
    state.day = 2
    asked = controller.chat(
        "p2", "おはよう。Player1さん。昨日の票の理由は？ あとで聞くね。", "public"
    )
    state.pending_questions["p1"] = [
        PendingQuestion("p2", "p1", "投票理由を説明してほしい", asked, 2)
    ]

    layer = _builder(state)._layer_pending_questions(state, "p1")

    assert f"[{asked}] Player2 →あなた:「Player1さん。昨日の票の理由は？」" in layer
    assert "投票理由を説明してほしい" not in layer


def test_a_question_whose_message_is_gone_keeps_its_recorded_text():
    state = make_controller(seed=4).state
    state.pending_questions["p1"] = [PendingQuestion("p2", "p1", "理由は？", "m99", 1)]

    layer = _builder(state)._layer_pending_questions(state, "p1")

    assert "[m99] Player2 →あなた:「理由は？」" in layer


def test_the_structured_register_is_unchanged():
    controller = make_controller(seed=4)
    state = controller.state
    asked = controller.chat("p2", "Player1さん、昨日の票の理由は？", "public")
    state.pending_questions["p1"] = [PendingQuestion("p2", "p1", "投票理由", asked, 1)]
    wolf_ids = [p.player_id for p in state.players_by_role(RoleName.WEREWOLF)]
    builder = ContextBuilder(
        personalities=assign_personalities(list(state.players), seed=1),
        day_summaries=DaySummaryManager(),
        wolf_deception=WolfDeceptionAssignment(
            pattern_name="all_lurk",
            pattern_label="全潜伏",
            fake_role_by_player={},
            lurking_player_ids=wolf_ids,
        ),
        madman_fake_role=None,
        fake_claim_guard=FakeClaimGuard(wolf_team_ids=set(wolf_ids)),
    )

    assert "→あなた:「投票理由」" in builder._layer_pending_questions(state, "p1")
