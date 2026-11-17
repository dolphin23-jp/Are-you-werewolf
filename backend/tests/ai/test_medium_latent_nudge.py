"""A medium may stay latent on day 1; it is given the fact and a nudge, and the
facilitator doctrine waits for its CO (owner decision on the seed-1 review
1-B3, 2-B2).

アカリ, the medium, had no CO duty on day 1 and nothing told her she had not
claimed; meanwhile the facilitator doctrine, selected by role, had her propose
the day's policy as if she had. The v3 medium now reads, while unclaimed,
「あなたはまだ霊媒COしていない。17Aでは1日目COが普通」, and the facilitator
doctrine reaches a medium or freemason only once the seat has claimed.
"""

from __future__ import annotations

from app.ai.knowledge_base import KnowledgeBase, KnowledgeContext
from app.engine.roles import RoleName
from tests.ai.seed1_replay import captured_prompts
from tests.conftest import make_controller

NUDGE = "あなたはまだ霊媒COしていない。17Aでは1日目COが普通"
LEADER = "【進行役】"


def test_an_unclaimed_medium_is_told_so_and_gets_no_facilitator_doctrine():
    for tag in ("d1-m5-akari-initial", "d2-m31-akari-immediate"):
        prompt = captured_prompts()[tag]
        assert NUDGE in prompt.system, tag
        assert LEADER not in prompt.text, tag


def test_a_claimed_medium_gets_the_doctrine_and_no_nudge():
    prompt = captured_prompts()["d3-m45-akari-immediate"]

    assert NUDGE not in prompt.text
    assert LEADER in prompt.user


def test_a_claimed_freemason_still_gets_the_doctrine():
    assert LEADER in captured_prompts()["d2-m35-makoto-initial"].user


def test_no_other_role_is_nudged():
    for tag, prompt in captured_prompts().items():
        if "akari" not in tag:
            assert NUDGE not in prompt.text, tag


def test_legacy_and_v2_select_the_facilitator_doctrine_as_before():
    state = make_controller(seed=4).state
    medium = state.players_by_role(RoleName.MEDIUM)[0].player_id
    base = KnowledgeBase()
    for engine in ("legacy", "v2"):
        ids = {
            doctrine.metadata["id"]
            for doctrine in base.select(
                KnowledgeContext(
                    state=state,
                    player_id=medium,
                    fake_role=None,
                    perspective_needed=False,
                    engine=engine,
                )
            )
        }
        assert any(doctrine_id.startswith("leader") for doctrine_id in ids), engine
