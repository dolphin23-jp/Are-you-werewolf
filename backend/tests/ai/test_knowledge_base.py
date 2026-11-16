import re
from pathlib import Path

import pytest

from app.ai.knowledge_base import KnowledgeBase, KnowledgeContext, parse_doctrine
from app.engine.roles import RoleName
from app.eval.analyzers import _TEMPLATE_PHRASE_RE
from tests.conftest import make_controller


def test_flat_front_matter_parser_strips_comments():
    doctrine = parse_doctrine(
        """---
id: example
priority: 80 # comment
factions: werewolf, madman
---
本文です。
"""
    )

    assert doctrine.metadata["priority"] == "80"
    assert doctrine.body == "本文です。"


def test_parser_rejects_unclosed_front_matter():
    with pytest.raises(ValueError, match="not closed"):
        parse_doctrine("---\nid: broken\n本文")


def test_conditions_and_priority_are_deterministic(tmp_path: Path):
    (tmp_path / "low.md").write_text("---\nid: low\npriority: 1\n---\nlow", encoding="utf-8")
    (tmp_path / "high.md").write_text(
        "---\nid: high\npriority: 90\nclaimed_roles: seer\nfake_only: true\n"
        "factions: werewolf, madman\nmin_day: 1\nmin_co_count: seer>=2\n---\nhigh",
        encoding="utf-8",
    )
    controller = make_controller(seed=4)
    state = controller.state
    wolf = state.players_by_role(RoleName.WEREWOLF)[0]
    other = next(player for player in state.players.values() if player.player_id != wolf.player_id)
    state.day = 1
    controller.co(wolf.player_id, RoleName.SEER.value)
    controller.co(other.player_id, RoleName.SEER.value)

    selected = KnowledgeBase(tmp_path).select(
        KnowledgeContext(state, wolf.player_id, fake_role=RoleName.SEER)
    )

    assert [item.metadata["id"] for item in selected] == ["high", "low"]


def test_a_doctrine_can_name_the_engines_it_is_written_for(tmp_path: Path):
    (tmp_path / "chat.md").write_text(
        "---\nid: chat\nengines: v3\n---\nchat register", encoding="utf-8"
    )
    (tmp_path / "all.md").write_text("---\nid: all\n---\nfor everyone", encoding="utf-8")
    state = make_controller(seed=4).state
    knowledge = KnowledgeBase(tmp_path)

    legacy = [d.metadata["id"] for d in knowledge.select(KnowledgeContext(state, "p1"))]
    v3 = [d.metadata["id"] for d in knowledge.select(KnowledgeContext(state, "p1", engine="v3"))]

    assert legacy == ["all"]
    assert v3 == ["all", "chat"]


def test_the_shipped_chat_doctrine_is_v3_only():
    state = make_controller(seed=4).state
    knowledge = KnowledgeBase()
    v2_context = KnowledgeContext(state, "p1", engine="v2")
    v3_context = KnowledgeContext(state, "p1", engine="v3")
    ids_v2 = {d.metadata["id"] for d in knowledge.select(v2_context)}
    ids_v3 = {d.metadata["id"] for d in knowledge.select(v3_context)}

    assert "jinro-17a-chat-register" not in ids_v2
    assert "jinro-17a-chat-register" in ids_v3


def _v3_reachable(doctrine) -> bool:  # type: ignore[no-untyped-def]
    engines = {e.strip() for e in doctrine.metadata.get("engines", "").split(",") if e.strip()}
    return not engines or "v3" in engines


def test_doctrine_v3_can_read_never_hands_it_template_wording():
    # Doctrine text is the model's register: a phrase in it comes back as speech.
    # The evaluator counts these phrases as code-written lines, and "独立(した)根拠"
    # is the v2 brief's own wording -- equivalent meaning is said without it.
    shipped = [d for d in KnowledgeBase().doctrines if _v3_reachable(d)]

    assert {d.metadata["id"] for d in shipped} >= {"village-gray", "freemason-facilitator"}
    for doctrine in shipped:
        body = doctrine.body
        assert "独立" not in body, doctrine.metadata["id"]
        assert not _TEMPLATE_PHRASE_RE.search(body), doctrine.metadata["id"]
        assert not re.search(r"思考|thought|reasoning", body, re.IGNORECASE)


def test_false_fake_only_selects_only_non_fakers(tmp_path: Path):
    (tmp_path / "lurker.md").write_text(
        "---\nid: lurker\nplayer_roles: werewolf\nfake_only: false\n---\nlurk",
        encoding="utf-8",
    )
    controller = make_controller(seed=4)
    state = controller.state
    wolf = state.players_by_role(RoleName.WEREWOLF)[0]
    knowledge = KnowledgeBase(tmp_path)

    assert knowledge.select(KnowledgeContext(state, wolf.player_id))
    assert not knowledge.select(KnowledgeContext(state, wolf.player_id, fake_role=RoleName.SEER))


def test_madman_can_match_role_named_faction(tmp_path: Path):
    (tmp_path / "madman.md").write_text(
        "---\nid: madman\nfactions: madman\n---\nconfuse",
        encoding="utf-8",
    )
    controller = make_controller(seed=4)
    state = controller.state
    madman = state.players_by_role(RoleName.MADMAN)[0]

    assert KnowledgeBase(tmp_path).select(KnowledgeContext(state, madman.player_id))
