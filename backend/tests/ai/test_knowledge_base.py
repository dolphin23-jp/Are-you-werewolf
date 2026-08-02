from pathlib import Path

import pytest

from app.ai.knowledge_base import KnowledgeBase, KnowledgeContext, parse_doctrine
from app.engine.roles import RoleName
from app.engine.state import CoDeclaration
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
    state.co_declarations.extend(
        [
            CoDeclaration(wolf.player_id, RoleName.SEER, 1),
            CoDeclaration(other.player_id, RoleName.SEER, 1),
        ]
    )

    selected = KnowledgeBase(tmp_path).select(
        KnowledgeContext(state, wolf.player_id, fake_role=RoleName.SEER)
    )

    assert [item.metadata["id"] for item in selected] == ["high", "low"]


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
