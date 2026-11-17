"""The v3 table's vocabulary, defined once and as the rules have it (seed-1
review R6 definitions: 1-B1, 1-B2, 2-B5, 4-B1, 2-B1).

Definitions only -- no new advice. The register named CO counts with 2-1 and
3-1 but never 3-0, the count the table actually had on day 1 (霊媒 0); defined
グレー twice ("無CO" and "無CO・未判定"); described a day as both a グレラン and
a 仮指定→本指定; defined GJ/平和 as a guard success alone, though an attack
on the fox also leaves no body; and said the medium's result verifies a seer
without saying the medium has to be alive.
"""

from __future__ import annotations

from app.ai.knowledge_base import KnowledgeBase
from tests.ai.seed1_replay import captured_prompts


def _register() -> str:
    return next(
        doctrine.body
        for doctrine in KnowledgeBase().doctrines
        if doctrine.metadata["id"] == "jinro-17a-chat-register"
    )


def test_co_counts_are_named_with_a_three_zero_example():
    assert "霊媒COがいなければ3-0" in _register()


def test_grey_has_one_definition():
    body = _register()

    assert body.count("グレー(無CO・未判定)") == 1
    assert "グレー(無CO)" not in body


def test_a_grey_vote_and_a_designation_are_two_different_days():
    body = _register()

    assert "グレラン(進行役は吊り先を指名せず、グレー全員が投票の候補)" in body
    assert "指定(進行役が吊り先を1人指名する。仮指定→弁明→本指定)" in body
    assert "その日をグレランにするか指定にするかを決める" in body
    assert "「仮指定→本指定」で吊り先をまとめることが多い" not in body


def test_a_quiet_morning_may_be_the_fox():
    body = _register()

    assert "GJ(護衛成功)" in body
    assert "平和(死体のない朝。GJか、襲撃先が妖狐だったとき)" in body
    assert "GJ・平和(護衛成功で死体なし)" not in body


def test_a_medium_verifies_only_while_alive():
    assert "霊媒が生きていれば、黒判定を受けて吊られた人の霊結果で占いの真偽が検証される" in (
        _register()
    )


def test_the_facilitator_doctrine_reads_designation_as_one_way_to_run_a_day():
    body = next(
        doctrine.body
        for doctrine in KnowledgeBase().doctrines
        if doctrine.metadata["id"] == "freemason-facilitator"
    )

    assert "指定の日は、仮指定に理由を添え" in body


def test_the_replayed_prompts_carry_the_definitions():
    user = captured_prompts()["d1-m14-souta-initial"].user

    assert "霊媒COがいなければ3-0" in user
    assert "平和(死体のない朝。GJか、襲撃先が妖狐だったとき)" in user
