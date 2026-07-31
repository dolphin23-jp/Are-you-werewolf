"""CO detection is load-bearing: a false positive injects a phantom role
claim into the CO composition that every other AI then reasons about, so
the negative cases matter as much as the positive ones."""

from __future__ import annotations

import pytest

from app.ai.co_detection import detect_claimed_role
from app.engine.roles import RoleName


@pytest.mark.parametrize(
    "text,expected",
    [
        ("占い師CO。占い師をやっています。", RoleName.SEER),
        ("私は占い師です。", RoleName.SEER),
        ("占い師COします。結果を伝えます。", RoleName.SEER),
        ("占い師 CO", RoleName.SEER),
        ("霊媒師です。COします。", RoleName.MEDIUM),
        ("狩人COします。護衛は任せてください。", RoleName.HUNTER),
        ("共有者です。COします。", RoleName.FREEMASON),
        ("占い師でした。昨夜の結果を話します。", RoleName.SEER),
    ],
)
def test_detects_genuine_self_claims(text: str, expected: RoleName):
    assert detect_claimed_role(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        # Regression: these all matched the original naive pattern.
        "占い師のCOを待ってから動いた方がいいと思います。",
        "占い師は誰ですか。",
        "あなたは占い師ですか？",
        "霊媒師のCOがまだ出ていません。",
        # Ordinary talk about roles that is plainly not a claim.
        "彼が占い師だと思う。",
        "偽の占い師でしょう。",
        "占い師が2人いる状況です。",
        "狩人は誰を護衛したのでしょうか。",
        "共有者はまだCOしないほうがいいですね。",
        "みなさんの発言をもう少し聞いてから判断したいです。",
    ],
)
def test_does_not_fire_on_talk_about_roles(text: str):
    assert detect_claimed_role(text) is None


def test_third_person_report_is_not_a_self_claim():
    others = ["ハルト", "ユイ"]
    assert detect_claimed_role("ハルトは占い師です。", others) is None
    # ...but the speaker claiming it themselves still registers, even when
    # other players are named later in the sentence.
    assert detect_claimed_role("占い師です。ハルトを占いました。", others) is RoleName.SEER


def test_claim_in_a_later_sentence_is_still_found():
    text = "みなさんこんにちは。よろしくお願いします。占い師COします。"
    assert detect_claimed_role(text) is RoleName.SEER
