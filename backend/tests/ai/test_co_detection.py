"""CO detection is load-bearing: a false positive injects a phantom role
claim into the CO composition that every other AI then reasons about, so
the negative cases matter as much as the positive ones."""

from __future__ import annotations

import pytest

from app.ai.co_detection import detect_claimed_role, detect_freemason_partner
from app.engine.roles import RoleName


@pytest.mark.parametrize(
    "text,expected",
    [
        ("占い師CO。占い師をやっています。", RoleName.SEER),
        ("私は占い師です。", RoleName.SEER),
        ("占い師COします。結果を伝えます。", RoleName.SEER),
        ("占い師 CO", RoleName.SEER),
        ("占いCO、初日はユイ白。", RoleName.SEER),
        ("霊媒師です。COします。", RoleName.MEDIUM),
        ("霊媒CO。昨日の処刑結果を伝えます。", RoleName.MEDIUM),
        ("狩人COします。護衛は任せてください。", RoleName.HUNTER),
        ("共有者です。COします。", RoleName.FREEMASON),
        ("共有CO、相方はリクです。", RoleName.FREEMASON),
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
        "占いCOの有無を確認します。",
        "占いCOを待ってから判断します。",
        "占いCOは3人です。",
        "霊媒COの人数を確認します。",
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


def test_shared_partner_confirmation_is_a_self_claim_despite_leading_name():
    text = "ユイ(p3)の共有者CO、相方は私ツムギ(p11)で間違いありません。"
    assert detect_claimed_role(text, ["ユイ", "ツムギ"]) is RoleName.FREEMASON


def test_extracts_partner_from_reveal_and_confirmation_forms():
    candidates = {"p3": "ユイ", "p11": "ツムギ"}
    assert detect_freemason_partner("共有者CO、相方はツムギ(p11)です。", candidates) == "p11"
    assert (
        detect_freemason_partner(
            "ユイ(p3)の共有者CO、相方は私ツムギ(p11)で間違いありません。",
            candidates,
        )
        == "p3"
    )
    assert detect_freemason_partner("共有は相方生存のようですね。", candidates) is None


def test_partner_id_matching_does_not_confuse_p1_with_p11():
    candidates = {"p1": "アカリ", "p11": "ツムギ"}
    assert detect_freemason_partner("共有CO、相方はツムギ(p11)です。", candidates) == "p11"


def test_partner_parser_stops_before_later_execution_target():
    candidates = {"p0": "ドルフィン", "p8": "カイト"}
    text = "共有者CO。相方はカイト(p8)。ドルフィン(p0)処刑で霊視します。"
    assert detect_freemason_partner(text, candidates) == "p8"


# -- talk about claims is not a claim (run 16, X22) --
#
# Run 16, day 1, m9: カイト (a wolf told to lurk) acknowledged the two freemason
# claims, and the old reader registered a third freemason CO for him; the
# table spent the game on it. The survey of every committed record found the
# same reader registering COs from the other lines below.

TABLE = ["マコト", "ホノカ", "カイト", "アオイ", "リク", "ケント", "Player9", "Player13"]


@pytest.mark.parametrize(
    "text",
    [
        # run 16 m9, and the replies the phantom claim drew
        "共有CO二人は了解。占いは2-1だから、今日はグレランで灰の中身を見たい。"
        "薄い発言から順に拾っていくよ。",
        "共有COが3人出ているので、カイトさんの真偽が気になります。まず本人の見解を伺いたいです。",
        "共有は3人。マコトとホノカの組が互いに確定なら、カイトは偽だと思う。",
        # run 15 m103 (ホノカ, a real freemason already claimed: a reason, not a claim)
        "アオイ、白ありがとう！共有だから私は白で合ってるよ。今日はリク吊りで霊結果を見よう！",
        # run 13 seed 2 m32 (ケント, saying when he had decided to claim)
        "リクの質問ね！自分は1日目の昼に、結果が出る前から霊媒COするって決めてたよ。",
        # the v2 records (live-ab): a counter-claim talked about, a time after a claim
        "ただ、共有CO対抗だったPlayer13(p13)が、Player3(p3)への投票を避けている。",
        "3日目も、p8のp9黒だけでp14さんを白置きせず、霊媒CO後の投票姿勢を継続して疑いました。",
        "私が先に共有COし、Player13(p13)が確認したため、Player13(p13)が相方です。",
        # acknowledgements, counts, requests and questions in other words
        "共有CO了解。",
        "共有CO、了解です。",
        "共有CO、二人とも確認しました。",
        "占いCO2人か。",
        "占いCO二人の真偽を見たい。",
        "二人の共有COは信用する。",
        "あいつは共有COだ。",
        "あっちが占いCOだ。",
        "占い師COしてください。",
        "共有COするなら今のうちにお願い。",
        "占い師COで決まりかな。",
        "占い師CO？",
        "共有CO相手は誰？",
    ],
)
def test_talk_about_claims_registers_nothing(text: str):
    assert detect_claimed_role(text, TABLE) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # the declared CO sentences of the committed Claude records
        ("…占い師COです。0日目夜はハルトさんを占って白でした。", RoleName.SEER),
        ("遅れてすみません、占い師COします。", RoleName.SEER),
        ("対抗で占い師CO。", RoleName.SEER),
        ("ふあ〜、霊媒師COするね〜", RoleName.MEDIUM),
        ("霊媒師CO〜", RoleName.MEDIUM),
        ("占いCO継続。", RoleName.SEER),
        ("占い師COだ。", RoleName.SEER),
        ("対抗でCOする、占い師だ。", RoleName.SEER),
        ("霊媒師だ、COしておく。", RoleName.MEDIUM),
        ("その黒は偽。対抗で占いCOする。", RoleName.SEER),
        ("今日は占い師COします。", RoleName.SEER),
        ("実は占い師です。", RoleName.SEER),
        ("やっぱり霊媒師です。", RoleName.MEDIUM),
        # the v2 records: a CO the structured field left out, said plainly
        ("霊媒CO、まだ結果はありません。", RoleName.MEDIUM),
        ("占い師CO、Player9(p9)は人狼でした。", RoleName.SEER),
        # the speaker as the subject, after a seat addressed by name
        ("ホノカ、俺は本物の共有だ。マコトとホノカの組が確定とは限らない。", RoleName.FREEMASON),
        ("ケントさんの占いCO了解、私も占い師CO。", RoleName.SEER),
        # the canonical slide sentence (`claims.render_switch_sentence`)
        ("共有COを取り下げ、占いCOに変更します。", RoleName.SEER),
    ],
)
def test_a_declaration_in_its_own_words_still_registers(text: str, expected: RoleName):
    assert detect_claimed_role(text, TABLE) is expected
