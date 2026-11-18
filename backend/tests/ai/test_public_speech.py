import pytest

from app.ai.public_speech import (
    DetectedPublicResult,
    detect_public_result,
    detect_public_results,
)
from app.engine.roles import RoleName


def test_compact_seer_co_includes_its_white_result():
    result = detect_public_result(
        "占いCOアカリは人狼ではない。理由は初日なので特にない。",
        RoleName.SEER,
        {"p1": "アカリ"},
        role_claimed_in_message=True,
    )

    assert result is not None
    assert (result.result_type, result.target_id, result.is_werewolf) == ("seer", "p1", False)


def test_role_talk_and_speculation_are_not_ability_results():
    candidates = {"p1": "アカリ"}
    assert (
        detect_public_result(
            "アカリは人狼ではないと思う。",
            RoleName.SEER,
            candidates,
            role_claimed_in_message=False,
        )
        is None
    )
    assert (
        detect_public_result(
            "アカリは人狼ではないと思う。",
            None,
            candidates,
            role_claimed_in_message=False,
        )
        is None
    )


def test_another_seers_black_result_is_not_re_registered_as_the_speakers_result():
    result = detect_public_result(
        "ホノカも占いCOしてソウタに黒出しとなると、占い内訳は真狂狼が濃厚。",
        RoleName.SEER,
        {"p4": "ソウタ", "p13": "ホノカ"},
        role_claimed_in_message=False,
    )

    assert result is None


def test_result_with_player_id_and_honorific_is_detected():
    result = detect_public_result(
        "占い師COです。初日占い結果はドルフィン(p0)さん●でした。",
        RoleName.SEER,
        {"p0": "ドルフィン"},
        role_claimed_in_message=True,
    )

    assert result == DetectedPublicResult("seer", "p0", True)


NAMES = {
    "p1": "アカリ",
    "p4": "ソウタ",
    "p8": "カイト",
    "p12": "ダイキ",
    "p15": "アオイ",
}


def _verdicts(text: str, *, role_claimed: bool = False) -> list[tuple[str, bool]]:
    return [
        (result.target_id, result.is_werewolf)
        for result in detect_public_results(
            text, RoleName.SEER, NAMES, role_claimed_in_message=role_claimed
        )
    ]


def test_a_rivals_result_quoted_with_its_owner_is_not_the_speakers_result():
    # Said by the one true seer in a live game: it is a rebuttal about アオイ's white,
    # yet it was recorded as a second result of the speaker's own that night.
    text = (
        "マコト、カイトは私の占いで狼確定だから先に吊り、霊結果で検証できる。"
        "ダイキ、アオイのソウタ白はカイトを占った結果ではない。"
        "私のカイト黒を覆す根拠にはならないよ。"
    )

    assert _verdicts(text) == [("p8", True)]


def test_every_way_of_attributing_a_verdict_to_someone_else_is_skipped():
    assert _verdicts("アオイさんのソウタ白は信用できない。占い結果はカイト黒。") == [("p8", True)]
    assert _verdicts("(p15)のソウタ白は偽だと思う。占い結果はカイト黒です。") == [("p8", True)]
    assert _verdicts("アオイ→ソウタ=白、ダイキ→カイト=黒。占い結果は私のアカリ白。") == [
        ("p1", False)
    ]


def test_a_verdict_under_someone_elses_result_noun_is_not_the_speakers():
    text = "アカリの霊結果ではダイキ黒、ソウタ白。私の占い結果はカイト黒でした。"

    assert _verdicts(text) == [("p8", True)]


def test_a_first_person_marker_hands_the_verdict_back_to_the_speaker():
    text = "アカリの霊結果とは別に、私の占い結果はソウタ白です。"

    assert _verdicts(text) == [("p4", False)]


def test_an_addressee_before_the_verdict_does_not_take_ownership_of_it():
    # "アカリ、" is who is being spoken to, not whose result it is.
    assert _verdicts("占いCO。アカリ、ソウタ白でした。", role_claimed=True) == [("p4", False)]
    assert _verdicts("占い結果はアカリ白、ダイキ黒でした。") == [("p1", False), ("p12", True)]



@pytest.mark.parametrize(
    "text",
    [
        "ソウタは黒っぽい。",
        "ソウタは白っぽいけど保留。",
        "ソウタが黒なら吊る。",
        "ソウタは黒寄り。",
        "ソウタは黒目。",
        "ソウタは人狼側。",
        "ソウタは黒ではない。",
        "ソウタは人狼だったら怖い。",
        "ソウタは人狼だとしたら票が合わない。",
        "ソウタが人狼でしたら吊ります。",
        "ソウタは人狼ではないなら誰が狼か。",
    ],
)
def test_an_impression_or_a_hypothetical_is_not_a_published_result(text: str) -> None:
    """A colour followed by an impression, a side, a negation or a condition is
    talk about the target, not a result. Each of these used to register as a
    binding verdict on the public record."""
    assert (
        detect_public_results(
            text, RoleName.SEER, {"p5": "ソウタ"}, role_claimed_in_message=True
        )
        == []
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ソウタは黒でした。", [("p5", True)]),
        ("ソウタは人狼です。", [("p5", True)]),
        ("ソウタは人狼だ。", [("p5", True)]),
        ("ソウタは人狼ではなかった。", [("p5", False)]),
        ("ソウタ●、ケント○", [("p5", True), ("p6", False)]),
        ("ソウタは黒、ケントは白でした。", [("p5", True), ("p6", False)]),
        ("ソウタを占って、黒。", [("p5", True)]),
    ],
)
def test_a_plain_verdict_is_still_read(text: str, expected: list[tuple[str, bool]]) -> None:
    found = detect_public_results(
        text, RoleName.SEER, {"p5": "ソウタ", "p6": "ケント"}, role_claimed_in_message=True
    )

    assert [(item.target_id, item.is_werewolf) for item in found] == expected
