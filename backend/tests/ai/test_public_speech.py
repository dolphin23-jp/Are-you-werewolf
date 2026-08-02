from app.ai.public_speech import detect_public_result
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
