"""Public lines the seed-1 review found missing: each one a fact the code
already held and the memo did not say (seed1-review-causes.md, R2 and §2.3).

Every line is pinned at the moment of run #13 seed 1 where its absence showed.
"""

from __future__ import annotations

from tests.ai.reasoning import run13_seed1_boards as s1


def _public(day: int, at: str = "morning", seat: str = s1.REN) -> str:
    return s1.public_part(s1.memo(s1.board(day, at), seat))


def _line(public: str, prefix: str) -> str:
    return next(line for line in public.splitlines() if line.startswith(prefix))


# -- 3. the medium, execution by execution --


def test_after_the_medium_dies_every_execution_reads_unpublished_for_good():
    """4-B1 / 5-B4: アカリ was bitten on night 3. From day 4 nobody alive had
    claimed medium, and the memo said nothing of what that meant."""
    line = _line(_public(4), "- 処刑者の霊媒結果: ")

    assert line == (
        "- 処刑者の霊媒結果: 1日目処刑 ユイ=白(アカリ・2日目公開) / "
        "2日目処刑 アオイ=白(アカリ・3日目公開) / 3日目処刑 ダイキ=未公開。"
        "生存する霊媒COなし: 新たに霊媒COが出ない限り、処刑者の結果は公開されない"
    )


def test_while_a_medium_claimant_lives_the_result_is_only_not_out_yet():
    line = _line(_public(3), "- 処刑者の霊媒結果: ")

    assert line == (
        "- 処刑者の霊媒結果: 1日目処刑 ユイ=白(アカリ・2日目公開) / 2日目処刑 アオイ=未公開"
    )
