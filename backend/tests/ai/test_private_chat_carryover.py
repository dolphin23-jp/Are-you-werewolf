"""Last night's team chat, carried into the next day for the seats that wrote
it; the freemasons' chat given a job (owner decisions on the seed-1 review
3-N1, 2-N1, 1-B5).

Wolves planned at night ("今夜ケント、通らなければ次はアカリ") and the next
day's prompts never showed it; the freemasons' chat ran on the wolves'
instruction ("今夜の行動と明日の方針"), so two seats with no night action spent
it on the hunter's guard. Now the freemasons' chat is where the next day's
progression is decided, and both teams read the previous night's chat in the
next day's discussion and vote prompts -- information only, with the line
that the team's words are not repeated at the table.
"""

from __future__ import annotations

from tests.ai.seed1_replay import captured_prompts

WOLF_HEADING = "【前夜の人狼チャット】内輪の内容は卓でそのまま言わない。"
MASON_HEADING = "【前夜の共有チャット】内輪の内容は卓でそのまま言わない。"
# Night 1, from the record.
WOLF_LINE = "ハルト: カイト、ダイキ、今夜は真占い候補を探したい。"
MASON_LINE = "マコト: ホノカさん、明日は霊媒の動きを見たいです。"


def test_the_freemason_chat_decides_the_next_days_progression():
    user = captured_prompts()["n1-mason-makoto"].user
    instruction = next(line for line in user.splitlines() if line.startswith("【指示】"))

    assert instruction.startswith("【指示】共有チャットの1通。相方と明日の進行を決める")
    for item in ("グレランか指定か", "候補", "占いCO者の扱い", "まとめ役"):
        assert item in instruction
    assert "狩人はこのチャットを読めない" in instruction
    assert "今夜の行動" not in instruction


def test_the_wolf_chat_keeps_its_instruction():
    user = captured_prompts()["n1-wolf-kaito"].user

    assert "【指示】内輪チャットの1通。仲間と、今夜の行動と明日の方針を" in user


def test_wolves_read_last_nights_chat_when_they_speak_and_vote():
    for tag in ("d2-m33-daiki-immediate", "d2-vote-haruto"):
        user = captured_prompts()[tag].user
        section = user.split(WOLF_HEADING, 1)[1].split("\n\n", 1)[0]
        assert WOLF_LINE in section, tag
        assert MASON_HEADING not in user, tag


def test_freemasons_read_last_nights_chat_when_they_speak_and_vote():
    for tag in ("d2-m35-makoto-initial", "d2-vote-makoto"):
        user = captured_prompts()[tag].user
        section = user.split(MASON_HEADING, 1)[1].split("\n\n", 1)[0]
        assert MASON_LINE in section, tag
        assert WOLF_HEADING not in user, tag


def test_no_other_seat_reads_either_chat():
    for tag in ("d2-vote-kento", "d2-m31-akari-immediate", "d2-m34-aoi-immediate"):
        prompt = captured_prompts()[tag].text
        assert "前夜の人狼チャット" not in prompt, tag
        assert "前夜の共有チャット" not in prompt, tag
        assert "今夜は真占い候補を探したい" not in prompt, tag
        assert "明日は霊媒の動きを見たいです" not in prompt, tag


def test_only_the_previous_night_is_carried():
    user = captured_prompts()["d4-vote-kaito"].user
    section = user.split(WOLF_HEADING, 1)[1].split("\n\n", 1)[0]

    # Night 3's chat, not night 1's.
    assert "今夜は慎重に" in section
    assert "今夜は真占い候補を探したい" not in section
