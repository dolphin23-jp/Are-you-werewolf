"""`app/eval/judgment_quiz.py`: building a fair blind quiz and scoring it honestly."""

from __future__ import annotations

import math
import time
from collections import Counter

import pytest

from app.eval.judgment_quiz import (
    ACCURACY_THRESHOLD,
    AI_LABEL,
    HUMAN_LABEL,
    LENGTH_BIN_EDGES,
    QuizInputError,
    Rating,
    SourceLine,
    ai_lines_from_transcript,
    ai_texts_from_key,
    anonymize,
    answer_key,
    answers_from_key,
    binomial_cdf,
    build_quiz,
    clopper_pearson_upper,
    human_lines_from_text,
    judgments_needed,
    judgments_needed_exact,
    length_bin,
    length_bin_label,
    names_from_text,
    normalize_guess,
    parse_ratings_csv,
    ratings_template,
    render_score,
    render_sheet,
    score,
)
from app.eval.transcript import GameTranscript, Utterance

NAMES = {"p0": "マコト", "p1": "アカリ", "p2": "ユイ", "p3": "ユイカ"}


def _utterance(text: str, *, kind: str = "discussion", fallback: bool = False) -> Utterance:
    return Utterance(
        day=1,
        phase="discussion",
        kind=kind,
        player_id="p0",
        player_name="マコト",
        role="villager",
        team="village",
        personality="calm",
        deception_role=None,
        text=text,
        used_fallback=fallback,
    )


def _transcript_dict(*utterances: Utterance) -> dict:
    return GameTranscript(
        game_id="g", seed=1, names=dict(NAMES), utterances=list(utterances)
    ).to_dict()


# --- sources ---------------------------------------------------------------


def test_ai_lines_are_the_public_discussion_lines_of_a_transcript():
    data = _transcript_dict(
        _utterance("アカリが怪しいと思う。"),
        _utterance("p3を噛もう", kind="wolf_chat"),
        _utterance("", kind="discussion"),
        _utterance("様子見です。", fallback=True),
        _utterance("ユイに投票", kind="vote"),
    )
    lines = ai_lines_from_transcript(data, "transcript-seed1.json")
    assert [line.text for line in lines] == ["アカリが怪しいと思う。"]
    (line,) = lines
    assert line.answer == AI_LABEL
    assert line.origin == "transcript-seed1.json#0"
    assert set(line.names) == set(NAMES.values())
    with_fallback = ai_lines_from_transcript(data, "t.json", include_fallback=True)
    assert [line.text for line in with_fallback] == ["アカリが怪しいと思う。", "様子見です。"]


def test_a_file_that_is_not_a_transcript_is_refused():
    with pytest.raises(QuizInputError, match="utterances"):
        ai_lines_from_transcript({"games": []}, "summary.json")


def test_human_lines_as_plain_text_skip_blanks_and_comments():
    content = "# 出典: 自分たちの村\n\nおはようございます\n  占いCOします  \n"
    lines = human_lines_from_text(content, "human.txt")
    assert [line.text for line in lines] == ["おはようございます", "占いCOします"]
    assert [line.origin for line in lines] == ["human.txt:3", "human.txt:4"]
    assert all(line.answer == HUMAN_LABEL and line.names == () for line in lines)


def test_human_lines_as_jsonl_carry_the_names_to_mask():
    content = (
        '{"text": "タロウさん白です", "speaker": "ハナコ", "names": ["タロウ"]}\n'
        '{"text": "了解", "extra": 1}\n'
    )
    first, second = human_lines_from_text(content, "human.jsonl")
    assert first.text == "タロウさん白です"
    assert set(first.names) == {"ハナコ", "タロウ"}
    assert second.names == ()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"text": "ok"}\n{"text": \n', "human.jsonl:2: not valid JSON"),
        ('{"speaker": "x"}\n', "string 'text'"),
        ('{"text": "ok", "names": "タロウ"}\n', "list of strings"),
    ],
)
def test_bad_jsonl_is_refused_with_its_line_number(content: str, message: str):
    with pytest.raises(QuizInputError, match=message):
        human_lines_from_text(content, "human.jsonl")


def test_a_mask_list_is_one_name_per_line():
    assert names_from_text("# 名簿\nタロウ\n\n ハナコ \n") == ("タロウ", "ハナコ")


# --- anonymization ---------------------------------------------------------


def test_names_become_placeholders_lettered_by_first_appearance():
    text = anonymize("アカリとマコトが対立。マコトは白、アカリは黒", ["マコト", "アカリ"])
    assert text == "〈A〉と〈B〉が対立。〈B〉は白、〈A〉は黒"


def test_a_longer_name_is_not_eaten_by_a_shorter_one():
    assert anonymize("ユイカとユイ", ["ユイ", "ユイカ"]) == "〈A〉と〈B〉"


def test_ids_and_reply_anchors_are_removed():
    text = anonymize("マコト(p3)の[m12 9:01]発言、p4 と m7 に >>12 ＞＞１２ @マコト", ["マコト"])
    assert text == "〈A〉の発言、 と に 〈A〉"
    for leftover in ("p3", "p4", "m7", "m12", ">>", "＞＞", "@"):
        assert leftover not in text


def test_anonymization_collapses_whitespace_and_ignores_name_order():
    assert anonymize("  ユイ\nと\r\nアカリ  ", ["アカリ", "ユイ"]) == "〈A〉 と 〈B〉"
    assert anonymize("ユイとアカリ", ["ユイ", "アカリ"]) == anonymize(
        "ユイとアカリ", ["アカリ", "ユイ"]
    )


def test_length_bins_cover_every_length():
    assert length_bin(0) == 0
    assert length_bin(LENGTH_BIN_EDGES[0] - 1) == 0
    assert length_bin(LENGTH_BIN_EDGES[0]) == 1
    assert length_bin(10_000) == len(LENGTH_BIN_EDGES)
    assert length_bin_label(0) == f"0〜{LENGTH_BIN_EDGES[0] - 1}字"
    assert length_bin_label(len(LENGTH_BIN_EDGES)).endswith("字以上")


# --- building --------------------------------------------------------------


def _ai(texts: list[str]) -> list[SourceLine]:
    return [
        SourceLine(AI_LABEL, t, f"t.json#{i}", tuple(NAMES.values())) for i, t in enumerate(texts)
    ]


def _human(texts: list[str]) -> list[SourceLine]:
    return [SourceLine(HUMAN_LABEL, t, f"h.txt:{i}", ("タロウ",)) for i, t in enumerate(texts)]


def _lines(prefix: str, count: int, length: int) -> list[str]:
    return [(f"{prefix}{i}" + "あ" * length)[:length] for i in range(count)]


def test_the_quiz_is_half_ai_half_human_and_length_matched():
    ai = _ai(_lines("AI短", 30, 15) + _lines("AI中", 30, 50) + _lines("AI長", 30, 150))
    human = _human(_lines("人短", 5, 15) + _lines("人中", 40, 50) + _lines("人長", 2, 150))
    quiz = build_quiz(ai, human, items=40, seed=1)

    answers = Counter(item.answer for item in quiz.items)
    assert answers == {AI_LABEL: 20, HUMAN_LABEL: 20}
    for b in {item.length_bin for item in quiz.items}:
        in_bin = Counter(item.answer for item in quiz.items if item.length_bin == b)
        assert in_bin[AI_LABEL] == in_bin[HUMAN_LABEL], b
    # A bin with only two human lines can never contribute more than two pairs.
    long_bin = length_bin(150)
    assert sum(1 for item in quiz.items if item.length_bin == long_bin) <= 4
    assert [item.item_id for item in quiz.items] == [f"Q{i:02d}" for i in range(1, 41)]
    assert {row["bin"]: row["pairs"] for row in quiz.bins}[long_bin] <= 2


def test_the_quiz_is_deterministic_by_seed_and_shuffled():
    ai = _ai(_lines("AI", 40, 30))
    human = _human(_lines("人", 40, 30))
    first = build_quiz(ai, human, items=20, seed=5)
    again = build_quiz(ai, human, items=20, seed=5)
    other = build_quiz(ai, human, items=20, seed=6)
    assert [i.text for i in first.items] == [i.text for i in again.items]
    assert [i.text for i in first.items] != [i.text for i in other.items]
    # Not all of one kind first: the order gives nothing away.
    assert {i.answer for i in first.items[:10]} == {AI_LABEL, HUMAN_LABEL}


def test_the_quiz_shrinks_to_what_can_be_matched_and_drops_repeats():
    ai = _ai(["同じ発言です。"] * 5 + ["別の発言です。"])
    human = _human(["人間の発言です。", "もう一つの発言。", "三つ目の発言。"])
    quiz = build_quiz(ai, human, items=41, seed=1)
    # Two distinct AI lines after de-duplication -> two pairs at most.
    assert len(quiz.items) == 4
    assert quiz.ai_pool == 2 and quiz.human_pool == 3
    assert quiz.requested_items == 41


def test_too_small_a_quiz_is_refused():
    with pytest.raises(QuizInputError):
        build_quiz(_ai(["a"]), _human(["b"]), items=1, seed=1)


def test_names_never_reach_the_rater():
    ai = _ai(["マコトとユイカが怪しい。", "アカリは白。"])
    human = _human(["タロウが怪しい。", "タロウ白です"])
    quiz = build_quiz(ai, human, items=4, seed=1)
    sheet = render_sheet(quiz)
    for name in (*NAMES.values(), "タロウ"):
        assert name not in sheet
    assert "〈A〉" in sheet
    # The sheet shows the lines and the instructions, never the answers.
    for item in quiz.items:
        assert item.text in sheet and item.origin not in sheet
    assert "t.json" not in sheet and "h.txt" not in sheet


def test_answer_key_round_trip_and_template():
    quiz = build_quiz(_ai(_lines("AI", 6, 30)), _human(_lines("人", 6, 30)), items=6, seed=1)
    key = answer_key(quiz)
    answers = answers_from_key(key)
    assert answers == {item.item_id: item.answer for item in quiz.items}
    ai_texts = ai_texts_from_key(key)
    assert set(ai_texts) == {i.item_id for i in quiz.items if i.answer == AI_LABEL}
    assert key["items"][0]["original_text"]
    template = ratings_template(quiz, rater="r1").splitlines()
    assert template[0] == "rater,item,guess"
    assert template[1:] == [f"r1,{item.item_id}," for item in quiz.items]


@pytest.mark.parametrize(
    "key",
    [
        {"items": []},
        {"format_version": 1, "items": [{"item_id": "Q01", "answer": "robot"}]},
        {"format_version": 1, "items": "Q01"},
    ],
)
def test_a_bad_answer_key_is_refused(key: dict):
    with pytest.raises(QuizInputError):
        answers_from_key(key)


# --- ratings ---------------------------------------------------------------

ANSWERS = {"Q01": AI_LABEL, "Q02": HUMAN_LABEL, "Q03": AI_LABEL, "Q04": HUMAN_LABEL}


def test_guesses_accept_the_obvious_spellings_only():
    assert normalize_guess(" AI ") == AI_LABEL
    assert normalize_guess("ai") == AI_LABEL
    assert normalize_guess("ＡＩ") == AI_LABEL
    assert normalize_guess("人間") == HUMAN_LABEL
    assert normalize_guess("Human") == HUMAN_LABEL
    assert normalize_guess("  ") is None
    with pytest.raises(QuizInputError):
        normalize_guess("たぶんAI")


def test_ratings_parse_with_bom_and_blank_template_rows():
    content = "﻿rater,item,guess\nr1,Q01,AI\nr1,Q02,人間\nr1,Q03,\n"
    ratings, blank = parse_ratings_csv(content, ANSWERS)
    assert ratings == [Rating("r1", "Q01", AI_LABEL), Rating("r1", "Q02", HUMAN_LABEL)]
    assert blank == 1


def test_every_ratings_problem_is_reported_at_once_with_its_line():
    content = (
        "rater,item,guess\n"
        "r1,Q01,AI\n"
        "r1,Q01,人間\n"  # line 3: duplicate
        "r1,Q99,AI\n"  # line 4: unknown item
        ",Q02,AI\n"  # line 5: no rater
        "r2,Q02,ロボット\n"  # line 6: bad guess
    )
    with pytest.raises(QuizInputError) as info:
        parse_ratings_csv(content, ANSWERS)
    message = str(info.value)
    for fragment in ("line 3", "twice", "line 4", "Q99", "line 5", "line 6"):
        assert fragment in message


def test_a_ratings_file_without_the_header_is_refused():
    with pytest.raises(QuizInputError, match="header"):
        parse_ratings_csv("r1,Q01,AI\n", ANSWERS)


# --- statistics ------------------------------------------------------------


def test_binomial_cdf_matches_the_direct_sum():
    for n, k, p in [(10, 3, 0.4), (50, 20, 0.55), (5, 0, 0.1), (300, 170, 0.6)]:
        direct = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
        assert binomial_cdf(k, n, p) == pytest.approx(direct, rel=1e-9)
    assert binomial_cdf(-1, 10, 0.5) == 0.0
    assert binomial_cdf(10, 10, 0.5) == 1.0


def test_clopper_pearson_upper_bound_known_values():
    # No successes: the bound solves (1 - p)^n = 0.05.
    assert clopper_pearson_upper(0, 10) == pytest.approx(1 - 0.05 ** (1 / 10), rel=1e-9)
    assert clopper_pearson_upper(10, 10) == 1.0
    assert clopper_pearson_upper(0, 0) == 1.0
    # The defining property: seeing k or fewer at the bound has probability 5%.
    bound = clopper_pearson_upper(34, 68)
    assert binomial_cdf(34, 68, bound) == pytest.approx(0.05, abs=1e-9)
    # Exact is wider than the normal approximation at the same data.
    wald = 0.5 + 1.6448536269514722 * math.sqrt(0.25 / 68)
    assert bound > wald


def test_judgments_needed_reproduces_the_approach_reset_numbers():
    assert judgments_needed(0.50) == 68
    assert judgments_needed(0.55) == 268
    assert judgments_needed(0.60) is None
    assert judgments_needed(0.70) is None


def test_judgments_needed_exact_is_the_first_n_the_exact_bound_passes():
    for accuracy in (0.50, 0.55):
        n = judgments_needed_exact(accuracy)
        assert n is not None and n > judgments_needed(accuracy)  # type: ignore[operator]
        assert clopper_pearson_upper(math.ceil(accuracy * n - 1e-9), n) <= ACCURACY_THRESHOLD
        before = n - 1
        assert clopper_pearson_upper(math.ceil(accuracy * before - 1e-9), before) > 0.60
    assert judgments_needed_exact(0.5) == 76
    assert judgments_needed_exact(0.55) == 289
    assert judgments_needed_exact(0.61) is None


def _first_passing_n_by_definition(accuracy: float, limit: int) -> int | None:
    """The definition, spelled out: scan n and compute the bound each time."""
    for n in range(1, limit + 1):
        if clopper_pearson_upper(math.ceil(accuracy * n - 1e-9), n) <= ACCURACY_THRESHOLD:
            return n
    return None


@pytest.mark.parametrize("accuracy", [0.0, 0.1, 0.3, 0.45, 0.5, 0.52, 0.55])
def test_judgments_needed_exact_agrees_with_its_definition(accuracy: float):
    assert judgments_needed_exact(accuracy) == _first_passing_n_by_definition(accuracy, 400)


@pytest.mark.parametrize("accuracy", [117 / 200, 0.59])
def test_judgments_needed_exact_is_right_near_the_threshold_too(accuracy: float):
    n = judgments_needed_exact(accuracy)
    assert n is not None
    assert clopper_pearson_upper(math.ceil(accuracy * n - 1e-9), n) <= ACCURACY_THRESHOLD
    before = n - 1
    assert clopper_pearson_upper(math.ceil(accuracy * before - 1e-9), before) > 0.60


@pytest.mark.parametrize("correct", [116, 117, 118, 119])
def test_scoring_just_under_sixty_percent_is_quick(correct: int):
    """Observed accuracy just under 60% is exactly what this target produces;
    the planning row for it used to take from a minute to half an hour."""
    answers = {f"Q{i:03d}": (AI_LABEL if i % 2 else HUMAN_LABEL) for i in range(200)}
    ratings = [
        Rating("r1", item, answers[item] if i < correct else _other(answers[item]))
        for i, item in enumerate(answers)
    ]
    started = time.perf_counter()
    rendered = render_score(score(ratings, answers))
    assert time.perf_counter() - started < 5.0
    assert "None" not in rendered


def test_an_observed_accuracy_too_close_to_sixty_percent_says_beyond_the_search():
    assert judgments_needed_exact(0.599) is None  # past the 20,000-judgment search
    answers = {f"Q{i:04d}": (AI_LABEL if i % 2 else HUMAN_LABEL) for i in range(1000)}
    ratings = [
        Rating("r1", item, answers[item] if i < 599 else _other(answers[item]))
        for i, item in enumerate(answers)
    ]
    rendered = render_score(score(ratings, answers))
    assert "| 観測値 59.9% |" in rendered
    assert "20,000超" in rendered and "約None" not in rendered


def _ratings(pattern: dict[str, list[tuple[str, str]]]) -> list[Rating]:
    return [Rating(rater, item, guess) for rater, rows in pattern.items() for item, guess in rows]


def test_score_reports_accuracy_per_rater_and_per_answer():
    ratings = _ratings(
        {
            "r1": [("Q01", AI_LABEL), ("Q02", HUMAN_LABEL), ("Q03", AI_LABEL), ("Q04", AI_LABEL)],
            "r2": [("Q01", AI_LABEL), ("Q02", AI_LABEL), ("Q03", HUMAN_LABEL)],
        }
    )
    report = score(ratings, ANSWERS, blank_rows=1, ai_texts={"Q01": "〈A〉が怪しい", "Q03": "白"})
    assert (report.judgments, report.correct) == (7, 4)
    assert report.accuracy == pytest.approx(4 / 7)
    assert [(r.rater, r.judgments, r.correct) for r in report.raters] == [
        ("r1", 4, 3),
        ("r2", 3, 1),
    ]
    assert (report.ai_judgments, report.ai_correct) == (4, 3)
    assert (report.human_judgments, report.human_correct) == (3, 1)
    assert report.items_judged == 4 and report.items_in_key == 4
    assert report.upper_bound == pytest.approx(clopper_pearson_upper(4, 7))
    assert not report.target_demonstrated  # far too few judgments
    # Q01 was spotted by both raters, Q03 by one of two.
    assert report.most_spotted_ai[0] == ("Q01", 2, 2, "〈A〉が怪しい")
    data = report.to_dict()
    assert data["bound"] == "clopper-pearson one-sided"
    assert data["judgments_needed"]["normal_approx"] == {"0.50": 68, "0.55": 268}


def test_the_target_is_demonstrated_only_when_the_upper_bound_is_at_most_60_percent():
    answers = {f"Q{i:03d}": (AI_LABEL if i % 2 else HUMAN_LABEL) for i in range(200)}
    items = list(answers)
    # 100 of 200 right -> bound about 56%: demonstrated.
    ratings = [
        Rating("r1", item, answers[item] if i < 100 else _other(answers[item]))
        for i, item in enumerate(items)
    ]
    report = score(ratings, answers)
    assert report.accuracy == pytest.approx(0.5)
    assert report.upper_bound <= ACCURACY_THRESHOLD and report.target_demonstrated
    # 60 of 100 right -> bound above 60%: not demonstrated.
    ratings = [
        Rating("r1", item, answers[item] if i < 60 else _other(answers[item]))
        for i, item in enumerate(items[:100])
    ]
    assert not score(ratings, answers).target_demonstrated


def _other(label: str) -> str:
    return HUMAN_LABEL if label == AI_LABEL else AI_LABEL


def test_the_score_shows_ai_text_only_and_says_what_was_shown():
    ratings = _ratings({"r1": [("Q01", AI_LABEL), ("Q02", HUMAN_LABEL)]})
    ratings += _ratings({"r2": [("Q01", AI_LABEL), ("Q02", AI_LABEL)]})
    report = score(ratings, ANSWERS, ai_texts={"Q01": "AIの本文|縦棒"})
    rendered = render_score(report)
    assert "Clopper-Pearson" in rendered
    assert "まだ示せていない" in rendered
    assert "約68" in rendered and "約268" in rendered
    assert "AIの本文｜縦棒" in rendered  # table-safe
    assert render_score(score([], ANSWERS)).startswith("# 人間判定の結果")
