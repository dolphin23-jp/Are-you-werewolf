"""Can a reader tell the AI's lines from a human's? A blind quiz, and its score.

The v3 target is that raters shown game lines guess "AI or human" correctly
no more than 60% of the time (approach-reset §6, §11.5 Phase 4). Showing a
raw game log does not measure that: one human and sixteen AI seats make
about 94% of lines AI, so answering "AI" every time scores 94%. This module
builds the quiz the measurement needs and scores it:

  - **Balanced.** Exactly half the items are AI lines and half are human
    lines, so always answering one way scores 50%.
  - **Length-matched.** Lines are binned by length and every AI item is
    paired with a human item from the same bin, so "the long ones are AI"
    cannot carry the score.
  - **Nothing to read but the words.** Speaker names are not shown; names
    inside a line become neutral placeholders (`〈A〉`, `〈B〉`, ...) in both
    sources alike, and player ids, message ids and reply anchors are
    removed. The AI seats are named from one fixed list and a human log
    names other people, so a name left in would answer the question.
  - **Deterministic.** The same inputs and `seed` give the same quiz.

The human reference lines are supplied by the owner; the repository ships
none. They are usually someone else's words, so they -- and the quiz sheet
and answer key that contain them -- stay out of the repository unless the
owner has the right to publish them. The score contains no human text.

Scoring reports the accuracy with a one-sided 95% **Clopper-Pearson** upper
bound (exact binomial; never below the nominal coverage, so slightly
conservative). The planning counts in approach-reset §11 (about 68 judgments
if raters are really at 50%, about 268 at 55%) come from the normal
approximation `n = z^2 p(1-p) / (0.60 - p)^2`; the exact bound this score
uses needs a few more (about 76 and 289), and both are reported. Every
judgment is treated as independent, which flatters the bound when a few
raters judge a few items many times -- use several raters and enough items.
"""

from __future__ import annotations

import csv
import functools
import io
import json
import math
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

AI_LABEL = "AI"
HUMAN_LABEL = "人間"
LABELS = (AI_LABEL, HUMAN_LABEL)

# The target: raters at or below this accuracy (approach-reset §6).
ACCURACY_THRESHOLD = 0.60
CONFIDENCE = 0.95
# One-sided 95% normal quantile, for the planning counts.
_Z_ONE_SIDED_95 = 1.6448536269514722

# Upper edges (exclusive) of the length bins, in characters of the text a
# rater sees. Chat lines cluster at 20-80 characters; the last bin is open.
LENGTH_BIN_EDGES: tuple[int, ...] = (20, 40, 60, 90, 130)

KEY_FORMAT_VERSION = 1

_PLACEHOLDER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Ids the game or a log uses to point at a seat or a message: `p3`, `(p3)`,
# `[m12]`, `m12`, and reply anchors such as `>>12` / `＞＞１２`.
_ID_PATTERNS = (
    re.compile(r"[\(（]\s*[pm]\d+\s*[\)）]"),
    re.compile(r"\[\s*m\d+[^\]]*\]"),
    re.compile(r"(?<![A-Za-z0-9])[pm]\d+(?![A-Za-z0-9])"),
    re.compile(r"[>＞]{1,2}\s*[0-9０-９]+"),
)
_AT_BEFORE_PLACEHOLDER = re.compile(r"[@＠]\s*(?=〈)")
_SPACES = re.compile(r"\s+")


class QuizInputError(ValueError):
    """An input file the quiz cannot be built or scored from, with the reason."""


# --- source lines ----------------------------------------------------------


@dataclass(frozen=True)
class SourceLine:
    """A candidate line before anonymization.

    `names` are the names to mask in this line (every seat of the game for
    an AI line; whatever the owner listed for a human one). `origin` says
    where it came from, for the answer key only.
    """

    answer: str
    text: str
    origin: str
    names: tuple[str, ...] = ()


def ai_lines_from_transcript(
    data: dict[str, Any], origin: str, *, include_fallback: bool = False
) -> list[SourceLine]:
    """Public discussion lines from an evaluation transcript (`transcript-seedN.json`).

    Only `discussion` utterances are used -- what a player reads at the
    table. Lines the code posted in place of the model (`used_fallback`) are
    left out by default: they are canned persona lines, and the quiz asks
    about the model's speech.
    """
    utterances = data.get("utterances")
    if not isinstance(utterances, list):
        raise QuizInputError(f"{origin}: not an evaluation transcript (no 'utterances' list)")
    names_map = data.get("names") or {}
    names = tuple(str(name) for name in names_map.values() if str(name).strip())
    lines: list[SourceLine] = []
    for position, utterance in enumerate(utterances):
        if not isinstance(utterance, dict) or utterance.get("kind") != "discussion":
            continue
        if utterance.get("used_fallback") and not include_fallback:
            continue
        text = str(utterance.get("text") or "").strip()
        if text:
            lines.append(SourceLine(AI_LABEL, text, f"{origin}#{position}", names))
    return lines


def human_lines_from_text(content: str, origin: str) -> list[SourceLine]:
    """Owner-supplied human lines: plain text or JSONL, one utterance per line.

    Plain text: every non-empty line not starting with `#` is one utterance.
    JSONL (every non-empty line is a JSON object): `text` is required;
    `speaker` (a string) and `names` (a list) are masked in that line.
    """
    raw_lines = content.splitlines()
    candidates = [
        (number, line.strip())
        for number, line in enumerate(raw_lines, start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not candidates:
        return []
    jsonl = all(line.startswith("{") for _, line in candidates)
    lines: list[SourceLine] = []
    for number, line in candidates:
        where = f"{origin}:{number}"
        if not jsonl:
            lines.append(SourceLine(HUMAN_LABEL, line, where))
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QuizInputError(f"{where}: not valid JSON ({exc.msg})") from exc
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise QuizInputError(f"{where}: a JSONL line needs a string 'text' field")
        names: list[str] = []
        if isinstance(record.get("speaker"), str):
            names.append(record["speaker"])
        extra = record.get("names", [])
        if not isinstance(extra, list) or not all(isinstance(n, str) for n in extra):
            raise QuizInputError(f"{where}: 'names' must be a list of strings")
        names.extend(extra)
        text = record["text"].strip()
        if text:
            lines.append(SourceLine(HUMAN_LABEL, text, where, tuple(n for n in names if n.strip())))
    return lines


def names_from_text(content: str) -> tuple[str, ...]:
    """A mask list: one name per line, `#` comments and blanks ignored."""
    return tuple(
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


# --- anonymization ---------------------------------------------------------


def anonymize(text: str, names: Iterable[str]) -> str:
    """Replace names with placeholders and strip ids, the same for both sources.

    Placeholders are lettered by first appearance *within this line*, so the
    same letter in two items says nothing about the same person. Longer
    names are replaced first, so `ユイカ` is not left as `〈A〉カ` by `ユイ`.
    A short name can still match inside an unrelated word; that errs toward
    hiding too much, which is the safe side for a blind test.
    """
    out = text.replace("\r", " ").replace("\n", " ")
    for pattern in _ID_PATTERNS:
        out = pattern.sub("", out)
    unique = sorted({name.strip() for name in names if name.strip()}, key=len, reverse=True)
    if unique:
        tokens: dict[str, str] = {}
        # Swap each name for a private-use marker first, then letter the
        # markers in order of first appearance.
        for i, name in enumerate(unique):
            marker = chr(0xE000 + i)
            if name in out:
                out = out.replace(name, marker)
                tokens[marker] = name
        letters: dict[str, str] = {}
        result: list[str] = []
        for char in out:
            if char in tokens:
                if char not in letters:
                    index = len(letters)
                    letters[char] = _placeholder(index)
                result.append(letters[char])
            else:
                result.append(char)
        out = "".join(result)
    out = _AT_BEFORE_PLACEHOLDER.sub("", out)
    return _SPACES.sub(" ", out).strip()


def _placeholder(index: int) -> str:
    letter = _PLACEHOLDER_LETTERS[index % len(_PLACEHOLDER_LETTERS)]
    suffix = str(index // len(_PLACEHOLDER_LETTERS)) if index >= len(_PLACEHOLDER_LETTERS) else ""
    return f"〈{letter}{suffix}〉"


def length_bin(length: int) -> int:
    for index, edge in enumerate(LENGTH_BIN_EDGES):
        if length < edge:
            return index
    return len(LENGTH_BIN_EDGES)


def length_bin_label(index: int) -> str:
    low = 0 if index == 0 else LENGTH_BIN_EDGES[index - 1]
    if index >= len(LENGTH_BIN_EDGES):
        return f"{low}字以上"
    return f"{low}〜{LENGTH_BIN_EDGES[index] - 1}字"


# --- building --------------------------------------------------------------


@dataclass(frozen=True)
class QuizItem:
    item_id: str
    text: str
    answer: str
    origin: str
    length: int
    length_bin: int
    original_text: str


@dataclass
class Quiz:
    seed: int
    items: list[QuizItem]
    requested_items: int
    ai_pool: int
    human_pool: int
    # Per length bin: how many AI and human candidates there were, and how
    # many pairs were used.
    bins: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pairs(self) -> int:
        return len(self.items) // 2


@dataclass(frozen=True)
class _Candidate:
    answer: str
    text: str
    original: str
    origin: str


def _candidates(lines: Iterable[SourceLine], extra_names: Sequence[str]) -> list[_Candidate]:
    """Anonymize, drop empties, and drop repeats (the same line said twice
    would be two items with one answer)."""
    seen: set[str] = set()
    out: list[_Candidate] = []
    for line in lines:
        text = anonymize(line.text, (*line.names, *extra_names))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(_Candidate(line.answer, text, line.text, line.origin))
    return out


def _allocate(capacity: dict[int, int], total: int) -> dict[int, int]:
    """Split `total` pairs over bins in proportion to their capacity
    (largest remainder), never above any bin's capacity."""
    available = sum(capacity.values())
    if available <= total:
        return dict(capacity)
    exact = {b: total * c / available for b, c in capacity.items()}
    alloc = {b: min(capacity[b], math.floor(v)) for b, v in exact.items()}
    remainder = total - sum(alloc.values())
    order = sorted(capacity, key=lambda b: (-(exact[b] - math.floor(exact[b])), b))
    for b in order:
        if remainder <= 0:
            break
        if alloc[b] < capacity[b]:
            alloc[b] += 1
            remainder -= 1
    return alloc


def build_quiz(
    ai_lines: Sequence[SourceLine],
    human_lines: Sequence[SourceLine],
    *,
    items: int,
    seed: int,
    extra_names: Sequence[str] = (),
) -> Quiz:
    """A balanced, length-matched quiz of up to `items` lines (rounded down
    to an even number). Fewer are used when the inputs cannot fill it with
    matched pairs; `Quiz.bins` says where the shortfall is."""
    if items < 2:
        raise QuizInputError("a quiz needs at least 2 items (one AI, one human)")
    rng = random.Random(f"judgment-quiz:{seed}")
    ai = _candidates(ai_lines, extra_names)
    human = _candidates(human_lines, extra_names)
    ai_by_bin: dict[int, list[_Candidate]] = {}
    human_by_bin: dict[int, list[_Candidate]] = {}
    for candidate in ai:
        ai_by_bin.setdefault(length_bin(len(candidate.text)), []).append(candidate)
    for candidate in human:
        human_by_bin.setdefault(length_bin(len(candidate.text)), []).append(candidate)

    all_bins = sorted(set(ai_by_bin) | set(human_by_bin))
    capacity = {b: min(len(ai_by_bin.get(b, [])), len(human_by_bin.get(b, []))) for b in all_bins}
    allocation = _allocate({b: c for b, c in capacity.items() if c > 0}, items // 2)

    chosen: list[_Candidate] = []
    bins: list[dict[str, Any]] = []
    for b in all_bins:
        used = allocation.get(b, 0)
        if used:
            chosen.extend(rng.sample(ai_by_bin[b], used))
            chosen.extend(rng.sample(human_by_bin[b], used))
        bins.append(
            {
                "bin": b,
                "label": length_bin_label(b),
                "ai_candidates": len(ai_by_bin.get(b, [])),
                "human_candidates": len(human_by_bin.get(b, [])),
                "pairs": used,
            }
        )
    rng.shuffle(chosen)
    width = max(2, len(str(len(chosen))))
    quiz_items = [
        QuizItem(
            item_id=f"Q{number:0{width}d}",
            text=candidate.text,
            answer=candidate.answer,
            origin=candidate.origin,
            length=len(candidate.text),
            length_bin=length_bin(len(candidate.text)),
            original_text=candidate.original,
        )
        for number, candidate in enumerate(chosen, start=1)
    ]
    return Quiz(
        seed=seed,
        items=quiz_items,
        requested_items=items,
        ai_pool=len(ai),
        human_pool=len(human),
        bins=bins,
    )


def render_sheet(quiz: Quiz) -> str:
    """The page raters see: instructions and the lines, nothing else."""
    lines = [
        "# 人間判定テスト",
        "",
        "人狼ゲーム(17人のチャット村)の議論で出た発言を、1件ずつ並べています。",
        "各発言が **AI** の発言か **人間** の発言かを判定してください。",
        "",
        f"- 全 {len(quiz.items)} 件。AI と人間はちょうど半分ずつです。",
        "- 発言者の名前は伏せ、発言の中の名前は〈A〉〈B〉のような記号に置き換えてあります。"
        "同じ記号でも、別の発言では別の人を指します。",
        "- 迷っても、どちらかを選んでください。他の人と相談しないでください。",
        "- 回答は CSV の1行に1件で、`rater,item,guess` の形にします。`guess` は `AI` か "
        "`人間` です(例: `rater1,Q01,人間`)。",
        "",
    ]
    for item in quiz.items:
        lines += [f"**{item.item_id}** 　AI ／ 人間", "", f"> {item.text}", ""]
    return "\n".join(lines)


def ratings_template(quiz: Quiz, rater: str = "") -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["rater", "item", "guess"])
    for item in quiz.items:
        writer.writerow([rater, item.item_id, ""])
    return buffer.getvalue()


def answer_key(quiz: Quiz) -> dict[str, Any]:
    """Everything needed to score, and to audit how the quiz was made.

    Contains the human lines verbatim: keep it away from raters and, unless
    the owner may publish that text, out of the repository.
    """
    return {
        "format_version": KEY_FORMAT_VERSION,
        "seed": quiz.seed,
        "requested_items": quiz.requested_items,
        "ai_pool": quiz.ai_pool,
        "human_pool": quiz.human_pool,
        "length_bin_edges": list(LENGTH_BIN_EDGES),
        "bins": quiz.bins,
        "items": [asdict(item) for item in quiz.items],
    }


def answers_from_key(data: dict[str, Any]) -> dict[str, str]:
    """`item_id -> answer` from an answer key, validated."""
    items = data.get("items")
    if data.get("format_version") != KEY_FORMAT_VERSION or not isinstance(items, list):
        raise QuizInputError("not an answer key written by this tool (format_version 1)")
    answers: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise QuizInputError("answer key item is not an object")
        item_id, answer = item.get("item_id"), item.get("answer")
        if not isinstance(item_id, str) or answer not in LABELS:
            raise QuizInputError(f"answer key item {item_id!r} has no valid answer")
        answers[item_id] = str(answer)
    return answers


def ai_texts_from_key(data: dict[str, Any]) -> dict[str, str]:
    """The (anonymized) text of the AI items -- the only text a score shows."""
    return {
        str(item["item_id"]): str(item["text"])
        for item in data.get("items", [])
        if isinstance(item, dict) and item.get("answer") == AI_LABEL
    }


# --- scoring ---------------------------------------------------------------


@dataclass(frozen=True)
class Rating:
    rater: str
    item: str
    guess: str


_GUESS_ALIASES = {
    "ai": AI_LABEL,
    "ａｉ": AI_LABEL,
    "人間": HUMAN_LABEL,
    "human": HUMAN_LABEL,
}


def normalize_guess(value: str) -> str | None:
    """`AI` / `人間` (and the obvious spellings of them); None if blank.

    Anything else raises: a typo silently counted as either answer would
    move the accuracy.
    """
    cleaned = value.strip()
    if not cleaned:
        return None
    label = _GUESS_ALIASES.get(cleaned.lower())
    if label is None:
        raise QuizInputError(f"guess {value!r} is not {AI_LABEL} or {HUMAN_LABEL}")
    return label


def parse_ratings_csv(content: str, answers: dict[str, str]) -> tuple[list[Rating], int]:
    """Ratings from a `rater,item,guess` CSV, checked against the key.

    Returns the ratings and the number of rows left blank (a template row
    the rater did not fill in). Every problem is reported at once, with its
    line number, rather than one per run.
    """
    reader = csv.DictReader(io.StringIO(content.lstrip("﻿")))
    required = {"rater", "item", "guess"}
    if reader.fieldnames is None or not required <= {f.strip() for f in reader.fieldnames}:
        raise QuizInputError("the ratings CSV needs a header with rater,item,guess")
    ratings: list[Rating] = []
    problems: list[str] = []
    blank = 0
    seen: set[tuple[str, str]] = set()
    for number, raw in enumerate(reader, start=2):
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        rater, item = row.get("rater", ""), row.get("item", "")
        try:
            guess = normalize_guess(row.get("guess", ""))
        except QuizInputError as exc:
            problems.append(f"line {number}: {exc}")
            continue
        if guess is None:
            blank += 1
            continue
        if not rater:
            problems.append(f"line {number}: rater is blank")
            continue
        if item not in answers:
            problems.append(f"line {number}: item {item!r} is not in the answer key")
            continue
        if (rater, item) in seen:
            problems.append(f"line {number}: {rater} judged {item} twice")
            continue
        seen.add((rater, item))
        ratings.append(Rating(rater, item, guess))
    if problems:
        raise QuizInputError("; ".join(problems))
    return ratings, blank


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p).

    Summed in log space from the largest term, so it neither underflows for
    a few thousand judgments nor needs a statistics package.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_comb = math.lgamma(n + 1)
    logs = [
        log_comb - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        for i in range(k + 1)
    ]
    peak = max(logs)
    return min(1.0, math.exp(peak) * sum(math.exp(v - peak) for v in logs))


def clopper_pearson_upper(correct: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided exact upper confidence bound on a binomial proportion.

    The p at which seeing `correct` or fewer successes has probability
    1 - confidence; found by bisection on the binomial CDF, which is
    decreasing in p.
    """
    if n <= 0:
        return 1.0
    if correct >= n:
        return 1.0
    alpha = 1.0 - confidence
    low, high = correct / n, 1.0
    # 60 halvings take the interval below 1e-18, past double precision.
    for _ in range(60):
        mid = (low + high) / 2
        if binomial_cdf(correct, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def judgments_needed(
    true_accuracy: float, threshold: float = ACCURACY_THRESHOLD, z: float = _Z_ONE_SIDED_95
) -> int | None:
    """§11's planning count: judgments at which an observed accuracy equal to
    the true one puts the one-sided 95% (normal-approximation) upper bound
    exactly at `threshold`. Reaching it demonstrates the target about half
    the time; more judgments make it likelier. None at or above threshold."""
    if true_accuracy >= threshold:
        return None
    return math.ceil(z * z * true_accuracy * (1 - true_accuracy) / (threshold - true_accuracy) ** 2)


# How far `judgments_needed_exact` searches before answering "more than this".
EXACT_SEARCH_LIMIT = 20_000


@functools.lru_cache(maxsize=64)
def judgments_needed_exact(
    true_accuracy: float, threshold: float = ACCURACY_THRESHOLD, limit: int = EXACT_SEARCH_LIMIT
) -> int | None:
    """The same question with the Clopper-Pearson bound this report uses:
    the smallest n whose bound is at or below `threshold` when the correct
    count is `true_accuracy * n` rounded *up* (never flattering the raters'
    accuracy by rounding). The exact bound is wider than the normal one, so
    this comes out larger than `judgments_needed` (about 76 and 289 for 50%
    and 55%, against 68 and 268); a few more than that make it stable,
    because the integer rounding makes the pass/fail line jagged in n.
    None when no n up to `limit` passes.

    The jagged line rules out a bisection over n, so every n is tried, and
    trying one must be cheap: an accuracy just under 60% -- the range this
    target produces -- is thousands of n from passing. Two things make it
    O(1) per n instead of a bisected bound over an O(n) sum (which took up
    to half an hour). The bound is at or below `threshold` exactly when
    P(X <= correct) <= 1 - confidence at p = `threshold`, because the CDF
    falls as p rises; so one CDF value per n decides it. And that value is
    carried from n to n + 1 by the binomial recurrences rather than summed
    afresh.
    """
    if true_accuracy >= threshold:
        return None
    alpha = 1.0 - CONFIDENCE
    p, q = threshold, 1.0 - threshold

    def correct_at(n: int) -> int:
        # The epsilon keeps 0.55 * 20 = 11.000000000000002 from rounding to 12.
        return max(0, math.ceil(true_accuracy * n - 1e-9))

    n = max(1, (judgments_needed(true_accuracy, threshold) or 2) // 2)
    if n > limit:
        return None
    k = correct_at(n)  # never above n, since true_accuracy < threshold <= 1
    # cdf = P(X_n <= k) and pmf = P(X_n = k) for X_n ~ Binomial(n, threshold).
    cdf = binomial_cdf(k, n, p)
    log_pmf = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    pmf = math.exp(log_pmf + k * math.log(p) + (n - k) * math.log(q))
    while True:
        if cdf <= alpha:
            return n
        if n >= limit:
            return None
        # One more judgment, same k: X_{n+1} <= k unless the new one is a
        # success landing on X_n = k.
        cdf -= p * pmf
        pmf *= q * (n + 1) / (n + 1 - k)
        n += 1
        if correct_at(n) > k:  # k grows by at most one per judgment
            pmf *= (n - k) / (k + 1) * (p / q)
            k += 1
            cdf += pmf


@dataclass
class RaterScore:
    rater: str
    judgments: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.judgments if self.judgments else 0.0


@dataclass
class ScoreReport:
    judgments: int
    correct: int
    blank_rows: int
    raters: list[RaterScore]
    items_in_key: int
    items_judged: int
    ai_judgments: int
    ai_correct: int
    human_judgments: int
    human_correct: int
    upper_bound: float
    # AI items raters spotted most often: (item, times spotted, judgments, text).
    most_spotted_ai: list[tuple[str, int, int, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.judgments if self.judgments else 0.0

    @property
    def target_demonstrated(self) -> bool:
        return self.judgments > 0 and self.upper_bound <= ACCURACY_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            accuracy=self.accuracy,
            target_demonstrated=self.target_demonstrated,
            threshold=ACCURACY_THRESHOLD,
            confidence=CONFIDENCE,
            bound="clopper-pearson one-sided",
            raters=[{**asdict(r), "accuracy": r.accuracy} for r in self.raters],
            judgments_needed={
                "normal_approx": {
                    "0.50": judgments_needed(0.50),
                    "0.55": judgments_needed(0.55),
                },
                "clopper_pearson": {
                    "0.50": judgments_needed_exact(0.50),
                    "0.55": judgments_needed_exact(0.55),
                },
            },
        )
        return data


def score(
    ratings: Sequence[Rating],
    answers: dict[str, str],
    *,
    blank_rows: int = 0,
    ai_texts: dict[str, str] | None = None,
    spotted_limit: int = 5,
) -> ScoreReport:
    correct = sum(1 for r in ratings if answers[r.item] == r.guess)
    per_rater: dict[str, RaterScore] = {}
    for rating in ratings:
        entry = per_rater.setdefault(rating.rater, RaterScore(rating.rater, 0, 0))
        entry.judgments += 1
        entry.correct += int(answers[rating.item] == rating.guess)
    ai_ratings = [r for r in ratings if answers[r.item] == AI_LABEL]
    human_ratings = [r for r in ratings if answers[r.item] == HUMAN_LABEL]

    spotted: list[tuple[str, int, int, str]] = []
    if ai_texts:
        by_item: dict[str, list[Rating]] = {}
        for rating in ai_ratings:
            by_item.setdefault(rating.item, []).append(rating)
        for item, group in by_item.items():
            hits = sum(1 for r in group if r.guess == AI_LABEL)
            if hits and len(group) >= 2:
                spotted.append((item, hits, len(group), ai_texts.get(item, "")))
        spotted.sort(key=lambda entry: (-entry[1] / entry[2], -entry[2], entry[0]))

    return ScoreReport(
        judgments=len(ratings),
        correct=correct,
        blank_rows=blank_rows,
        raters=sorted(per_rater.values(), key=lambda r: r.rater),
        items_in_key=len(answers),
        items_judged=len({r.item for r in ratings}),
        ai_judgments=len(ai_ratings),
        ai_correct=sum(1 for r in ai_ratings if r.guess == AI_LABEL),
        human_judgments=len(human_ratings),
        human_correct=sum(1 for r in human_ratings if r.guess == HUMAN_LABEL),
        upper_bound=clopper_pearson_upper(correct, len(ratings)),
        most_spotted_ai=spotted[:spotted_limit],
    )


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_score(report: ScoreReport) -> str:
    """The score as markdown. No human text appears in it."""
    if report.judgments == 0:
        return "# 人間判定の結果\n\n判定がまだ1件もありません。\n"
    verdict = (
        f"**示せた**: 正答率の片側95%上限 {_pct(report.upper_bound)} が "
        f"{ACCURACY_THRESHOLD:.0%} 以下。"
        if report.target_demonstrated
        else f"**まだ示せていない**: 片側95%上限 {_pct(report.upper_bound)} が "
        f"{ACCURACY_THRESHOLD:.0%} を超える(判定を増やすか、正答率が本当に高い)。"
    )
    lines = [
        "# 人間判定の結果",
        "",
        f"- 判定: {report.judgments}件(評価者 {len(report.raters)}人、"
        f"判定された発言 {report.items_judged} / {report.items_in_key}件)",
        f"- 正答率: **{_pct(report.accuracy)}**({report.correct} / {report.judgments})",
        f"- 片側95%上限(Clopper-Pearson): {_pct(report.upper_bound)}",
        f"- 目標「正答率 {ACCURACY_THRESHOLD:.0%} 以下」: {verdict}",
    ]
    if report.blank_rows:
        lines.append(f"- 空欄の行: {report.blank_rows}件(数えていない)")
    ai_rate = report.ai_correct / report.ai_judgments if report.ai_judgments else 0.0
    human_rate = report.human_correct / report.human_judgments if report.human_judgments else 0.0
    lines += [
        "",
        "| 正解 | 判定 | 正しく当てた | 率 |",
        "|---|---:|---:|---:|",
        f"| AI の発言 | {report.ai_judgments} | {report.ai_correct} | {_pct(ai_rate)} |",
        f"| 人間の発言 | {report.human_judgments} | {report.human_correct} | {_pct(human_rate)} |",
        "",
        "AI の発言を当てた率と人間の発言を当てた率が大きく違うときは、評価者が片方に寄って"
        "答えている(どちらとも言えないものを一方に倒している)。",
        "",
        "## 評価者ごと",
        "",
        "| 評価者 | 判定 | 正答 | 正答率 |",
        "|---|---:|---:|---:|",
    ]
    for rater in report.raters:
        lines.append(
            f"| {rater.rater} | {rater.judgments} | {rater.correct} | {_pct(rater.accuracy)} |"
        )
    need_50, need_55 = judgments_needed(0.50), judgments_needed(0.55)
    exact_50, exact_55 = judgments_needed_exact(0.50), judgments_needed_exact(0.55)
    lines += [
        "",
        "## 必要な判定の数",
        "",
        "正答率 60% 以下を片側95%で示すのに要る判定の数(観測された正答率が真の値どおりだった"
        "ときに、上限がちょうど60%に届く数。届く確率はおよそ半分なので、余裕を持って集める)。",
        "",
        "| 真の正答率 | 正規近似(approach-reset §11) | Clopper-Pearson(この表の上限) |",
        "|---|---:|---:|",
        f"| 50% | 約{need_50} | 約{exact_50} |",
        f"| 55% | 約{need_55} | 約{exact_55} |",
    ]
    if report.accuracy < ACCURACY_THRESHOLD:
        observed = judgments_needed(report.accuracy)
        observed_exact = judgments_needed_exact(report.accuracy)
        exact_text = (
            f"約{observed_exact}"
            if observed_exact is not None
            else f"{EXACT_SEARCH_LIMIT:,}超"  # past the search, not "None"
        )
        lines.append(f"| 観測値 {_pct(report.accuracy)} | 約{observed} | {exact_text} |")
    lines += [
        "",
        "各判定を独立とみなしている。少ない評価者が少ない発言を何度も判定すると、上限は"
        "実際より楽観的になる。評価者を複数にし、発言の数も十分にとる。",
    ]
    if report.most_spotted_ai:
        lines += [
            "",
            "## 見抜かれやすかった AI の発言",
            "",
            "ドクトリンと口調の例を直すときの手がかり(§7 Phase D)。",
            "",
            "| 発言 | AIと当てた | 判定 | 本文 |",
            "|---|---:|---:|---|",
        ]
        for item, hits, total, text in report.most_spotted_ai:
            cell = text.replace("|", "｜")
            lines.append(f"| {item} | {hits} | {total} | {cell} |")
    return "\n".join(lines) + "\n"


__all__ = [
    "ACCURACY_THRESHOLD",
    "AI_LABEL",
    "CONFIDENCE",
    "EXACT_SEARCH_LIMIT",
    "HUMAN_LABEL",
    "LABELS",
    "LENGTH_BIN_EDGES",
    "Quiz",
    "QuizInputError",
    "QuizItem",
    "Rating",
    "RaterScore",
    "ScoreReport",
    "SourceLine",
    "ai_lines_from_transcript",
    "ai_texts_from_key",
    "answer_key",
    "answers_from_key",
    "anonymize",
    "binomial_cdf",
    "build_quiz",
    "clopper_pearson_upper",
    "human_lines_from_text",
    "judgments_needed",
    "judgments_needed_exact",
    "length_bin",
    "length_bin_label",
    "names_from_text",
    "normalize_guess",
    "parse_ratings_csv",
    "ratings_template",
    "render_score",
    "render_sheet",
    "score",
]
