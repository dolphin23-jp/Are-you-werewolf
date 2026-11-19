"""Skill: how soon a seat notices what follows from the board.

The owner's decision after run #14 splits the board memo in two:

* **Records** -- CO counts, deaths and when, the grey, ballots, "a seat killed
  at night is not a wolf", which claimants died at night, each seer
  claimant's account. Every seat reads them at once.
* **Derived conclusions** -- a claimant's account breaking down (破綻), what
  ends the game, what a curse says about a claimant (and, to a fake seer about
  to publish, what a real seer's account of the curse night must say), what a
  freemason's naming makes of a rival claim. Code always computes them
  correctly; *when a seat is shown them* depends on its skill.

Skill is separate from the persona (tone and amount only) and is dealt per game
from the seed, so a replay deals the same skills:

* 熟練 (expert)  -- sees a derived conclusion as soon as it is derivable;
* 普通 (average) -- from the next day, or earlier once someone states it at
  the table;
* 初心者 (novice) -- only once someone states it at the table.

"Stated at the table" is read by rule, never by intent: a public line, said
after the conclusion became derivable, that names one of the conclusion's seats
together with one of `TABLE_STATEMENT_WORDS`. Crude on purpose -- it can be
fooled by a line that says 「レンが破綻とは思わない」, which still puts the
conclusion on the table for anyone listening.

Role-private conclusions (from the seat's own card, results or allies) are not
gated: a seat knows what it knows.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from app.engine.state import ChatChannel, GameState


class Skill(StrEnum):
    EXPERT = "expert"
    AVERAGE = "average"
    NOVICE = "novice"

    @property
    def label_ja(self) -> str:
        return _LABELS[self]


_LABELS = {Skill.EXPERT: "熟練", Skill.AVERAGE: "普通", Skill.NOVICE: "初心者"}

# Words that put a contradiction, a certainty or an ending on the table. A line
# naming a conclusion's seat with one of these counts as stating it.
TABLE_STATEMENT_WORDS: tuple[str, ...] = (
    "破綻",
    "偽確定",
    "確定",
    "ありえない",
    "あり得ない",
    "矛盾",
    "最後の人狼",
    "最後の狼",
    "試合が終わ",
    "ゲームが終わ",
    "終了",
    "決着",
)


def assign_skills(player_ids: Sequence[str], seed: int | None = None) -> dict[str, Skill]:
    """Deal skills per game from the seed: a third of the seats each, shuffled.

    Its own random stream (keyed on the seed and "skill"), so it is independent
    of the personality deal, which draws from the bare seed.
    """
    rng = random.Random(f"{seed}:skill")
    levels = tuple(Skill)
    pool = [levels[index % len(levels)] for index in range(len(player_ids))]
    rng.shuffle(pool)
    return dict(zip(player_ids, pool, strict=True))


@dataclass(frozen=True)
class DerivedItem:
    """One derived conclusion as the memo would print it.

    `key` names the conclusion, not its wording: a conditional line whose
    candidate list grows is the same conclusion, while "ツムギ is the only
    living wolf" is a new one. `seats` are the seats a table statement has to
    name for it to count as stated.
    """

    key: tuple[str, ...]
    text: str
    seats: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stamp:
    """When a conclusion first became derivable: the day, and how many public
    lines were on the table then (a statement must come after them)."""

    day: int
    position: int


def public_lines(state: GameState) -> list[str]:
    return [m.content for m in state.chat_log if m.channel is ChatChannel.PUBLIC]


def stamp_now(state: GameState) -> Stamp:
    return Stamp(day=state.day, position=len(public_lines(state)))


@dataclass
class DerivationClock:
    """First-derivable stamps, kept across the game by the reasoning runtime.

    The memo is rendered on every turn, and a conclusion can only become
    derivable when the board changes; so the first render that computes it is
    the moment it became derivable, to within the turn.
    """

    stamps: dict[tuple[str, ...], Stamp] = field(default_factory=dict)

    def observe(self, items: Iterable[DerivedItem], state: GameState) -> None:
        now = stamp_now(state)
        for item in items:
            self.stamps.setdefault(item.key, now)

    def stamp(self, key: tuple[str, ...], state: GameState) -> Stamp:
        return self.stamps.get(key, stamp_now(state))


def stated_at_table(
    state: GameState, item: DerivedItem, since: Stamp, names: Mapping[str, str]
) -> bool:
    """A public line after `since` names one of the item's seats with a
    contradiction / certainty / ending word."""
    wanted = [names[pid] for pid in item.seats if pid in names]
    if not wanted:
        return False
    for text in public_lines(state)[since.position :]:
        if any(name in text for name in wanted) and any(
            word in text for word in TABLE_STATEMENT_WORDS
        ):
            return True
    return False


def is_visible(
    item: DerivedItem,
    skill: Skill,
    state: GameState,
    clock: DerivationClock | None,
    names: Mapping[str, str],
) -> bool:
    if skill is Skill.EXPERT:
        return True
    since = clock.stamp(item.key, state) if clock is not None else stamp_now(state)
    if skill is Skill.AVERAGE and since.day < state.day:
        return True
    return stated_at_table(state, item, since, names)


__all__ = [
    "TABLE_STATEMENT_WORDS",
    "DerivationClock",
    "DerivedItem",
    "Skill",
    "Stamp",
    "assign_skills",
    "is_visible",
    "public_lines",
    "stamp_now",
    "stated_at_table",
]
