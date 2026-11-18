"""The first live Claude game (2026-11-16, v3, seed 1), rebuilt as offline boards.

`backend/artifacts/live-v3-claude-opus55-seed1-2026-11-16/` holds the record;
its `analysis.md` (P1) is why these exist. Two mornings matter:

* **Day 2.** Two corpses after night 1 (ソウタ, the fox, cursed by ケント; ケント,
  the real seer, bitten). ダイキ (a wolf) published シオン white for night 1
  while シオン was alive -- from public information alone he could not be the
  real seer, and the fox was already dead. The memo said neither, and the
  village hanged レン, the villager who said it.
* **Day 4.** ダイキ executed and read black, アオイ (the madman) blacks ツムギ, and
  the table still argued as if a live fox were possible.

Only what the solver and memo read is rebuilt: deal, claims, verdicts (with the
night each was stated for), votes, deaths and the night actions that happened.
Speech is not -- these boards test logic, not wording.
"""

from __future__ import annotations

from app.engine.roles import RoleName
from app.engine.speech_events import SpeechEventType
from app.engine.state import GameState, MediumRecord, VoteRecord
from tests.ai.reasoning.solver import boards

NAMES: dict[str, str] = {
    "p0": "マコト",
    "p1": "アカリ",
    "p2": "ハルト",
    "p3": "ユイ",
    "p4": "ソウタ",
    "p5": "ミオ",
    "p6": "レン",
    "p7": "サクラ",
    "p8": "カイト",
    "p9": "ノゾミ",
    "p10": "リク",
    "p11": "ツムギ",
    "p12": "ダイキ",
    "p13": "ホノカ",
    "p14": "シオン",
    "p15": "アオイ",
    "p16": "ケント",
}

ROLES: dict[str, RoleName] = {
    "p0": RoleName.FREEMASON,
    "p1": RoleName.MEDIUM,
    "p2": RoleName.WEREWOLF,
    "p4": RoleName.FOX,
    "p5": RoleName.HUNTER,
    "p8": RoleName.WEREWOLF,
    "p12": RoleName.WEREWOLF,
    "p13": RoleName.FREEMASON,
    "p15": RoleName.MADMAN,
    "p16": RoleName.SEER,
}

# Seats by name, so a test reads like the game it is about.
MAKOTO, AKARI, HARUTO, YUI, SOTA, MIO, REN, SAKURA, KAITO = (
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
)
NOZOMI, RIKU, TSUMUGI, DAIKI, HONOKA, SHION, AOI, KENT = (
    "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16",
)

_DAY_ONE_VOTES = {
    "p2": "p14", "p5": "p11", "p0": "p6", "p3": "p2", "p7": "p2", "p1": "p2",
    "p4": "p14", "p6": "p14", "p8": "p6", "p12": "p14", "p11": "p2", "p13": "p2",
    "p10": "p2", "p15": "p14", "p14": "p2", "p16": "p2",
}


def _named(state: GameState) -> GameState:
    for pid, name in NAMES.items():
        state.players[pid].name = name
    return state


def day_two_morning() -> GameState:
    """Day 2 discussion, after ダイキ and アオイ published their night-1 results."""
    state = _named(boards.deal(ROLES, day=2))
    # Night 0: the scripted victim; the real seer looks at リク.
    boards.kill_first_victim(state, NOZOMI)
    boards.divine(state, KENT, RIKU, night=0)
    # Day 1: three seer claims with night-0 whites, a medium, the mason pair.
    boards.claim(state, KENT, RoleName.SEER, day=1)
    boards.verdict(state, KENT, "seer", RIKU, False, day=1, referenced_day=0)
    boards.claim(state, MAKOTO, RoleName.FREEMASON, day=1)
    boards._event(state, 1, MAKOTO, SpeechEventType.PARTNER_CLAIM, target_id=HONOKA)
    boards.claim(state, DAIKI, RoleName.SEER, day=1)
    boards.verdict(state, DAIKI, "seer", SOTA, False, day=1, referenced_day=0)
    boards.claim(state, AOI, RoleName.SEER, day=1)
    boards.verdict(state, AOI, "seer", YUI, False, day=1, referenced_day=0)
    boards.claim(state, AKARI, RoleName.MEDIUM, day=1)
    boards.claim(state, HONOKA, RoleName.FREEMASON, day=1)
    boards._event(state, 1, HONOKA, SpeechEventType.PARTNER_CLAIM, target_id=MAKOTO)
    state.vote_records.extend(
        VoteRecord(voter_id=voter, target_id=target, day=1, round=1)
        for voter, target in _DAY_ONE_VOTES.items()
    )
    boards.execute(state, HARUTO, day=1)
    # Night 1: ケント divines the fox, the wolves bite ケント. Two corpses.
    boards.divine(state, KENT, SOTA, night=1)
    boards.attack(state, DAIKI, KENT, night=1, succeeded=True)
    boards.die_by_curse(state, SOTA, night=1)
    boards.die_by_attack(state, KENT, night=1)
    state.medium_records.append(_medium(AKARI, HARUTO, day=1, black=True))
    # Day 2 morning results.
    boards.verdict(state, AKARI, "medium", HARUTO, True, day=2, referenced_day=1)
    boards.verdict(state, DAIKI, "seer", SHION, False, day=2, referenced_day=1)
    boards.verdict(state, AOI, "seer", SOTA, False, day=2, referenced_day=1)
    return state


def day_four_morning() -> GameState:
    """Day 4 discussion: ダイキ hanged and read black, アオイ blacks ツムギ."""
    state = day_two_morning()
    boards.execute(state, REN, day=2)
    # Night 2: the hunter's guard holds -- no corpse.
    boards.guard(state, MIO, AOI, night=2)
    boards.attack(state, KAITO, AOI, night=2, succeeded=False)
    state.medium_records.append(_medium(AKARI, REN, day=2, black=False))
    boards.verdict(state, AKARI, "medium", REN, False, day=3, referenced_day=2)
    # アオイ published nothing for night 2; ダイキ "covered" カイト.
    boards.verdict(state, DAIKI, "seer", KAITO, False, day=3, referenced_day=2)
    boards.execute(state, DAIKI, day=3)
    # Night 3: guarded again, no corpse.
    boards.guard(state, MIO, AOI, night=3)
    boards.attack(state, KAITO, AOI, night=3, succeeded=False)
    state.medium_records.append(_medium(AKARI, DAIKI, day=3, black=True))
    boards.verdict(state, AKARI, "medium", DAIKI, True, day=4, referenced_day=3)
    boards.verdict(state, AOI, "seer", TSUMUGI, True, day=4, referenced_day=3)
    state.day = 4
    return state


def _medium(medium_id: str, target_id: str, *, day: int, black: bool) -> MediumRecord:
    return MediumRecord(medium_id=medium_id, target_id=target_id, day=day, is_werewolf=black)
