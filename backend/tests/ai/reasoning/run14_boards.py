"""AI evaluation run #14 (2026-11-17, v3), seeds 1-3, rebuilt as offline boards.

`backend/artifacts/live-v3-claude-sonnet55-seeds1-3-2026-11-17/` holds the
record; its `analysis.md` and the owner's review of it found the endgame
consequences and the fake claimants' contradictions never reaching the board.
Each fix is pinned at the real moment of these games (see `record_boards`).

The deals, by name (seat ids are the same p0..p16 order in every seed):

* seed 1 -- ケント seer (bitten night 2); ダイキ, ハルト, カイト wolves, ダイキ
  the fake seer; アオイ madman and fake seer (bitten night 1); ソウタ fox
  (hanged day 3); アカリ medium. On day 6 ダイキ had given white to all five
  other living seats; the village hanged him and won.
* seed 2 -- ハルト seer (bitten night 1); ミオ, ダイキ, ケント wolves, ダイキ
  the fake seer and ケント the fake medium; ソウタ madman; アオイ fox; レン
  medium (bitten night 2). On day 6 ケント had reported white for all five
  executions; he was hanged last and the fox won.
* seed 3 -- カイト seer (bitten night 1, his black on ツムギ unpublished);
  マコト, レン, ツムギ wolves, レン the fake seer; ミオ madman (first victim);
  ホノカ fox, divined by レン's fake account on night 3; ソウタ medium. On day 6
  レン genuine would have left ツムギ the only living wolf; the wolves won.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from app.ai.reasoning.board_memo import render_board_memo
from app.ai.reasoning.observations import ObservationSet
from app.ai.reasoning.perspectives import CommonPublicPerspective, PlayerPrivatePerspective
from app.ai.reasoning.solver import build_solver
from app.engine.state import GameState
from tests.ai.reasoning.record_boards import board_from_record

RUN = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "live-v3-claude-sonnet55-seeds1-3-2026-11-17"
)

MAKOTO, AKARI, HARUTO, YUI, SOTA, MIO, REN, SAKURA, KAITO = (
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
)
NOZOMI, RIKU, TSUMUGI, DAIKI, HONOKA, SHION, AOI, KENT = (
    "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16",
)


@cache
def record(seed: int) -> dict[str, Any]:
    path = RUN / f"seed{seed}" / f"transcript-seed{seed}.json"
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def board(seed: int, day: int, at: str = "morning") -> GameState:
    return board_from_record(record(seed)["final_state"], f"run14-seed{seed}", day, at)


def name(seed: int, seat: str) -> str:
    return str(record(seed)["names"][seat])


def memo(state: GameState, seat: str, **kwargs: Any) -> str:
    """The memo the seat reads, through the same solvers the runtime builds."""
    observations = ObservationSet.from_state(state)
    return render_board_memo(
        state,
        seat,
        observations=observations,
        public_solver=build_solver(observations, CommonPublicPerspective()),
        seat_solver=build_solver(observations, PlayerPrivatePerspective(seat)),
        **kwargs,
    )
