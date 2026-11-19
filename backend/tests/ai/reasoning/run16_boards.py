"""AI evaluation run #16 (2026-11-18, v3, seed 1), rebuilt as offline boards.

`backend/artifacts/live-v3-claude-sonnet55-seed1-2026-11-18b/` holds the record.
The owner's review of it (X12, X23-X25) is pinned at the real moments of this
game (see `record_boards`).

The deal, by name: ケント the seer, divined ソウタ (the fox) on night 1 and
was bitten the same night -- two corpses on day 2. アオイ the madman and ダイキ
a wolf claimed seer and both published a living night-1 target; the village
hanged them on days 2 and 3. ハルト and カイト the other wolves; マコト and
ホノカ the freemasons, confirmed on day 1, ホノカ bitten on night 3; アカリ the
medium (bitten night 2); ミオ the hunter.

The record has one claim that was never made: カイト's m9 「共有CO二人は了解」
acknowledged the freemasons and was registered as his own freemason CO (an
extractor fault fixed elsewhere). He then stood on it (m18 「本物だから出た」),
and a third freemason claim can happen in a real game, so boards keep it by
default; `false_co=False` drops it, for the facts that must not depend on it.
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
    / "live-v3-claude-sonnet55-seed1-2026-11-18b"
)

MAKOTO, AKARI, HARUTO, YUI, SOTA, MIO, REN, SAKURA, KAITO = (
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
)
NOZOMI, RIKU, TSUMUGI, DAIKI, HONOKA, SHION, AOI, KENT = (
    "p9", "p10", "p11", "p12", "p13", "p14", "p15", "p16",
)

# The speech event that registered カイト's acknowledgement (m9) as a CO.
FALSE_CO_EVENT = "e10"


@cache
def record() -> dict[str, Any]:
    path = RUN / "transcript-seed1.json"
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def _final(false_co: bool) -> dict[str, Any]:
    final = record()["final_state"]
    if false_co:
        return final
    return {
        **final,
        "speech_events": [
            event for event in final["speech_events"] if event["event_id"] != FALSE_CO_EVENT
        ],
    }


def board(day: int, at: str = "morning", *, false_co: bool = True) -> GameState:
    return board_from_record(_final(false_co), "run16-seed1", day, at)


def name(seat: str) -> str:
    return str(record()["names"][seat])


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
