#!/usr/bin/env python3
"""Play one real-provider game under the v2 engine and report what it did.

Everything else in this repository is measured against `MockProvider`, which is
the right default -- tests must cost nothing and never touch a network. But a
mock never argues badly, never misquotes a ballot and never invents a schedule,
so "v2 reasons well" cannot be concluded from mock runs alone. This is the
script that goes and looks.

Manual on purpose. It spends real requests, so it is never run by CI and never
imported by a test:

    WEREWOLF_LLM_PROVIDER=luna LUNA_API_KEY=... \\
        python scripts/live_reasoning_check.py --seed 11

Run `scripts/check_llm.py` first -- one call to find a bad key beats a few
hundred. What this reports beyond the mock comparison:

* the correctness counters, which must be zero against a live model too;
* how often a model tried to substitute its own execution target and was
  overridden -- the number that says whether the enforcement is load-bearing;
* published accounts that contradicted the public calendar, which mocks do not
  generate at all;
* the transcript path, so a human can read the game and judge the parts no
  counter captures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.coordinator import AICoordinator  # noqa: E402
from app.ai.metrics import MetricsCollector  # noqa: E402
from app.ai.provider.factory import build_llm_provider  # noqa: E402
from app.ai.reasoning.runtime import ReasoningRuntime  # noqa: E402
from app.ai.reasoning.timeline import find_timeline_conflicts  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.engine.game import GameController, PlayerSpec  # noqa: E402
from app.engine.phases import Phase  # noqa: E402
from app.engine.roles import RoleName  # noqa: E402
from app.eval.transcript import TranscriptRecorder  # noqa: E402

MAX_LOOPS = 150
HUMAN_ID = "p0"
AI_IDS = [f"p{i}" for i in range(1, 17)]


class _Session:
    def __init__(self, controller: GameController, coordinator: AICoordinator) -> None:
        self.controller = controller
        self.coordinator = coordinator
        self.human_id = HUMAN_ID
        self.discussion_lock = asyncio.Lock()
        self.discussion_round: Any = None
        self.discussion_paused = False
        self.discussion_pause_requested = False
        self.discussion_step_budget: int | None = None


def _specs() -> list[PlayerSpec]:
    return [
        PlayerSpec(player_id=f"p{i}", name=f"Player{i}", is_human=(i == 0))
        for i in range(17)
    ]


def _human_night(controller: GameController, rng: random.Random) -> None:
    state = controller.state
    human = state.players[HUMAN_ID]
    if not human.alive:
        return
    candidates = [pid for pid in state.alive_ids() if pid != HUMAN_ID]
    if not candidates:
        return
    if human.role is RoleName.SEER:
        controller.submit_night_action(HUMAN_ID, "divine", rng.choice(candidates))
    elif state.day > 0 and human.role is RoleName.HUNTER:
        controller.submit_night_action(HUMAN_ID, "guard", rng.choice(candidates))
    elif (
        state.day > 0
        and human.role is RoleName.WEREWOLF
        and controller.alpha_wolf_id == HUMAN_ID
    ):
        prey = [
            pid for pid in candidates if state.players[pid].role is not RoleName.WEREWOLF
        ]
        if prey:
            controller.submit_night_action(HUMAN_ID, "attack", rng.choice(prey))


async def run(seed: int, transcript_path: Path | None) -> dict[str, Any]:
    settings = get_settings()
    if settings.werewolf_llm_provider == "mock":
        raise SystemExit(
            "この確認は実プロバイダ用です。WEREWOLF_LLM_PROVIDER=luna を設定してください。"
        )
    metrics = MetricsCollector()
    provider = build_llm_provider(settings, seed=seed, metrics=metrics)
    controller = GameController(
        session_id=f"live-{seed}", player_specs=_specs(), seed=seed
    )
    reasoning = ReasoningRuntime(controller.state, AI_IDS, seed=seed)
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(
        controller.state,
        AI_IDS,
        provider,
        seed=seed,
        recorder=recorder,
        reasoning=reasoning,
        pacing_scale=0.0,
    )
    session = _Session(controller, coordinator)
    controller.start_game()
    rng = random.Random(seed)
    conflicts: list[str] = []

    for _ in range(MAX_LOOPS):
        state = controller.state
        if state.phase is Phase.GAME_OVER:
            break
        if state.phase is Phase.NIGHT:
            _human_night(controller, rng)
            await coordinator.run_night_phase(session)
        elif state.phase is Phase.DAWN:
            controller.start_discussion()
        elif state.phase is Phase.DISCUSSION:
            await coordinator.run_discussion_round(session)
            reasoning.refresh(state)
            conflicts.extend(
                conflict.explanation
                for conflict in find_timeline_conflicts(reasoning.observations)
            )
            controller.end_discussion()
        elif state.phase in (Phase.VOTING, Phase.RUNOFF):
            human = state.players[HUMAN_ID]
            if human.alive:
                candidates = state.votable_ids(HUMAN_ID)
                if candidates:
                    controller.vote(HUMAN_ID, rng.choice(candidates))
            await coordinator.generate_all_votes(session)
        elif state.phase is Phase.VOTE_RESULT:
            controller.start_night()
        else:
            break

    codes = coordinator.validation.codes()
    report: dict[str, Any] = {
        "seed": seed,
        "provider": settings.werewolf_llm_provider,
        "model": settings.luna_model,
        "days": controller.state.day,
        "llm_requests": metrics.summary().get("total_calls", 0),
        "tokens": metrics.summary().get("tokens", {}),
        # Must be zero against a live model exactly as against a mock.
        "dead_target_selections": sum(
            code.endswith("_dead") or code.endswith("target_invalid") for code in codes
        ),
        "public_fact_flips": codes.count("result_polarity_conflict"),
        "vote_history_misreads": codes.count("vote_history_misread"),
        # Does the enforcement carry weight, or was the model going to agree?
        "vote_plan_mismatches": len(coordinator.validation.vote_plan_mismatches),
        "timeline_conflicts": sorted(set(conflicts)),
        **reasoning.metrics(),
    }
    if transcript_path is not None:
        transcript = recorder.finalize(controller.get_debug_view())
        transcript_path.write_text(
            json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report["transcript"] = str(transcript_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="どこへ解析用ログを書くか。指定しなければ書かない。",
    )
    args = parser.parse_args()
    report = asyncio.run(run(args.seed, args.transcript))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
