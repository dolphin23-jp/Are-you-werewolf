#!/usr/bin/env python3
"""Headless full-game runner: plays one game end-to-end (mock or live
provider) and prints a readable transcript. Useful for eyeballing AI
behavior quality during development.

Usage:
    python scripts/dry_run.py --seed 1 --provider mock
    python scripts/dry_run.py --seed 1 --provider luna   # requires LUNA_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.coordinator import AICoordinator  # noqa: E402
from app.ai.provider.factory import build_llm_provider  # noqa: E402
from app.config import Settings  # noqa: E402
from app.engine.game import GameController, PlayerSpec  # noqa: E402
from app.engine.phases import Phase  # noqa: E402
from app.engine.roles import ROLE_DEFINITIONS, RoleName  # noqa: E402

HUMAN_ID = "p0"


def _drive_human_night_action(controller: GameController, rng: random.Random) -> None:
    """The real product always has exactly one human seat; a headless dry
    run stands in for that seat with simple random choices so the AI
    coordinator sees the same "1 human + 16 AI" shape as production."""
    state = controller.state
    human = state.players[HUMAN_ID]
    if not human.alive:
        return
    candidates = [pid for pid in state.alive_ids() if pid != HUMAN_ID]
    if not candidates:
        return
    if state.day == 0:
        if human.role == RoleName.SEER:
            controller.submit_night_action(HUMAN_ID, "divine", rng.choice(candidates))
        return
    if human.role == RoleName.SEER:
        controller.submit_night_action(HUMAN_ID, "divine", rng.choice(candidates))
    elif human.role == RoleName.HUNTER:
        controller.submit_night_action(HUMAN_ID, "guard", rng.choice(candidates))
    elif human.role == RoleName.WEREWOLF and controller.alpha_wolf_id == HUMAN_ID:
        non_wolves = [pid for pid in candidates if state.players[pid].role != RoleName.WEREWOLF]
        if non_wolves:
            controller.submit_night_action(HUMAN_ID, "attack", rng.choice(non_wolves))


def _drive_human_vote(controller: GameController, rng: random.Random) -> None:
    state = controller.state
    human = state.players[HUMAN_ID]
    if not human.alive:
        return
    # votable_ids, not alive_ids: in a runoff only the tied players are legal
    # targets, and the engine now rejects anyone else.
    candidates = state.votable_ids(HUMAN_ID)
    if candidates:
        controller.vote(HUMAN_ID, rng.choice(candidates))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    # No default: an explicit kwarg to Settings outranks the environment, so a
    # CLI *default* of "mock" would silently beat a WEREWOLF_LLM_PROVIDER the
    # operator deliberately set. Settings supplies the real default instead.
    parser.add_argument("--provider", choices=["mock", "luna"], default=None)
    parser.add_argument("--max-loops", type=int, default=200)
    args = parser.parse_args()

    overrides = {"werewolf_llm_provider": args.provider} if args.provider else {}
    settings = Settings(**overrides)
    provider = build_llm_provider(settings, seed=args.seed)

    specs = [PlayerSpec(player_id=f"p{i}", name=f"P{i}", is_human=(i == 0)) for i in range(17)]
    controller = GameController(session_id="dry-run", player_specs=specs, seed=args.seed)
    ai_ids = [s.player_id for s in specs if not s.is_human]
    coordinator = AICoordinator(controller.state, ai_ids, provider, seed=args.seed)
    session = SimpleNamespace(
        controller=controller,
        human_id=HUMAN_ID,
        coordinator=coordinator,
        discussion_lock=asyncio.Lock(),
    )

    print(f"=== seed={args.seed} provider={settings.werewolf_llm_provider} ===")
    for p in specs:
        role = ROLE_DEFINITIONS[controller.state.players[p.player_id].role]
        print(f"  {p.player_id} {p.name}: {role.label_ja}")

    controller.start_game()

    rng = random.Random(args.seed)
    seen_chat = 0
    reached_game_over = False
    for _ in range(args.max_loops):
        state = controller.state
        phase = state.phase

        if phase == Phase.GAME_OVER:
            reached_game_over = True
            break

        if phase == Phase.NIGHT:
            _drive_human_night_action(controller, rng)
            await coordinator.run_night_phase(session)
        elif phase == Phase.DAWN:
            print(f"--- 夜明け (day {state.day}) ---")
            for record in [d for d in state.death_records if d.day == state.day]:
                name = state.players[record.player_id].name
                print(f"  {name} が死亡 ({record.cause.value})")
            controller.start_discussion()
        elif phase == Phase.DISCUSSION:
            if state.players[HUMAN_ID].alive:
                controller.chat(HUMAN_ID, "よろしくお願いします。", "public")
            await coordinator.run_discussion_round(session)
            controller.end_discussion()
        elif phase in (Phase.VOTING, Phase.RUNOFF):
            _drive_human_vote(controller, rng)
            await coordinator.generate_all_votes(session)
        elif phase == Phase.VOTE_RESULT:
            for msg in state.chat_log[seen_chat:]:
                print(f"  [{msg.channel.value}] {state.players[msg.author_id].name}: {msg.content}")
            seen_chat = len(state.chat_log)
            controller.start_night()
        else:
            raise RuntimeError(f"unexpected phase {phase}")

    if not reached_game_over:
        phase = controller.state.phase
        print(f"!!! did not reach GAME_OVER within {args.max_loops} loops (phase={phase})")

    print("=== GAME OVER ===")
    print(f"winner={controller.state.winner} is_draw={controller.state.is_draw}")
    print(f"reason={controller.state.victory_reason}")


if __name__ == "__main__":
    asyncio.run(main())
