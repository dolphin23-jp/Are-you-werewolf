#!/usr/bin/env python3
"""Plays N full games and reports on AI quality and running cost.

    # ハーネスの動作確認(通信なし・費用ゼロ)
    python scripts/evaluate.py --games 2 --provider mock

    # 実際の gpt-5.6-luna で評価
    python scripts/evaluate.py --games 3 --provider luna --judge \
        --price-in 0.15 --price-out 0.60

Writes to --out (default `eval-out/`):
  report.md              集計レポート
  summary.json           機械可読な集計
  transcript-seedN.md    目視確認用の対戦記録
  transcript-seedN.json  生データ

Cost warning: one 17-player game is on the order of a few hundred LLM
calls, so start with --games 1 against a real endpoint and read the
reported token counts before scaling up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.coordinator import AICoordinator  # noqa: E402
from app.ai.metrics import MetricsCollector  # noqa: E402
from app.ai.provider.factory import build_llm_provider  # noqa: E402
from app.config import Settings  # noqa: E402
from app.engine.game import GameController, PlayerSpec  # noqa: E402
from app.engine.phases import Phase  # noqa: E402
from app.engine.roles import RoleName  # noqa: E402
from app.eval.analyzers import AnalysisResult, analyze  # noqa: E402
from app.eval.judge import judge_transcript  # noqa: E402
from app.eval.report import render_report, render_transcript  # noqa: E402
from app.eval.transcript import GameTranscript, TranscriptRecorder  # noqa: E402

HUMAN_ID = "p0"
MAX_LOOPS = 200

AI_NAMES = [
    "アカリ",
    "ハルト",
    "ユイ",
    "ソウタ",
    "ミオ",
    "レン",
    "サクラ",
    "カイト",
    "ノゾミ",
    "リク",
    "ツムギ",
    "ダイキ",
    "ホノカ",
    "シオン",
    "アオイ",
    "ケント",
]


def _make_specs() -> list[PlayerSpec]:
    specs = [PlayerSpec(player_id=HUMAN_ID, name="あなた", is_human=True)]
    specs += [
        PlayerSpec(player_id=f"p{i}", name=AI_NAMES[i - 1], is_human=False) for i in range(1, 17)
    ]
    return specs


def _drive_human_night(controller: GameController, rng: random.Random) -> None:
    """Stands in for the single human seat so the AI sees a realistic
    16-AI-plus-1-human table rather than an all-AI one."""
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
        prey = [pid for pid in candidates if state.players[pid].role != RoleName.WEREWOLF]
        if prey:
            controller.submit_night_action(HUMAN_ID, "attack", rng.choice(prey))


async def play_one_game(seed: int, settings: Settings, metrics: MetricsCollector) -> GameTranscript:
    provider = build_llm_provider(settings, seed=seed, metrics=metrics)
    specs = _make_specs()
    controller = GameController(session_id=f"eval-{seed}", player_specs=specs, seed=seed)
    ai_ids = [s.player_id for s in specs if not s.is_human]
    recorder = TranscriptRecorder()
    coordinator = AICoordinator(controller.state, ai_ids, provider, seed=seed, recorder=recorder)
    session = SimpleNamespace(
        controller=controller,
        human_id=HUMAN_ID,
        coordinator=coordinator,
        discussion_lock=asyncio.Lock(),
    )
    rng = random.Random(seed)

    controller.start_game()
    for _ in range(MAX_LOOPS):
        phase = controller.state.phase
        if phase == Phase.GAME_OVER:
            break
        if phase == Phase.NIGHT:
            _drive_human_night(controller, rng)
            await coordinator.run_night_phase(session)
        elif phase == Phase.DAWN:
            controller.start_discussion()
        elif phase == Phase.DISCUSSION:
            if controller.state.players[HUMAN_ID].alive:
                controller.chat(HUMAN_ID, "よろしくお願いします。", "public")
            await coordinator.run_discussion_round(session)
            controller.end_discussion()
        elif phase in (Phase.VOTING, Phase.RUNOFF):
            if controller.state.players[HUMAN_ID].alive:
                candidates = [pid for pid in controller.state.alive_ids() if pid != HUMAN_ID]
                if candidates:
                    controller.vote(HUMAN_ID, rng.choice(candidates))
            await coordinator.generate_all_votes(session)
        elif phase == Phase.VOTE_RESULT:
            controller.start_night()
        else:
            raise RuntimeError(f"unexpected phase {phase}")

    return recorder.finalize(controller.get_debug_view())


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1)
    # No default: passing one explicitly to Settings would override
    # WEREWOLF_LLM_PROVIDER from the environment, so a CLI *default* would
    # silently beat an env var the operator deliberately set. Settings
    # supplies the real default (mock) when this is omitted.
    parser.add_argument("--provider", choices=["mock", "luna"], default=None)
    parser.add_argument("--seed", type=int, default=1, help="最初のゲームのシード")
    parser.add_argument("--out", type=Path, default=Path("eval-out"))
    parser.add_argument("--judge", action="store_true", help="日本語/人格をLLMで採点する(追加費用)")
    parser.add_argument("--judge-model", default=None, help="判定に使うモデル(既定: 本体と同じ)")
    parser.add_argument("--price-in", type=float, default=0.0, help="入力100万トークンあたり単価")
    parser.add_argument("--price-out", type=float, default=0.0, help="出力100万トークンあたり単価")
    args = parser.parse_args()

    overrides = {"werewolf_llm_provider": args.provider} if args.provider else {}
    settings = Settings(**overrides)
    provider_name = settings.werewolf_llm_provider
    print(f"==> LLM プロバイダ: {provider_name}", flush=True)
    metrics = MetricsCollector()
    args.out.mkdir(parents=True, exist_ok=True)

    games: list[tuple[GameTranscript, AnalysisResult]] = []
    for offset in range(args.games):
        seed = args.seed + offset
        print(f"==> game {offset + 1}/{args.games} (seed={seed})", flush=True)
        transcript = await play_one_game(seed, settings, metrics)
        analysis = analyze(transcript)
        games.append((transcript, analysis))

        (args.out / f"transcript-seed{seed}.json").write_text(
            json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.out / f"transcript-seed{seed}.md").write_text(
            render_transcript(transcript), encoding="utf-8"
        )
        winner = transcript.final_state.get("winner")
        print(f"    winner={winner} findings={len(analysis.findings)}", flush=True)

    judge_summary: dict[str, Any] | None = None
    judge_detail: dict[str, Any] = {}
    if args.judge:
        print("==> LLM判定を実行中", flush=True)
        judge_settings = settings
        if args.judge_model:
            judge_settings = Settings(**overrides, luna_model=args.judge_model)
        # Judge calls are deliberately kept out of `metrics`, so the game's
        # own cost/latency numbers are not polluted by evaluation overhead.
        judge_provider = build_llm_provider(judge_settings, seed=args.seed)
        result = await judge_transcript(games[0][0], judge_provider)
        judge_summary = result.summary
        judge_detail = result.per_player

    metrics_summary = metrics.summary(args.price_in, args.price_out)
    report = render_report(
        games=games,
        metrics_summary=metrics_summary,
        judge_summary=judge_summary,
        provider=games[0][0].provider if games else "unknown",
        model=settings.luna_model if provider_name == "luna" else "(mock)",
    )
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "provider": provider_name,
                "model": settings.luna_model if provider_name == "luna" else "(mock)",
                "games": args.games,
                "metrics": metrics_summary,
                "judge": {"summary": judge_summary, "per_player": judge_detail},
                "findings": [
                    {
                        "seed": t.seed,
                        "winner": t.final_state.get("winner"),
                        "counts": {
                            check: a.count(check) for check in {f.check for f in a.findings}
                        },
                        "stats": a.stats,
                    }
                    for t, a in games
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(report)
    print(f"==> 出力先: {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
