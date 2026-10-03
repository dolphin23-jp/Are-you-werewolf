# Are you werewolf — repo notes

- `backend/`: Python 3.12 + FastAPI. Engine (`app/engine/`) is pure/LLM-agnostic;
  AI layer (`app/ai/`) talks to it the same way a human via `app/api/` does.
- `frontend/`: Vite + React + TypeScript, strict mode, no `@ts-nocheck`.
- Default LLM provider is `mock` (`WEREWOLF_LLM_PROVIDER=mock`) — zero cost, zero
  network. Tests always run against mock; never wire a live provider into CI.
- Run backend tests: `cd backend && source .venv/bin/activate && pytest`.
- Run backend lint/types: `ruff check app tests scripts && mypy app`.
- Run frontend tests: `cd frontend && pnpm test`. Lint/types: `pnpm exec eslint . && pnpm exec tsc -b --noEmit`.
- See `docs/architecture.md` for the design philosophy carried over from the
  prior implementation and what was deliberately changed.
- `docs/approach-reset-2026-10.md` decides the current priorities: the v3
  engine (code guarantees facts and logic, the model reasons and speaks) is
  the path to completion; the self-play RL path and the real-log training
  pipeline are frozen, not deleted. Read it before adding to either.
- Before changing external-log ingestion, possible-world reasoning, evidence
  weighting, vote evaluation, or expert-scenario data, read
  `docs/real_game_log_reasoning_plan.md`. It is the design contract for keeping
  facts, hard logical possibility, soft world weighting, and faction utility
  separate. Do not scale a log dataset until one complete game passes its
  `training-ready` gate.
