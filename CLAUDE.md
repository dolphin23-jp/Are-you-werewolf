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
