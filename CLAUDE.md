# Are you werewolf — repo notes

- `backend/`: Python 3.12 + FastAPI. Engine (`app/engine/`) is pure/LLM-agnostic;
  AI layer (`app/ai/`) talks to it the same way a human via `app/api/` does.
- `frontend/`: Vite + React + TypeScript, strict mode, no `@ts-nocheck`.
- Default LLM provider is `mock` (`WEREWOLF_LLM_PROVIDER=mock`) — zero cost, zero
  network. Tests always run against mock; never wire a live provider into CI.
- The real provider is `anthropic` (`WEREWOLF_LLM_PROVIDER=anthropic`, default model
  `claude-opus-5-5`); `luna` stays selectable but dormant. `WEREWOLF_ANTHROPIC_API_KEY`
  lives only as a GitHub Actions secret for the manually dispatched eval workflows or
  as a Codespaces secret — never in `ci.yml`, never in a committed file. The app reads
  only `WEREWOLF_ANTHROPIC_*`, never the SDK's ambient `ANTHROPIC_*` variables.
  A `claude-*` id typed into `WEREWOLF_LLM_PROVIDER` selects `anthropic` and becomes
  the model when `WEREWOLF_ANTHROPIC_MODEL` is unset; a different explicit model
  fails at startup.
- Keep every comment in `backend/.env.example` on its own line, never
  `KEY=value  # comment`: `docker run --env-file` keeps the comment as part of the
  value. Settings treats a whitespace-then-`#` access password as empty, but nothing
  else is protected that way.
- Keep every schema a v3 game or an evaluation run sends within Anthropic's
  structured-output limits (24 optional / 16 union-typed parameters per request,
  nested included); `backend/tests/ai/test_schema_complexity.py` enforces it. v3
  sends `ChatDiscussionOutput`. v2 and legacy still send `DiscussionOutput` (36
  optional parameters): Claude refuses its grammar once per provider, and from then
  on those discussion turns are parsed from the schema embedded in the prompt
  (`json_object`, not grammar-enforced), so a v2/legacy arm on Claude is not
  parsed under the same conditions as v3.
- Do not word prompts or wire fields as the model's 思考 / thought / reasoning: a
  request to write out its thinking can be refused as `reasoning_extraction`, and
  refusal fallbacks do not retry that.
- Read `docs/approach-reset-2026-10.md` §11 before changing the provider.
- Run backend tests: `cd backend && source .venv/bin/activate && pytest`.
- Run backend lint/types: `ruff check app tests scripts && mypy app --exclude '^app/training/torch_.*\.py$'`
  (the CI form; the torch modules are type-checked in the separate transformer job).
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
