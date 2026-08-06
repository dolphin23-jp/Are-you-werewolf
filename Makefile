.PHONY: serve build frontend-build backend-dev frontend-dev backend-test frontend-test test lint dry-run expert-eval

# Build the UI and serve everything (API + UI + WebSocket) on one port.
serve:
	bash .devcontainer/start.sh

# Mirror the production frontend stage in Dockerfile.
build: frontend-build

frontend-build:
	cd frontend && pnpm build

backend-dev:
	cd backend && . .venv/bin/activate && uvicorn app.main:app --reload

frontend-dev:
	cd frontend && pnpm dev

backend-test:
	cd backend && . .venv/bin/activate && pytest

frontend-test:
	cd frontend && pnpm test

test: backend-test frontend-test

lint:
	cd backend && . .venv/bin/activate && ruff check app tests scripts && mypy app
	cd frontend && pnpm exec eslint . && pnpm exec tsc -b --noEmit

dry-run:
	cd backend && . .venv/bin/activate && python scripts/dry_run.py --seed 1 --provider mock

expert-eval:
	cd backend && . .venv/bin/activate && python scripts/evaluate_expert_scenarios.py --provider baseline
