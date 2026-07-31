.PHONY: serve backend-dev frontend-dev backend-test frontend-test test lint dry-run

# Build the UI and serve everything (API + UI + WebSocket) on one port.
serve:
	bash .devcontainer/start.sh

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
