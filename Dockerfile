FROM node:22-bookworm-slim AS frontend-build

WORKDIR /build/frontend
RUN corepack enable && corepack prepare pnpm@10 --activate
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app
COPY backend/ ./backend/
RUN pip install --no-cache-dir ./backend
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

RUN useradd --create-home --uid 10001 werewolf \
    && chown -R werewolf:werewolf /app
USER werewolf
WORKDIR /app/backend

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
