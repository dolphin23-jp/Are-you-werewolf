#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${WEREWOLF_REPO_DIR:-/workspace/werewolf-training/Are-you-werewolf}"
PILOT_ROOT="${WEREWOLF_PILOT_ROOT:-/workspace/werewolf-training/pilot-002}"
ITERATION="${WEREWOLF_AUDIT_ITERATION:-7}"
VILLAGE_CHALLENGER="${WEREWOLF_AUDIT_VILLAGE_CHALLENGER:-g000079}"
WEREWOLF_CHALLENGER="${WEREWOLF_AUDIT_WEREWOLF_CHALLENGER:-g000078}"
FOX_CHALLENGER="${WEREWOLF_AUDIT_FOX_CHALLENGER:-g000081}"
GAMES_PER_PROFILE="${WEREWOLF_AUDIT_GAMES_PER_PROFILE:-20}"
SEED="${WEREWOLF_AUDIT_SEED:-17001}"
PARALLEL_GAMES="${WEREWOLF_AUDIT_PARALLEL_GAMES:-16}"
INFERENCE_BATCH_SIZE="${WEREWOLF_AUDIT_INFERENCE_BATCH_SIZE:-64}"

OUTPUT_DIR="${PILOT_ROOT}/population/retention-audit-iteration-$(printf '%04d' "$ITERATION")"
LOG_FILE="${PILOT_ROOT}/retention-audit-iteration-${ITERATION}.log"
PID_FILE="${PILOT_ROOT}/retention-audit-iteration-${ITERATION}.pid"
LOCK_FILE="${PILOT_ROOT}/retention-audit-iteration-${ITERATION}.lock"

say() {
  printf '[retention-audit] %s\n' "$*"
}

die() {
  printf '[retention-audit] ERROR: %s\n' "$*" >&2
  exit 1
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

validate() {
  [[ -d "$REPO_DIR/.git" ]] || die "missing repository: $REPO_DIR"
  [[ -f "$PILOT_ROOT/population/population.run.json" ]] || die \
    "missing pilot population.run.json"
  [[ -f "$PILOT_ROOT/pool/manifest.json" ]] || die "missing pilot pool manifest"
  command -v flock >/dev/null 2>&1 || die "flock is required"

  (
    cd "$REPO_DIR/backend"
    PYTHONPATH=. python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; deploy an NVIDIA GPU Pod")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
  )
}

worker() {
  trap 'rm -f "$PID_FILE"' EXIT
  validate
  mkdir -p "$OUTPUT_DIR"
  cd "$REPO_DIR/backend"
  env PYTHONPATH=. python scripts/audit_population_retention_torch.py \
    --pool-dir "$PILOT_ROOT/pool" \
    --run-dir "$PILOT_ROOT/population" \
    --iteration "$ITERATION" \
    --output-dir "$OUTPUT_DIR" \
    --village-challenger "$VILLAGE_CHALLENGER" \
    --werewolf-challenger "$WEREWOLF_CHALLENGER" \
    --fox-challenger "$FOX_CHALLENGER" \
    --games-per-profile "$GAMES_PER_PROFILE" \
    --seed "$SEED" \
    --parallel-games "$PARALLEL_GAMES" \
    --inference-batch-size "$INFERENCE_BATCH_SIZE" \
    --device cuda
}

start() {
  validate
  if is_running; then
    say "already running pid=$(cat "$PID_FILE")"
    return 0
  fi
  mkdir -p "$PILOT_ROOT"
  touch "$LOG_FILE"
  nohup flock -n "$LOCK_FILE" bash "$0" _worker >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  say "started pid=$pid"
  say "log=$LOG_FILE"
  say "output=$OUTPUT_DIR"
}

status() {
  if is_running; then
    say "running pid=$(cat "$PID_FILE")"
  else
    say "not running"
  fi
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 30 "$LOG_FILE"
  fi
}

report() {
  local report_file="$OUTPUT_DIR/report.txt"
  [[ -f "$report_file" ]] || die "report is not complete: $report_file"
  cat "$report_file"
}

case "${1:-start}" in
  start)
    start
    ;;
  status)
    status
    ;;
  report)
    report
    ;;
  _worker)
    worker
    ;;
  *)
    die "usage: $0 [start|status|report]"
    ;;
esac
