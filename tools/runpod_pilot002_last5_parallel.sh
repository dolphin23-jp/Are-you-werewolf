#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${WEREWOLF_REPO_DIR:-/workspace/werewolf-training/Are-you-werewolf}"
PILOT_ROOT="${WEREWOLF_PILOT_ROOT:-/workspace/werewolf-training/pilot-002}"
PYTHON_BIN="${WEREWOLF_PYTHON:-python3}"
WORKERS="${WEREWOLF_EVAL_WORKERS:-12}"
PARALLEL_GAMES="${WEREWOLF_PARALLEL_GAMES:-16}"
INFERENCE_BATCH_SIZE="${WEREWOLF_INFERENCE_BATCH_SIZE:-64}"
EVAL_DIR="$PILOT_ROOT/fixed-evaluation-last5-historical-v1"
MASTER_TABLE="$EVAL_DIR/payoffs.json"
SHARD_DIR="$EVAL_DIR/shards-$WORKERS"
LOG_DIR="$SHARD_DIR/logs"
RUN_LOG="$PILOT_ROOT/runpod-last5-parallel.log"
PID_FILE="$PILOT_ROOT/runpod-last5-parallel.pid"
DONE_FILE="$PILOT_ROOT/runpod-last5-parallel.done"
FAIL_FILE="$PILOT_ROOT/runpod-last5-parallel.failed"
META_FILE="$EVAL_DIR/meta.json"
REPORT_FILE="$PILOT_ROOT/PILOT-002-LAST5-FIXED-EVALUATION-REPORT.txt"

say() {
  printf '[runpod-pilot002-last5-parallel] %s\n' "$*"
}

die() {
  printf '[runpod-pilot002-last5-parallel] ERROR: %s\n' "$*" >&2
  exit 1
}

backend_dir() {
  printf '%s/backend' "$REPO_DIR"
}

validate_runtime() {
  [[ -d "$REPO_DIR/.git" ]] || die "persistent repo is missing: $REPO_DIR"
  [[ -f "$PILOT_ROOT/population/population.run.json" ]] || die "pilot-002 run state is missing"
  [[ -f "$PILOT_ROOT/pool/manifest.json" ]] || die "pilot-002 pool manifest is missing"
  [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || die "WEREWOLF_EVAL_WORKERS must be a positive integer"
  [[ "$PARALLEL_GAMES" =~ ^[1-9][0-9]*$ ]] || die "WEREWOLF_PARALLEL_GAMES must be positive"
  [[ "$INFERENCE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "WEREWOLF_INFERENCE_BATCH_SIZE must be positive"

  (
    cd "$(backend_dir)"
    "$PYTHON_BIN" - "$PILOT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import torch

from app.engine.roles import Team
from app.training.torch_pool import TorchPolicyPool

print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"gpu={torch.cuda.get_device_name(0)}")

root = Path(sys.argv[1])
state = json.loads((root / "population" / "population.run.json").read_text())
print(f"completed_iterations={state.get('completed_iterations')} phase={state.get('phase')}")
if int(state.get("completed_iterations", -1)) < 4 or state.get("phase") != "idle":
    raise SystemExit("pilot-002 must be iteration-4 complete and idle")

expected = {
    Team.VILLAGE: ("g000078", "g000079", "g000082", "g000085", "g000088"),
    Team.WEREWOLF: ("g000078", "g000080", "g000083", "g000086", "g000089"),
    Team.FOX: ("g000078", "g000081", "g000084", "g000087", "g000090"),
}
pool = TorchPolicyPool(root / "pool", device="cpu")
for team, wanted in expected.items():
    got = pool.policy_ids_for_team(team, last=5)
    print(f"{team.value}={','.join(got)}")
    if got != wanted:
        raise SystemExit(f"unexpected {team.value} last-5 cohort: {got}")
print("historical_last5_cohort=validated")
PY
  )
}

old_evaluator_running() {
  pgrep -af "measure_population_payoffs_torch.py.*$MASTER_TABLE" >/dev/null 2>&1
}

manage_shards() {
  local action="$1"
  (
    cd "$(backend_dir)"
    "$PYTHON_BIN" scripts/manage_population_payoff_shards.py "$action" \
      --pool-dir "$PILOT_ROOT/pool" \
      --table "$MASTER_TABLE" \
      --shard-dir "$SHARD_DIR" \
      --last 5 \
      --shards "$WORKERS" \
      --games-per-profile 20
  )
}

run_worker() {
  local shard_index="$1"
  local shard_table="$SHARD_DIR/shard-$(printf '%02d' "$shard_index").json"
  local shard_log="$LOG_DIR/shard-$(printf '%02d' "$shard_index").log"
  (
    cd "$(backend_dir)"
    "$PYTHON_BIN" scripts/measure_population_payoffs_torch.py \
      --pool-dir "$PILOT_ROOT/pool" \
      --table "$shard_table" \
      --last 5 \
      --games-per-profile 20 \
      --extra-games 0 \
      --seed 5101 \
      --parallel-games "$PARALLEL_GAMES" \
      --inference-batch-size "$INFERENCE_BATCH_SIZE" \
      --shard-count "$WORKERS" \
      --shard-index "$shard_index" \
      --device auto
  ) >"$shard_log" 2>&1
}

run_parallel_evaluation() {
  mkdir -p "$LOG_DIR"
  manage_shards prepare
  say "Launching $WORKERS independent payoff workers on the shared GPU"

  local pids=()
  local indices=()
  local index
  for ((index = 0; index < WORKERS; index++)); do
    run_worker "$index" &
    pids+=("$!")
    indices+=("$index")
  done

  local failed=0
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      say "worker ${indices[$i]} complete"
    else
      say "worker ${indices[$i]} failed; see $LOG_DIR"
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || die "one or more payoff workers failed"

  manage_shards merge

  say "Solving meta strategy"
  (
    cd "$(backend_dir)"
    "$PYTHON_BIN" scripts/solve_population_meta_torch.py \
      --table "$MASTER_TABLE" \
      --pool-dir "$PILOT_ROOT/pool" \
      --output "$META_FILE" \
      --last 5 \
      --temperature 0.25 \
      --iterations 100 \
      --damping 0.5 \
      --device cpu
  ) | tee "$EVAL_DIR/meta.log"

  say "Writing consolidated report"
  (
    cd "$(backend_dir)"
    "$PYTHON_BIN" scripts/report_population_meta_torch.py \
      --table "$MASTER_TABLE" \
      --strategy "$META_FILE" \
      --pool-dir "$PILOT_ROOT/pool" \
      --last 5 \
      --games-per-profile 20
  ) | tee "$REPORT_FILE"
}

worker_main() {
  rm -f "$DONE_FILE" "$FAIL_FILE"
  trap 'rc=$?; rm -f "$PID_FILE"; if [[ $rc -eq 0 ]]; then touch "$DONE_FILE"; else printf "%s\n" "$rc" > "$FAIL_FILE"; fi' EXIT
  validate_runtime
  old_evaluator_running && die "another evaluator is writing the master table; stop it before parallel resume"
  run_parallel_evaluation
  say "Complete. Report: $REPORT_FILE"
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_background() {
  validate_runtime
  old_evaluator_running && die "existing single-process evaluator is still running; stop it first"
  if is_running; then
    say "Already running pid=$(cat "$PID_FILE")"
    return 0
  fi
  mkdir -p "$EVAL_DIR"
  rm -f "$DONE_FILE" "$FAIL_FILE"
  touch "$RUN_LOG"
  nohup bash "$REPO_DIR/tools/runpod_pilot002_last5_parallel.sh" _worker >>"$RUN_LOG" 2>&1 </dev/null &
  printf '%s\n' "$!" > "$PID_FILE"
  say "Started parallel evaluator pid=$! workers=$WORKERS"
  say "Status: bash $REPO_DIR/tools/runpod_pilot002_last5_parallel.sh status"
}

show_status() {
  if is_running; then
    say "RUNNING pid=$(cat "$PID_FILE") workers=$WORKERS"
  elif [[ -f "$DONE_FILE" ]]; then
    say "COMPLETE"
  elif [[ -f "$FAIL_FILE" ]]; then
    say "FAILED exit_code=$(cat "$FAIL_FILE")"
  else
    say "NOT RUNNING"
  fi
  if [[ -d "$SHARD_DIR" ]]; then
    manage_shards status || true
  elif [[ -f "$MASTER_TABLE" ]]; then
    "$PYTHON_BIN" - "$MASTER_TABLE" <<'PY'
import json
import sys
raw = json.load(open(sys.argv[1], encoding="utf-8"))
records = raw.get("records", [])
games = sum(min(int(item["games"]), 20) for item in records)
print(f"master_before_sharding games={games}/2500 ({100.0 * games / 2500:.1f}%) profiles={len(records)}/125")
PY
  fi
  if [[ -f "$RUN_LOG" ]]; then
    printf '\n--- runner log tail ---\n'
    tail -n 40 "$RUN_LOG"
  fi
}

show_report() {
  [[ -f "$REPORT_FILE" ]] || die "report not found yet: $REPORT_FILE"
  cat "$REPORT_FILE"
}

case "${1:-start}" in
  start) start_background ;;
  status) show_status ;;
  report) show_report ;;
  foreground) worker_main ;;
  _worker) worker_main ;;
  *) echo "Usage: $0 [start|status|report|foreground]"; exit 2 ;;
esac
