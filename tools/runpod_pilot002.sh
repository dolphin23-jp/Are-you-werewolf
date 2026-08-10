#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${WEREWOLF_REPO_URL:-https://github.com/dolphin23-jp/Are-you-werewolf.git}"
REPO_DIR="${WEREWOLF_REPO_DIR:-/workspace/Are-you-werewolf}"
PILOT_ROOT="${WEREWOLF_PILOT_ROOT:-/workspace/werewolf-training/pilot-002}"
VENV_DIR="${WEREWOLF_VENV_DIR:-/workspace/.venvs/werewolf-py312}"
TARGET_ITERATIONS="${WEREWOLF_TARGET_ITERATIONS:-4}"
EVAL_NAME="${WEREWOLF_EVAL_NAME:-fixed-evaluation-last3-v2}"

RUN_LOG="${PILOT_ROOT}/runpod-iterations-3-4.log"
PID_FILE="${PILOT_ROOT}/runpod-iterations-3-4.pid"
DONE_FILE="${PILOT_ROOT}/runpod-iterations-3-4.done"
FAIL_FILE="${PILOT_ROOT}/runpod-iterations-3-4.failed"
REPORT_FILE="${PILOT_ROOT}/PILOT-002-ITERATIONS-3-4-REPORT.txt"
EVENT_LOG="${PILOT_ROOT}/population/events-iterations-3-4-runpod.log"
EVAL_DIR="${PILOT_ROOT}/${EVAL_NAME}"

say() {
  printf '[runpod-pilot002] %s\n' "$*"
}

die() {
  printf '[runpod-pilot002] ERROR: %s\n' "$*" >&2
  exit 1
}

ensure_repo() {
  mkdir -p "$(dirname "$REPO_DIR")"
  if [[ -d "$REPO_DIR/.git" ]]; then
    say "Updating repository at $REPO_DIR"
    git -C "$REPO_DIR" fetch --depth 1 origin main
    git -C "$REPO_DIR" checkout -q main
    git -C "$REPO_DIR" reset --hard origin/main
  else
    say "Cloning repository into $REPO_DIR"
    rm -rf "$REPO_DIR"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
  fi
}

ensure_runtime() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is missing; deploy an NVIDIA GPU Pod."
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv for an isolated Python 3.12 environment"
    command -v curl >/dev/null 2>&1 || die "curl is required to install uv."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    say "Creating persistent Python 3.12 environment at $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    uv python install 3.12
    uv venv --python 3.12 --seed "$VENV_DIR"
  fi

  say "Installing/updating werewolf RL dependencies"
  (
    cd "$REPO_DIR/backend"
    uv pip install --python "$VENV_DIR/bin/python" -e ".[rl,transformer]"
  )

  "$VENV_DIR/bin/python" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA on this Pod")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"gpu_vram_gb={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
PY
}

validate_pilot() {
  [[ -d /workspace ]] || die "/workspace is missing."
  [[ -f "$PILOT_ROOT/population/population.run.json" ]] || die \
    "Missing $PILOT_ROOT/population/population.run.json. Transfer the existing pilot-002 first."
  [[ -f "$PILOT_ROOT/pool/manifest.json" ]] || die \
    "Missing $PILOT_ROOT/pool/manifest.json. Transfer the complete pilot-002 folder first."

  "$VENV_DIR/bin/python" - "$PILOT_ROOT/population/population.run.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = json.loads(path.read_text())
completed = int(state.get("completed_iterations", -1))
phase = state.get("phase")
print(f"saved_completed_iterations={completed} saved_phase={phase}")
if completed < 2:
    raise SystemExit("pilot-002 is older than the expected completed iteration 2 state")
PY
}

run_experiment() {
  local py="$VENV_DIR/bin/python"
  local backend="$REPO_DIR/backend"
  mkdir -p "$EVAL_DIR" "$(dirname "$EVENT_LOG")"

  say "Resuming population research to iteration $TARGET_ITERATIONS"
  (
    cd "$backend"
    "$py" scripts/run_population_iterations_torch.py \
      --pool-dir "$PILOT_ROOT/pool" \
      --run-dir "$PILOT_ROOT/population" \
      --iterations "$TARGET_ITERATIONS" \
      --resume \
      --device auto
  ) | tee -a "$EVENT_LOG"

  say "Running/recovering fixed evaluation: 27 profiles x 20 games"
  (
    cd "$backend"
    "$py" scripts/measure_population_payoffs_torch.py \
      --pool-dir "$PILOT_ROOT/pool" \
      --table "$EVAL_DIR/payoffs.json" \
      --last 3 \
      --games-per-profile 20 \
      --extra-games 0 \
      --seed 4101 \
      --parallel-games 16 \
      --inference-batch-size 64 \
      --device auto
  ) | tee "$EVAL_DIR/measure.log"

  say "Solving fixed-policy meta strategy"
  (
    cd "$backend"
    "$py" scripts/solve_population_meta_torch.py \
      --table "$EVAL_DIR/payoffs.json" \
      --pool-dir "$PILOT_ROOT/pool" \
      --output "$EVAL_DIR/meta.json" \
      --last 3 \
      --temperature 0.25 \
      --iterations 100 \
      --damping 0.5 \
      --device cpu
  ) | tee "$EVAL_DIR/meta.log"

  say "Writing consolidated report"
  (
    cd "$backend"
    "$py" - "$PILOT_ROOT" "$EVAL_DIR" "$EVENT_LOG" <<'PY'
import json
import sys
from pathlib import Path

from app.engine.roles import Team
from app.training.torch_pool import TorchPolicyPool

root = Path(sys.argv[1])
eval_dir = Path(sys.argv[2])
event_log = Path(sys.argv[3])
pool = TorchPolicyPool(root / "pool", device="cpu")
targets = {
    team.value: list(pool.policy_ids_for_team(team, last=3))
    for team in Team
}

print("===== PILOT-002 ITERATIONS 3-4 REPORT =====")
print("targets:", json.dumps(targets, ensure_ascii=False))

for iteration in (3, 4):
    summary_path = root / "population" / f"iteration-{iteration:04d}" / "summary.json"
    print(f"\n===== POPULATION SUMMARY {iteration} =====")
    if summary_path.exists():
        print(json.dumps(json.loads(summary_path.read_text()), ensure_ascii=False, indent=2))
    else:
        print(f"MISSING: {summary_path}")
    print(f"===== END POPULATION SUMMARY {iteration} =====")

print("\n===== FIXED EVALUATION META =====")
meta_log = eval_dir / "meta.log"
print(meta_log.read_text().strip() if meta_log.exists() else f"MISSING: {meta_log}")
print("===== END FIXED EVALUATION META =====")

print("\n===== FIXED EVALUATION PROFILE PAYOFFS =====")
measure_log = eval_dir / "measure.log"
print(measure_log.read_text().strip() if measure_log.exists() else f"MISSING: {measure_log}")
print("===== END FIXED EVALUATION PROFILE PAYOFFS =====")

print("\n===== LEARNING EVENTS ITERATIONS 3-4 (LAST 80) =====")
if event_log.exists():
    lines = [line for line in event_log.read_text().splitlines() if line.strip()]
    print("\n".join(lines[-80:]))
else:
    print(f"MISSING: {event_log}")
print("===== END LEARNING EVENTS =====")
print("\n===== END PILOT-002 ITERATIONS 3-4 REPORT =====")
PY
  ) | tee "$REPORT_FILE"
}

worker() {
  mkdir -p "$PILOT_ROOT"
  rm -f "$DONE_FILE" "$FAIL_FILE"
  trap 'rc=$?; rm -f "$PID_FILE"; if [[ $rc -eq 0 ]]; then touch "$DONE_FILE"; else printf "%s\n" "$rc" > "$FAIL_FILE"; fi' EXIT
  ensure_repo
  ensure_runtime
  validate_pilot
  run_experiment
  say "Complete. Report: $REPORT_FILE"
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_background() {
  ensure_repo
  [[ -f "$PILOT_ROOT/population/population.run.json" ]] || die \
    "pilot-002 has not been transferred to $PILOT_ROOT yet."
  if is_running; then
    say "Already running with PID $(cat "$PID_FILE")."
    say "Use: bash $REPO_DIR/tools/runpod_pilot002.sh status"
    return 0
  fi

  mkdir -p "$PILOT_ROOT"
  rm -f "$DONE_FILE" "$FAIL_FILE"
  touch "$RUN_LOG"
  say "Starting background worker. Closing Jupyter/SSH will not stop this shell process while the Pod remains running."
  nohup bash "$REPO_DIR/tools/runpod_pilot002.sh" _worker >>"$RUN_LOG" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  say "Started PID $pid"
  say "Log: $RUN_LOG"
  say "Status: bash $REPO_DIR/tools/runpod_pilot002.sh status"
}

show_status() {
  if is_running; then
    say "RUNNING pid=$(cat "$PID_FILE")"
  elif [[ -f "$DONE_FILE" ]]; then
    say "COMPLETE"
  elif [[ -f "$FAIL_FILE" ]]; then
    say "FAILED exit_code=$(cat "$FAIL_FILE")"
  else
    say "NOT RUNNING"
  fi
  if [[ -f "$RUN_LOG" ]]; then
    printf '\n--- last 80 log lines ---\n'
    tail -n 80 "$RUN_LOG"
  fi
}

show_report() {
  [[ -f "$REPORT_FILE" ]] || die "Report not found yet: $REPORT_FILE"
  cat "$REPORT_FILE"
}

usage() {
  cat <<EOF
Usage: bash tools/runpod_pilot002.sh [start|status|report|foreground]

  start       Update repo and start iteration 3-4 + fixed evaluation in background.
  status      Show running/completed/failed state and the latest 80 log lines.
  report      Print the final report after completion.
  foreground  Run the same workflow in the current terminal (debugging only).

Expected persistent data:
  $PILOT_ROOT
EOF
}

case "${1:-start}" in
  start) start_background ;;
  status) show_status ;;
  report) show_report ;;
  foreground) worker ;;
  _worker) worker ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
