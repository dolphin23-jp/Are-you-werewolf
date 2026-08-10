#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${WEREWOLF_REPO_URL:-https://github.com/dolphin23-jp/Are-you-werewolf.git}"
REPO_DIR="${WEREWOLF_REPO_DIR:-/workspace/Are-you-werewolf}"
PILOT_ROOT="${WEREWOLF_PILOT_ROOT:-/workspace/werewolf-training/pilot-002}"
VENV_DIR="${WEREWOLF_VENV_DIR:-/workspace/.venvs/werewolf-py312}"
EVAL_NAME="${WEREWOLF_EVAL_NAME:-fixed-evaluation-last5-historical-v1}"

RUN_LOG="${PILOT_ROOT}/runpod-last5-fixed-evaluation.log"
PID_FILE="${PILOT_ROOT}/runpod-last5-fixed-evaluation.pid"
DONE_FILE="${PILOT_ROOT}/runpod-last5-fixed-evaluation.done"
FAIL_FILE="${PILOT_ROOT}/runpod-last5-fixed-evaluation.failed"
REPORT_FILE="${PILOT_ROOT}/PILOT-002-LAST5-FIXED-EVALUATION-REPORT.txt"
EVAL_DIR="${PILOT_ROOT}/${EVAL_NAME}"

say() {
  printf '[runpod-pilot002-last5] %s\n' "$*"
}

die() {
  printf '[runpod-pilot002-last5] ERROR: %s\n' "$*" >&2
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
  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
      say "nvidia-smi probe failed; continuing to the authoritative PyTorch CUDA probe."
    fi
  else
    say "nvidia-smi is unavailable in this container; continuing to the authoritative PyTorch CUDA probe."
  fi

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
    raise SystemExit(
        "PyTorch cannot see CUDA on this Pod. Verify that an NVIDIA GPU is attached "
        "to the running Pod and restart/redeploy the Pod if needed."
    )
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"gpu_vram_gb={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
PY
}

validate_pilot() {
  [[ -d /workspace ]] || die "/workspace is missing."
  [[ -f "$PILOT_ROOT/population/population.run.json" ]] || die \
    "Missing $PILOT_ROOT/population/population.run.json. Transfer pilot-002 first."
  [[ -f "$PILOT_ROOT/pool/manifest.json" ]] || die \
    "Missing $PILOT_ROOT/pool/manifest.json. Transfer the complete pilot-002 first."

  (
    cd "$REPO_DIR/backend"
    "$VENV_DIR/bin/python" - "$PILOT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from app.engine.roles import Team
from app.training.torch_pool import TorchPolicyPool

root = Path(sys.argv[1])
state_path = root / "population" / "population.run.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
completed = int(state.get("completed_iterations", -1))
phase = state.get("phase")
print(f"saved_completed_iterations={completed} saved_phase={phase}")
if completed < 4:
    raise SystemExit("pilot-002 has not completed population iteration 4")
if phase != "idle":
    raise SystemExit(
        f"population runner is not idle (phase={phase!r}); do not evaluate a mutating pool"
    )

expected = {
    Team.VILLAGE: (
        "g000078",
        "g000079",
        "g000082",
        "g000085",
        "g000088",
    ),
    Team.WEREWOLF: (
        "g000078",
        "g000080",
        "g000083",
        "g000086",
        "g000089",
    ),
    Team.FOX: (
        "g000078",
        "g000081",
        "g000084",
        "g000087",
        "g000090",
    ),
}

pool = TorchPolicyPool(root / "pool", device="cpu")
for team, wanted in expected.items():
    selected = pool.policy_ids_for_team(team, last=5)
    print(f"{team.value}_last5={','.join(selected)}")
    if selected != wanted:
        raise SystemExit(
            f"unexpected {team.value} last-5 cohort: {selected}; expected {wanted}. "
            "This runner intentionally refuses to drift after newer generations are added."
        )
    for policy_id in wanted:
        entry = pool.get(policy_id)
        checkpoint = root / "pool" / entry.checkpoint
        if not checkpoint.is_file():
            raise SystemExit(f"missing checkpoint for {policy_id}: {checkpoint}")

print("historical_last5_cohort=validated")
PY
  )
}

run_evaluation() {
  local py="$VENV_DIR/bin/python"
  local backend="$REPO_DIR/backend"
  mkdir -p "$EVAL_DIR"

  say "Running/recovering historical last-5 evaluation: 125 profiles x 20 games = 2500 villages"
  (
    cd "$backend"
    "$py" scripts/measure_population_payoffs_torch.py \
      --pool-dir "$PILOT_ROOT/pool" \
      --table "$EVAL_DIR/payoffs.json" \
      --last 5 \
      --games-per-profile 20 \
      --extra-games 0 \
      --seed 5101 \
      --parallel-games 16 \
      --inference-batch-size 64 \
      --device auto
  ) | tee "$EVAL_DIR/measure.log"

  say "Solving meta strategy on the complete historical last-5 cube"
  (
    cd "$backend"
    "$py" scripts/solve_population_meta_torch.py \
      --table "$EVAL_DIR/payoffs.json" \
      --pool-dir "$PILOT_ROOT/pool" \
      --output "$EVAL_DIR/meta.json" \
      --last 5 \
      --temperature 0.25 \
      --iterations 100 \
      --damping 0.5 \
      --device cpu
  ) | tee "$EVAL_DIR/meta.log"

  say "Writing consolidated historical last-5 report"
  (
    cd "$backend"
    "$py" - "$EVAL_DIR" <<'PY'
import itertools
import sys
from pathlib import Path

from app.engine.roles import Team
from app.training.meta_strategy import PopulationMetaStrategy, diagnose_meta_strategy
from app.training.population_payoff import PolicyProfile, PopulationPayoffTable

eval_dir = Path(sys.argv[1])
table = PopulationPayoffTable(eval_dir / "payoffs.json")
strategy = PopulationMetaStrategy.load(eval_dir / "meta.json")

targets = {
    Team.VILLAGE: (
        "g000078",
        "g000079",
        "g000082",
        "g000085",
        "g000088",
    ),
    Team.WEREWOLF: (
        "g000078",
        "g000080",
        "g000083",
        "g000086",
        "g000089",
    ),
    Team.FOX: (
        "g000078",
        "g000081",
        "g000084",
        "g000087",
        "g000090",
    ),
}

profiles = tuple(
    PolicyProfile(village, werewolf, fox)
    for village, werewolf, fox in itertools.product(
        targets[Team.VILLAGE],
        targets[Team.WEREWOLF],
        targets[Team.FOX],
    )
)
records = []
for profile in profiles:
    record = table.get(profile)
    if record is None:
        raise SystemExit(f"missing payoff profile: {profile}")
    if record.games != 20:
        raise SystemExit(
            f"profile {profile} has {record.games} games; controlled historical evaluation requires exactly 20"
        )
    records.append(record)

if len(records) != 125:
    raise SystemExit(f"expected 125 profiles, found {len(records)}")
total_games = sum(record.games for record in records)
if total_games != 2500:
    raise SystemExit(f"expected 2500 total games, found {total_games}")

wins = {
    Team.VILLAGE: sum(record.village_wins for record in records),
    Team.WEREWOLF: sum(record.werewolf_wins for record in records),
    Team.FOX: sum(record.fox_wins for record in records),
}
draws = sum(record.draws for record in records)
if sum(wins.values()) + draws != total_games:
    raise SystemExit("terminal outcome counts do not sum to total games")

def short(policy_id: str) -> str:
    return f"g{int(policy_id[1:])}"

print("===== PILOT-002 HISTORICAL LAST-5 FIXED EVALUATION =====")
for team in Team:
    print(f"{team.value}_targets=" + ",".join(short(item) for item in targets[team]))
print(f"profiles={len(records)} games_per_profile=20 total_games={total_games}")
print("\n===== SIMPLE OUTCOME TOTALS =====")
for team in Team:
    count = wins[team]
    print(f"{team.value}_wins={count}/{total_games} ({100.0 * count / total_games:.2f}%)")
print(f"draws={draws}/{total_games} ({100.0 * draws / total_games:.2f}%)")

print("\n===== POLICY MARGINAL SIMPLE WIN RATES =====")
for team in Team:
    print(f"[{team.value}]")
    for policy_id in targets[team]:
        selected = [
            record
            for record in records
            if record.profile.policy_id(team) == policy_id
        ]
        games = sum(record.games for record in selected)
        policy_wins = sum(record.wins(team) for record in selected)
        if games != 500:
            raise SystemExit(
                f"{policy_id} expected 500 marginal games, found {games}"
            )
        print(
            f"{short(policy_id)} wins={policy_wins}/{games} "
            f"({100.0 * policy_wins / games:.2f}%)"
        )

print("\n===== META WEIGHTS =====")
for team in Team:
    rendered = ", ".join(
        f"{short(item.policy_id)}={item.probability:.6f}"
        for item in strategy.weights(team)
    )
    print(f"{team.value}: {rendered}")

diagnostics = diagnose_meta_strategy(table, strategy)
print("\n===== RESTRICTED DEVIATION DIAGNOSTICS =====")
for team in Team:
    item = diagnostics.for_team(team)
    print(
        f"{team.value}: mixture={item.mixture_payoff:+.6f} "
        f"best={short(item.best_policy_id)} "
        f"best_payoff={item.best_policy_payoff:+.6f} "
        f"deviation_gain={item.deviation_gain:.6f}"
    )
print(f"max_restricted_deviation_gain={diagnostics.max_deviation_gain:.6f}")

weights = {
    team: {item.policy_id: item.probability for item in strategy.weights(team)}
    for team in Team
}
meta_outcomes = {Team.VILLAGE: 0.0, Team.WEREWOLF: 0.0, Team.FOX: 0.0}
meta_draw = 0.0
for record in records:
    probability = (
        weights[Team.VILLAGE][record.profile.village]
        * weights[Team.WEREWOLF][record.profile.werewolf]
        * weights[Team.FOX][record.profile.fox]
    )
    for team in Team:
        meta_outcomes[team] += probability * record.wins(team) / record.games
    meta_draw += probability * record.draws / record.games

print("\n===== META-MIXTURE OUTCOME RATES =====")
for team in Team:
    print(f"{team.value}={100.0 * meta_outcomes[team]:.3f}%")
print(f"draw={100.0 * meta_draw:.3f}%")
print(f"sum={100.0 * (sum(meta_outcomes.values()) + meta_draw):.6f}%")

print("\n===== RAW META SOLVER LOG =====")
meta_log = eval_dir / "meta.log"
print(meta_log.read_text(encoding="utf-8").strip())
print("===== END RAW META SOLVER LOG =====")
print("\n===== END PILOT-002 HISTORICAL LAST-5 FIXED EVALUATION =====")
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
  run_evaluation
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
    say "Use: bash $REPO_DIR/tools/runpod_pilot002_last5.sh status"
    return 0
  fi

  mkdir -p "$PILOT_ROOT"
  rm -f "$DONE_FILE" "$FAIL_FILE"
  touch "$RUN_LOG"
  say "Starting background fixed evaluation. The Network Volume keeps partial payoff results across Pod restarts."
  nohup bash "$REPO_DIR/tools/runpod_pilot002_last5.sh" _worker >>"$RUN_LOG" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  say "Started PID $pid"
  say "Log: $RUN_LOG"
  say "Status: bash $REPO_DIR/tools/runpod_pilot002_last5.sh status"
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
Usage: bash tools/runpod_pilot002_last5.sh [start|status|report|foreground]

  start       Validate pilot-002 and start the 125 x 20 historical fixed evaluation in background.
  status      Show running/completed/failed state and the latest 80 log lines.
  report      Print the consolidated last-5 evaluation report after completion.
  foreground  Run the same workflow in the current terminal (debugging only).

Expected persistent data:
  $PILOT_ROOT

Historical cohort (fixed):
  Village:  g78 g79 g82 g85 g88
  Werewolf: g78 g80 g83 g86 g89
  Fox:      g78 g81 g84 g87 g90
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
