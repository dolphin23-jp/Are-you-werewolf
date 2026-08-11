#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${WEREWOLF_REPO_DIR:-/workspace/werewolf-training/Are-you-werewolf}"
PILOT_ROOT="${WEREWOLF_PILOT_ROOT:-/workspace/werewolf-training/pilot-002}"
ITERATION="${WEREWOLF_RETENTION_ITERATION:-7}"
CI_LOW_THRESHOLD="${WEREWOLF_RETENTION_CI_LOW_THRESHOLD:-0.0}"
AUDIT_DIR="${PILOT_ROOT}/population/retention-audit-iteration-$(printf '%04d' "$ITERATION")"

say() {
  printf '[strategic-retention] %s\n' "$*"
}

die() {
  printf '[strategic-retention] ERROR: %s\n' "$*" >&2
  exit 1
}

validate() {
  [[ -d "$REPO_DIR/.git" ]] || die "missing repository: $REPO_DIR"
  [[ -f "$PILOT_ROOT/population/population.run.json" ]] || die \
    "missing pilot population.run.json"
  [[ -f "$PILOT_ROOT/pool/manifest.json" ]] || die "missing pilot pool manifest"
  [[ -f "$AUDIT_DIR/audit.config.json" ]] || die "missing audit configuration"
  [[ -f "$AUDIT_DIR/payoffs.json" ]] || die "missing audit payoff table"
  [[ -f "$AUDIT_DIR/report.json" ]] || die "missing completed audit report"
}

run_plan() {
  local apply="${1:-false}"
  validate
  cd "$REPO_DIR/backend"
  local args=(
    scripts/prepare_population_retention_torch.py
    --pool-dir "$PILOT_ROOT/pool"
    --run-dir "$PILOT_ROOT/population"
    --audit-dir "$AUDIT_DIR"
    --ci-low-threshold "$CI_LOW_THRESHOLD"
  )
  if [[ "$apply" == "true" ]]; then
    args+=(--apply)
  fi
  env PYTHONPATH=. python "${args[@]}"
}

case "${1:-plan}" in
  plan)
    say "planning from iteration $ITERATION audit without changing run-state"
    run_plan false
    ;;
  apply)
    say "applying audited population selection at the idle boundary"
    run_plan true
    ;;
  *)
    die "usage: $0 [plan|apply]"
    ;;
esac
