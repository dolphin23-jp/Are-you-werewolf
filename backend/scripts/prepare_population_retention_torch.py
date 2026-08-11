"""Prepare the next Transformer population from a completed retention audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.engine.roles import Team
from app.training.population_payoff import PopulationPayoffTable
from app.training.strategic_retention import (
    StrategicRetentionSelection,
    retention_triggered_by_fixed_audit,
    select_team_population_subset,
)
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_population_research import (
    TorchPopulationResearchState,
    TorchPopulationRunPhase,
    load_torch_population_research_state,
    save_torch_population_research_state,
)

_PLAN_VERSION = 1


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object in {path}")
    return raw


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _team_string_tuple(raw: dict[str, Any], team: Team) -> tuple[str, ...]:
    value = raw.get(team.value)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"invalid {team.value} policy list")
    return tuple(value)


def _diagnostic_payload(selection: StrategicRetentionSelection) -> dict[str, Any]:
    candidates = []
    for candidate in selection.candidates:
        diagnostics = []
        for diagnostic in candidate.held_out_diagnostics:
            raw = asdict(diagnostic)
            raw["team"] = diagnostic.team.value
            diagnostics.append(raw)
        candidates.append(
            {
                "selected_policy_ids": list(candidate.selected_policy_ids),
                "held_out_policy_ids": list(candidate.held_out_policy_ids),
                "max_held_out_gain": candidate.max_held_out_gain,
                "max_held_out_ci95_high": candidate.max_held_out_ci95_high,
                "sum_positive_held_out_gain": candidate.sum_positive_held_out_gain,
                "held_out_diagnostics": diagnostics,
            }
        )
    return {
        "candidate_policy_ids": list(selection.candidate_policy_ids),
        "keep": selection.keep,
        "selected_policy_ids": list(selection.selected_policy_ids),
        "held_out_policy_ids": list(selection.held_out_policy_ids),
        "max_held_out_gain": selection.max_held_out_gain,
        "max_held_out_ci95_high": selection.max_held_out_ci95_high,
        "sum_positive_held_out_gain": selection.sum_positive_held_out_gain,
        "candidates": candidates,
    }


def _parse_oracles(
    summary: dict[str, Any],
    pool: TorchPolicyPool,
) -> dict[Team, str]:
    raw = summary.get("oracle_policy_ids")
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ValueError("iteration summary is missing oracle_policy_ids")
    by_team: dict[Team, str] = {}
    for policy_id in raw:
        entry = pool.get(policy_id)
        team = entry.specialized_team
        if team is None:
            raise ValueError(f"oracle {policy_id} is not faction-specialized")
        if team in by_team:
            raise ValueError(f"iteration has multiple {team.value} oracle policies")
        by_team[team] = policy_id
    if set(by_team) != set(Team):
        raise ValueError("iteration must have exactly one oracle for each faction")
    return by_team


def _generation_sorted(pool: TorchPolicyPool, policy_ids: tuple[str, ...]) -> tuple[str, ...]:
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("next population contains duplicate policy IDs")
    return tuple(sorted(policy_ids, key=lambda policy_id: pool.get(policy_id).generation))


def _validate_source_population(
    *,
    audit_config: dict[str, Any],
    report: dict[str, Any],
    summary: dict[str, Any],
) -> dict[Team, tuple[str, ...]]:
    config_population = audit_config.get("current_population")
    summary_population = summary.get("restricted_population")
    if not isinstance(config_population, dict) or not isinstance(summary_population, dict):
        raise ValueError("audit or iteration summary is missing its restricted population")
    current = {
        team: _team_string_tuple(config_population, team)
        for team in Team
    }
    summary_current = {
        team: _team_string_tuple(summary_population, team)
        for team in Team
    }
    if current != summary_current:
        raise ValueError("audit current population differs from source iteration summary")
    if report.get("challengers") != audit_config.get("challengers"):
        raise ValueError("audit report challengers differ from audit configuration")
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--ci-low-threshold", type=float, default=0.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    state_path = args.run_dir / "population.run.json"
    audit_config_path = args.audit_dir / "audit.config.json"
    report_path = args.audit_dir / "report.json"
    payoff_path = args.audit_dir / "payoffs.json"
    for path in (state_path, audit_config_path, report_path, payoff_path):
        if not path.is_file():
            parser.error(f"missing required file: {path}")

    try:
        state = load_torch_population_research_state(state_path)
        if state.phase is not TorchPopulationRunPhase.IDLE:
            raise ValueError(
                f"population run must be idle before retention planning; phase={state.phase.value}"
            )
        audit_config = _load_json(audit_config_path)
        report = _load_json(report_path)
        source_iteration = int(audit_config["iteration"])
        if int(report.get("iteration", -1)) != source_iteration:
            raise ValueError("audit report iteration differs from audit configuration")
        if state.completed_iterations != source_iteration:
            raise ValueError(
                "retention audit must describe the latest completed iteration; "
                f"audit={source_iteration} run={state.completed_iterations}"
            )

        summary_path = args.run_dir / f"iteration-{source_iteration:04d}" / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"missing source iteration summary: {summary_path}")
        summary = _load_json(summary_path)
        if int(summary.get("iteration", -1)) != source_iteration:
            raise ValueError("source summary has an unexpected iteration number")

        pool = TorchPolicyPool(args.pool_dir, device="cpu")
        expected_generation = summary.get("pool_generation_after")
        if not isinstance(expected_generation, int):
            raise ValueError("source summary is missing pool_generation_after")
        if pool.next_generation != expected_generation:
            raise ValueError(
                "policy pool changed after the audited iteration; "
                f"expected next generation {expected_generation}, got {pool.next_generation}"
            )

        current = _validate_source_population(
            audit_config=audit_config,
            report=report,
            summary=summary,
        )
        challengers_raw = audit_config.get("challengers")
        if not isinstance(challengers_raw, dict):
            raise ValueError("audit configuration is missing challengers")
        challengers = {
            team: str(challengers_raw[team.value])
            for team in Team
        }
        for team, challenger in challengers.items():
            if challenger in current[team]:
                raise ValueError(f"{team.value} challenger is already in source population")
            pool.get(challenger)

        oracles = _parse_oracles(summary, pool)
        capacity = state.config.recent_policies
        if capacity <= 0:
            raise ValueError("restricted population capacity must be positive")

        table = PopulationPayoffTable(payoff_path)
        retention_report = report.get("retention_diagnostics")
        if not isinstance(retention_report, dict):
            raise ValueError("audit report is missing retention diagnostics")
        saved_report = retention_report.get("saved_meta")
        fixed_report = retention_report.get("fixed_meta")
        if not isinstance(saved_report, dict) or not isinstance(fixed_report, dict):
            raise ValueError("audit report is missing saved/fixed retention diagnostics")

        team_plans: dict[Team, dict[str, Any]] = {}
        next_population: dict[Team, tuple[str, ...]] = {}
        for team in Team:
            saved_team = saved_report.get(team.value)
            fixed_team = fixed_report.get(team.value)
            if not isinstance(saved_team, dict) or not isinstance(fixed_team, dict):
                raise ValueError(f"audit report is missing {team.value} diagnostics")
            saved_ci_low = float(saved_team["challenger_gain_ci95_low"])
            fixed_ci_low = float(fixed_team["challenger_gain_ci95_low"])
            trigger = retention_triggered_by_fixed_audit(
                saved_ci95_low=saved_ci_low,
                fixed_ci95_low=fixed_ci_low,
                threshold=args.ci_low_threshold,
            )
            recent = pool.policy_ids_for_team(team, last=capacity)
            if not recent:
                raise ValueError(f"no eligible recent policies for {team.value}")

            team_payload: dict[str, Any] = {
                "challenger_policy_id": challengers[team],
                "saved_meta_ci95_low": saved_ci_low,
                "fixed_meta_ci95_low": fixed_ci_low,
                "triggered": trigger,
                "recent_baseline": list(recent),
                "reserved_newest_oracle": oracles[team],
            }
            if not trigger:
                selected = recent
                team_payload["mode"] = "recent"
            else:
                historical_slots = capacity - 1
                if historical_slots <= 0:
                    raise ValueError(
                        "strategic retention requires capacity for both a newest oracle and history"
                    )
                candidate_history = current[team] + (challengers[team],)
                if historical_slots >= len(candidate_history):
                    raise ValueError(
                        f"{team.value} retention candidate set is too small to compress"
                    )
                selection = select_team_population_subset(
                    table,
                    current_population=current,
                    team=team,
                    candidate_policy_ids=candidate_history,
                    keep=historical_slots,
                    temperature=float(audit_config["meta_temperature"]),
                    iterations=int(audit_config["meta_iterations"]),
                    damping=float(audit_config["meta_damping"]),
                )
                selected = _generation_sorted(
                    pool,
                    selection.selected_policy_ids + (oracles[team],),
                )
                team_payload["mode"] = "strategic_retention"
                team_payload["compression"] = _diagnostic_payload(selection)

            if len(selected) != min(capacity, len(pool.policy_ids_for_team(team))):
                raise ValueError(
                    f"{team.value} next population has unexpected size {len(selected)}"
                )
            if oracles[team] not in selected:
                raise ValueError(f"{team.value} newest oracle was not retained")
            next_population[team] = selected
            team_payload["next_population"] = list(selected)
            team_plans[team] = team_payload

        target_iteration = source_iteration + 1
        payload = {
            "version": _PLAN_VERSION,
            "source_iteration": source_iteration,
            "target_iteration": target_iteration,
            "source_audit": str(args.audit_dir),
            "ci_low_threshold": args.ci_low_threshold,
            "capacity_per_faction": capacity,
            "teams": {
                team.value: team_plans[team]
                for team in Team
            },
            "next_population": {
                team.value: list(next_population[team])
                for team in Team
            },
            "pool_generation_boundary": pool.next_generation,
        }
        plan_path = args.audit_dir / "selection-plan.json"
        _atomic_json(plan_path, payload)

        print("===== STRATEGIC RETENTION PLAN =====")
        print(
            f"source_iteration={source_iteration} target_iteration={target_iteration} "
            f"capacity={capacity} ci_low_threshold={args.ci_low_threshold:+.6f}"
        )
        for team in Team:
            item = team_plans[team]
            line = (
                f"{team.value}: trigger={str(item['triggered']).lower()} "
                f"mode={item['mode']} challenger={item['challenger_policy_id']} "
                f"saved_ci_low={item['saved_meta_ci95_low']:+.6f} "
                f"fixed_ci_low={item['fixed_meta_ci95_low']:+.6f} "
                f"reserved={item['reserved_newest_oracle']}"
            )
            compression = item.get("compression")
            if isinstance(compression, dict):
                line += (
                    f" history={','.join(compression['selected_policy_ids'])} "
                    f"held_out={','.join(compression['held_out_policy_ids'])} "
                    f"max_held_out_gain={compression['max_held_out_gain']:+.6f} "
                    f"max_ci_high={compression['max_held_out_ci95_high']:+.6f}"
                )
            line += f" next={','.join(item['next_population'])}"
            print(line)
        print(f"plan_json={plan_path}")

        if args.apply:
            target_dir = args.run_dir / f"iteration-{target_iteration:04d}"
            conflicting = tuple(
                target_dir / name
                for name in ("meta.json", "oracle.run.npz", "summary.json")
                if (target_dir / name).exists()
            )
            if conflicting:
                raise ValueError(
                    "target iteration already has research artifacts: "
                    + ", ".join(str(path) for path in conflicting)
                )
            provenance_path = target_dir / "retention-selection.json"
            _atomic_json(provenance_path, payload)
            prepared = TorchPopulationResearchState(
                config=state.config,
                completed_iterations=state.completed_iterations,
                phase=TorchPopulationRunPhase.MEASURE,
                village_policy_ids=next_population[Team.VILLAGE],
                werewolf_policy_ids=next_population[Team.WEREWOLF],
                fox_policy_ids=next_population[Team.FOX],
                iteration_pool_generation=pool.next_generation,
            )
            save_torch_population_research_state(prepared, state_path)
            print(
                f"applied=true phase={prepared.phase.value} "
                f"pool_generation={prepared.iteration_pool_generation} "
                f"provenance={provenance_path}"
            )
        else:
            print("applied=false")
        print("===== END STRATEGIC RETENTION PLAN =====")
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
