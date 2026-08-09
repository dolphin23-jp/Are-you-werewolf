"""Crash-resumable Transformer empirical-population research iterations.

This orchestration layer composes existing sparse-reward training primitives. It
freezes the restricted policy cube at each iteration boundary, measures missing
terminal payoffs, solves the empirical meta-strategy, and runs the resumable
Village/Werewolf/Fox oracle cycle. It does not alter game rules or policy inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.meta_strategy import (
    PolicyWeight,
    PopulationMetaDiagnostics,
    PopulationMetaStrategy,
    diagnose_meta_strategy,
    solve_logit_response_mixture,
)
from app.training.population_payoff import (
    PolicyProfile,
    PopulationPayoffTable,
    ProfilePayoff,
)
from app.training.torch_oracle_cycle import (
    TorchOracleRunProgress,
    finalize_torch_oracle,
    start_torch_oracle_cycle,
    train_torch_oracle_subbatch,
    validate_torch_oracle_pool_boundary,
)
from app.training.torch_oracle_run_state import (
    load_torch_oracle_run_state,
    save_torch_oracle_run_state,
)
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_population import (
    TorchProfileEvaluationRequest,
    evaluate_torch_policy_profiles,
)
from app.training.torch_trainer import TorchPPOConfig

_RUN_STATE_VERSION = 1


class TorchPopulationRunPhase(StrEnum):
    IDLE = "idle"
    MEASURE = "measure"
    ORACLE = "oracle"


@dataclass(frozen=True)
class TorchPopulationResearchConfig:
    """Settings frozen for all iterations in one empirical-population run."""

    recent_policies: int = 3
    games_per_profile: int = 3
    extra_games: int = 0
    uncertainty_prior: float = 0.5
    oracle_episodes: int = 4
    oracle_batch_size: int = 4
    evaluation_seed: int = 30_000
    oracle_seed: int = 40_000
    opponent_seed: int = 1
    max_discussion_ticks: int = 8
    max_parallel_games: int = 8
    max_inference_batch_size: int | None = None
    meta_temperature: float = 0.25
    meta_iterations: int = 100
    meta_damping: float = 0.5
    ppo_config: TorchPPOConfig = TorchPPOConfig()

    def __post_init__(self) -> None:
        if self.recent_policies <= 0:
            raise ValueError("recent_policies must be positive")
        if self.games_per_profile <= 0:
            raise ValueError("games_per_profile must be positive")
        if self.extra_games < 0:
            raise ValueError("extra_games cannot be negative")
        if self.uncertainty_prior <= 0:
            raise ValueError("uncertainty_prior must be positive")
        if self.oracle_episodes <= 0:
            raise ValueError("oracle_episodes must be positive")
        if self.oracle_batch_size <= 0:
            raise ValueError("oracle_batch_size must be positive")
        if self.max_discussion_ticks < 0:
            raise ValueError("max_discussion_ticks cannot be negative")
        if self.max_parallel_games <= 0:
            raise ValueError("max_parallel_games must be positive")
        if (
            self.max_inference_batch_size is not None
            and self.max_inference_batch_size <= 0
        ):
            raise ValueError("max_inference_batch_size must be positive")
        if self.meta_temperature <= 0:
            raise ValueError("meta_temperature must be positive")
        if self.meta_iterations <= 0:
            raise ValueError("meta_iterations must be positive")
        if not 0 < self.meta_damping <= 1:
            raise ValueError("meta_damping must be in (0, 1]")


@dataclass(frozen=True)
class AdaptivePayoffPending:
    """One adaptive payoff game committed before its environment is executed."""

    profile: PolicyProfile
    games_before: int
    seed: int

    def __post_init__(self) -> None:
        if self.games_before < 0:
            raise ValueError("games_before cannot be negative")


@dataclass(frozen=True)
class TorchPopulationResearchState:
    """Durable outer state around payoff files and the oracle NPZ."""

    config: TorchPopulationResearchConfig
    completed_iterations: int = 0
    phase: TorchPopulationRunPhase = TorchPopulationRunPhase.IDLE
    village_policy_ids: tuple[str, ...] = ()
    werewolf_policy_ids: tuple[str, ...] = ()
    fox_policy_ids: tuple[str, ...] = ()
    iteration_pool_generation: int | None = None
    adaptive_games_completed: int = 0
    adaptive_pending: AdaptivePayoffPending | None = None
    meta_strategy: PopulationMetaStrategy | None = None

    def __post_init__(self) -> None:
        if self.completed_iterations < 0:
            raise ValueError("completed_iterations cannot be negative")
        active_sets = (
            self.village_policy_ids,
            self.werewolf_policy_ids,
            self.fox_policy_ids,
        )
        if self.phase is TorchPopulationRunPhase.IDLE:
            if any(active_sets):
                raise ValueError("idle research state cannot retain policy sets")
            if self.iteration_pool_generation is not None:
                raise ValueError("idle research state cannot retain a pool boundary")
            if self.adaptive_games_completed != 0 or self.adaptive_pending is not None:
                raise ValueError("idle research state cannot retain adaptive progress")
            if self.meta_strategy is not None:
                raise ValueError("idle research state cannot retain a meta-strategy")
            return

        if not all(active_sets):
            raise ValueError("active research state requires all faction policy sets")
        if self.iteration_pool_generation is None or self.iteration_pool_generation < 0:
            raise ValueError("active research state requires a pool generation boundary")
        if not 0 <= self.adaptive_games_completed <= self.config.extra_games:
            raise ValueError("adaptive_games_completed is out of range")
        if self.adaptive_pending is not None and (
            self.phase is not TorchPopulationRunPhase.MEASURE
            or self.adaptive_games_completed >= self.config.extra_games
        ):
            raise ValueError("adaptive pending game is inconsistent with run phase")
        if self.phase is TorchPopulationRunPhase.MEASURE:
            if self.meta_strategy is not None:
                raise ValueError("measurement phase cannot retain a meta-strategy")
        elif self.phase is TorchPopulationRunPhase.ORACLE:
            if self.adaptive_games_completed != self.config.extra_games:
                raise ValueError("oracle phase requires completed adaptive evaluation")
            if self.adaptive_pending is not None:
                raise ValueError("oracle phase cannot retain an adaptive pending game")
            if self.meta_strategy is None:
                raise ValueError("oracle phase requires a frozen meta-strategy")

    @property
    def iteration_number(self) -> int:
        return self.completed_iterations + 1

    @property
    def profiles(self) -> tuple[PolicyProfile, ...]:
        if self.phase is TorchPopulationRunPhase.IDLE:
            return ()
        return tuple(
            PolicyProfile(village, werewolf, fox)
            for village in self.village_policy_ids
            for werewolf in self.werewolf_policy_ids
            for fox in self.fox_policy_ids
        )


@dataclass(frozen=True)
class TorchPopulationResearchEvent:
    kind: str
    iteration: int
    message: str


class TorchPopulationResearchRun:
    """Advance a persistent Transformer population run one durable step at a time."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        pool: TorchPolicyPool,
        run_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.player_specs = player_specs
        self.pool = pool
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.state_path = self.run_dir / "population.run.json"
        self.payoff_path = self.run_dir / "payoffs.json"
        self.table = PopulationPayoffTable(self.payoff_path)
        self.state: TorchPopulationResearchState | None = None

    def start(self, config: TorchPopulationResearchConfig) -> TorchPopulationResearchState:
        if self.state_path.exists():
            raise ValueError("research run-state already exists; use resume")
        if self.payoff_path.exists():
            raise ValueError("fresh research run cannot reuse an existing payoff table")
        if not self.pool.entries:
            raise ValueError("population research requires a non-empty policy pool")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state = TorchPopulationResearchState(config=config)
        self._save_state()
        return self.state

    def resume(self) -> TorchPopulationResearchState:
        if not self.state_path.exists():
            raise ValueError("research run-state does not exist")
        self.state = load_torch_population_research_state(self.state_path)
        self.table = PopulationPayoffTable(self.payoff_path)
        self._validate_active_pool()
        return self.state

    def run_until(
        self,
        target_iterations: int,
        *,
        on_event: Callable[[TorchPopulationResearchEvent], None] | None = None,
    ) -> TorchPopulationResearchState:
        if target_iterations <= 0:
            raise ValueError("target_iterations must be positive")
        state = self._require_state()
        if target_iterations < state.completed_iterations:
            raise ValueError("target_iterations cannot be below completed progress")
        while state.completed_iterations < target_iterations:
            event = self.step()
            state = self._require_state()
            if on_event is not None:
                on_event(event)
        return state

    def step(self) -> TorchPopulationResearchEvent:
        state = self._require_state()
        if state.phase is TorchPopulationRunPhase.IDLE:
            return self._begin_iteration()
        if state.phase is TorchPopulationRunPhase.MEASURE:
            return self._measurement_step()
        if state.phase is TorchPopulationRunPhase.ORACLE:
            return self._oracle_step()
        raise RuntimeError(f"unsupported research phase {state.phase}")

    def _begin_iteration(self) -> TorchPopulationResearchEvent:
        state = self._require_state()
        if state.phase is not TorchPopulationRunPhase.IDLE:
            raise RuntimeError("cannot begin an already active iteration")
        config = state.config
        village = self.pool.policy_ids_for_team(
            Team.VILLAGE,
            last=config.recent_policies,
        )
        werewolf = self.pool.policy_ids_for_team(
            Team.WEREWOLF,
            last=config.recent_policies,
        )
        fox = self.pool.policy_ids_for_team(Team.FOX, last=config.recent_policies)
        if not village or not werewolf or not fox:
            raise ValueError("each faction must have at least one eligible policy")
        self.state = replace(
            state,
            phase=TorchPopulationRunPhase.MEASURE,
            village_policy_ids=village,
            werewolf_policy_ids=werewolf,
            fox_policy_ids=fox,
            iteration_pool_generation=self.pool.next_generation,
        )
        self._save_state()
        return TorchPopulationResearchEvent(
            kind="iteration_started",
            iteration=self.state.iteration_number,
            message=(
                f"frozen village={','.join(village)} werewolf={','.join(werewolf)} "
                f"fox={','.join(fox)}"
            ),
        )

    def _measurement_step(self) -> TorchPopulationResearchEvent:
        state = self._require_state()
        self._validate_measurement_pool()
        missing = self._missing_base_requests(state)
        if missing:
            stats = evaluate_torch_policy_profiles(
                self.player_specs,
                self.pool,
                self.table,
                missing,
                max_discussion_ticks=state.config.max_discussion_ticks,
                max_parallel_games=state.config.max_parallel_games,
                max_inference_batch_size=state.config.max_inference_batch_size,
            )
            return TorchPopulationResearchEvent(
                kind="payoff_batch",
                iteration=state.iteration_number,
                message=(
                    f"games={stats.games} chunks={stats.rollout_chunks} "
                    f"games_s={stats.games_per_second:.2f} "
                    f"checkpoint_loads={stats.checkpoint_loads} "
                    f"inference_mean_batch={stats.mean_inference_batch:.1f} "
                    f"inference_max_batch={stats.max_inference_batch}"
                ),
            )

        if state.adaptive_pending is not None:
            return self._run_or_reconcile_adaptive(state)

        if state.adaptive_games_completed < state.config.extra_games:
            pending = self._select_adaptive_pending(state)
            self.state = replace(state, adaptive_pending=pending)
            self._save_state()
            return TorchPopulationResearchEvent(
                kind="adaptive_selected",
                iteration=state.iteration_number,
                message=(
                    f"profile={_profile_text(pending.profile)} "
                    f"games_before={pending.games_before} seed={pending.seed}"
                ),
            )

        return self._solve_and_start_oracle(state)

    def _run_or_reconcile_adaptive(
        self,
        state: TorchPopulationResearchState,
    ) -> TorchPopulationResearchEvent:
        pending = state.adaptive_pending
        if pending is None:
            raise RuntimeError("adaptive recovery requires pending state")
        record = self.table.get(pending.profile)
        if record is None:
            raise ValueError("adaptive pending profile is missing from payoff table")
        if record.games == pending.games_before:
            stats = evaluate_torch_policy_profiles(
                self.player_specs,
                self.pool,
                self.table,
                (
                    TorchProfileEvaluationRequest(
                        pending.profile,
                        (pending.seed,),
                    ),
                ),
                max_discussion_ticks=state.config.max_discussion_ticks,
                max_parallel_games=state.config.max_parallel_games,
                max_inference_batch_size=state.config.max_inference_batch_size,
            )
            recovered = False
            detail = f"games_s={stats.games_per_second:.2f}"
        elif record.games == pending.games_before + 1:
            recovered = True
            detail = "recovered_table_commit=true"
        else:
            raise ValueError("adaptive payoff table advanced beyond pending boundary")

        updated = self.table.get(pending.profile)
        if updated is None or updated.games != pending.games_before + 1:
            raise RuntimeError("adaptive payoff game did not reach expected boundary")
        self.state = replace(
            state,
            adaptive_games_completed=state.adaptive_games_completed + 1,
            adaptive_pending=None,
        )
        self._save_state()
        return TorchPopulationResearchEvent(
            kind="adaptive_completed",
            iteration=state.iteration_number,
            message=(
                f"profile={_profile_text(pending.profile)} "
                f"adaptive={self.state.adaptive_games_completed}/{state.config.extra_games} "
                f"{detail} recovered={str(recovered).lower()}"
            ),
        )

    def _solve_and_start_oracle(
        self,
        state: TorchPopulationResearchState,
    ) -> TorchPopulationResearchEvent:
        if state.adaptive_games_completed != state.config.extra_games:
            raise RuntimeError("cannot solve meta-strategy before adaptive evaluation")
        strategy = solve_logit_response_mixture(
            self.table,
            village=state.village_policy_ids,
            werewolf=state.werewolf_policy_ids,
            fox=state.fox_policy_ids,
            temperature=state.config.meta_temperature,
            iterations=state.config.meta_iterations,
            damping=state.config.meta_damping,
        )
        diagnostics = diagnose_meta_strategy(self.table, strategy)
        meta_path = self._meta_path(state.iteration_number)
        strategy.save(meta_path)
        oracle_path = self._oracle_path(state.iteration_number)

        if oracle_path.exists():
            loop, progress = load_torch_oracle_run_state(
                oracle_path,
                self.player_specs,
                self.pool,
                device=self.device,
            )
            if loop is None:
                raise ValueError("uncommitted oracle startup cannot already be complete")
            if loop.opponent_strategy != strategy:
                raise ValueError("existing oracle state does not match solved strategy")
            if (
                progress.team_index != 0
                or progress.completed_episodes != 0
                or progress.completed_policy_ids
            ):
                raise ValueError("existing oracle state advanced before outer phase commit")
            self._validate_oracle_progress(state, progress)
            validate_torch_oracle_pool_boundary(self.pool, progress)
        else:
            expected_generation = state.iteration_pool_generation
            if expected_generation is None or self.pool.next_generation != expected_generation:
                raise ValueError("pool changed before oracle startup")
            oracle_seed = self._oracle_seed(state)
            loop, progress = start_torch_oracle_cycle(
                self.player_specs,
                self.pool,
                strategy,
                episodes_per_oracle=state.config.oracle_episodes,
                oracle_batch_size=state.config.oracle_batch_size,
                base_seed=oracle_seed,
                opponent_seed=self._opponent_seed(state),
                trainer_seed=oracle_seed,
                ppo_config=state.config.ppo_config,
                max_discussion_ticks=state.config.max_discussion_ticks,
                max_parallel_games=state.config.max_parallel_games,
                max_inference_batch_size=state.config.max_inference_batch_size,
            )
            save_torch_oracle_run_state(loop, progress, oracle_path)

        self.state = replace(
            state,
            phase=TorchPopulationRunPhase.ORACLE,
            meta_strategy=strategy,
        )
        self._save_state()
        return TorchPopulationResearchEvent(
            kind="meta_solved",
            iteration=state.iteration_number,
            message=(
                f"restricted_max_deviation_gain={diagnostics.max_deviation_gain:.4f} "
                f"oracle_state={oracle_path}"
            ),
        )

    def _oracle_step(self) -> TorchPopulationResearchEvent:
        state = self._require_state()
        oracle_path = self._oracle_path(state.iteration_number)
        if not oracle_path.exists():
            raise ValueError("oracle phase is missing its run-state")
        loop, progress = load_torch_oracle_run_state(
            oracle_path,
            self.player_specs,
            self.pool,
            device=self.device,
        )
        self._validate_oracle_progress(state, progress)
        validate_torch_oracle_pool_boundary(self.pool, progress)
        if loop is not None and loop.opponent_strategy != state.meta_strategy:
            raise ValueError("oracle opponent strategy differs from outer run-state")

        if progress.is_complete:
            return self._finish_iteration(state, progress)
        if loop is None:
            raise ValueError("active oracle is missing its learner state")
        team = progress.active_team
        if team is None:
            raise RuntimeError("active oracle is missing its faction")

        if progress.next_batch_episodes > 0:
            stats, progress = train_torch_oracle_subbatch(loop, progress)
            save_torch_oracle_run_state(loop, progress, oracle_path)
            return TorchPopulationResearchEvent(
                kind="oracle_batch",
                iteration=state.iteration_number,
                message=(
                    f"team={team.value} episodes={progress.completed_episodes}/"
                    f"{progress.episodes_per_oracle} record="
                    f"{progress.active_wins}-{progress.active_losses}-"
                    f"{progress.active_draws} rollout_eps_s="
                    f"{stats.rollout_episodes_per_second:.2f} "
                    f"kl={stats.update.mean_approx_kl:.6f} "
                    f"entropy={stats.update.mean_path_entropy:.4f} "
                    f"value_ev={stats.update.rollout_value_explained_variance:.4f}"
                ),
            )

        next_loop, next_progress, entry = finalize_torch_oracle(
            self.player_specs,
            self.pool,
            loop,
            progress,
        )
        save_torch_oracle_run_state(next_loop, next_progress, oracle_path)
        return TorchPopulationResearchEvent(
            kind="oracle_finalized",
            iteration=state.iteration_number,
            message=(
                f"team={team.value} parent={progress.active_parent_policy_id} "
                f"saved={entry.policy_id}"
            ),
        )

    def _finish_iteration(
        self,
        state: TorchPopulationResearchState,
        progress: TorchOracleRunProgress,
    ) -> TorchPopulationResearchEvent:
        if not progress.is_complete:
            raise RuntimeError("cannot finish an incomplete oracle cycle")
        if progress.teams != tuple(Team) or len(progress.completed_policy_ids) != len(Team):
            raise ValueError("completed oracle cycle does not contain all factions")
        expected_start = state.iteration_pool_generation
        if expected_start is None:
            raise RuntimeError("active iteration lost its pool boundary")
        if progress.next_pool_generation != expected_start + len(Team):
            raise ValueError("oracle cycle finished at an unexpected pool generation")
        if self.pool.next_generation != progress.next_pool_generation:
            raise ValueError("policy pool does not match completed oracle state")
        strategy = state.meta_strategy
        if strategy is None:
            raise RuntimeError("oracle phase lost its meta-strategy")
        diagnostics = diagnose_meta_strategy(self.table, strategy)
        self._write_iteration_summary(state, diagnostics, progress)

        completed = state.completed_iterations + 1
        self.state = TorchPopulationResearchState(
            config=state.config,
            completed_iterations=completed,
        )
        self._save_state()
        return TorchPopulationResearchEvent(
            kind="iteration_completed",
            iteration=state.iteration_number,
            message=(
                f"oracles={','.join(progress.completed_policy_ids)} "
                f"restricted_max_deviation_gain={diagnostics.max_deviation_gain:.4f}"
            ),
        )

    def _missing_base_requests(
        self,
        state: TorchPopulationResearchState,
    ) -> tuple[TorchProfileEvaluationRequest, ...]:
        requests: list[TorchProfileEvaluationRequest] = []
        base_seed = self._evaluation_seed(state)
        for profile in state.profiles:
            existing = self.table.get(profile)
            games = existing.games if existing is not None else 0
            missing = max(0, state.config.games_per_profile - games)
            if missing == 0:
                continue
            start = _profile_seed(base_seed, profile) + games
            requests.append(
                TorchProfileEvaluationRequest(
                    profile,
                    tuple(range(start, start + missing)),
                )
            )
        return tuple(requests)

    def _select_adaptive_pending(
        self,
        state: TorchPopulationResearchState,
    ) -> AdaptivePayoffPending:
        selected = max(
            state.profiles,
            key=lambda profile: (
                self._require_record(profile).max_posterior_payoff_std(
                    prior=state.config.uncertainty_prior
                ),
                -self._require_record(profile).games,
                profile,
            ),
        )
        record = self._require_record(selected)
        seed = _profile_seed(self._evaluation_seed(state), selected) + record.games
        return AdaptivePayoffPending(selected, record.games, seed)

    def _validate_active_pool(self) -> None:
        state = self._require_state()
        if state.phase is TorchPopulationRunPhase.IDLE:
            return
        for team, policy_ids in (
            (Team.VILLAGE, state.village_policy_ids),
            (Team.WEREWOLF, state.werewolf_policy_ids),
            (Team.FOX, state.fox_policy_ids),
        ):
            eligible = {entry.policy_id for entry in self.pool.entries_for_team(team)}
            missing = set(policy_ids) - eligible
            if missing:
                raise ValueError(
                    f"frozen {team.value} policies are missing from pool: "
                    + ",".join(sorted(missing))
                )
        if state.phase is TorchPopulationRunPhase.MEASURE:
            self._validate_measurement_pool()
        else:
            oracle_path = self._oracle_path(state.iteration_number)
            if not oracle_path.exists():
                raise ValueError("oracle phase is missing its run-state")
            _loop, progress = load_torch_oracle_run_state(
                oracle_path,
                self.player_specs,
                self.pool,
                device=self.device,
            )
            self._validate_oracle_progress(state, progress)
            validate_torch_oracle_pool_boundary(self.pool, progress)

    def _validate_measurement_pool(self) -> None:
        state = self._require_state()
        expected = state.iteration_pool_generation
        if expected is None or self.pool.next_generation != expected:
            raise ValueError("policy pool changed during frozen payoff measurement")

    def _validate_oracle_progress(
        self,
        state: TorchPopulationResearchState,
        progress: TorchOracleRunProgress,
    ) -> None:
        expected_generation = state.iteration_pool_generation
        if expected_generation is None:
            raise RuntimeError("oracle phase lost its starting pool boundary")
        if progress.teams != tuple(Team):
            raise ValueError("oracle faction order differs from research run")
        if progress.episodes_per_oracle != state.config.oracle_episodes:
            raise ValueError("oracle episode target differs from research run")
        if progress.oracle_batch_size != state.config.oracle_batch_size:
            raise ValueError("oracle batch size differs from research run")
        if progress.base_seed != self._oracle_seed(state):
            raise ValueError("oracle seed base differs from research run")
        if progress.opponent_seed != self._opponent_seed(state):
            raise ValueError("oracle opponent seed differs from research run")
        if progress.trainer_seed != self._oracle_seed(state):
            raise ValueError("oracle trainer seed differs from research run")
        if progress.next_pool_generation != expected_generation + progress.team_index:
            raise ValueError("oracle pool generation differs from research run")

    def _write_iteration_summary(
        self,
        state: TorchPopulationResearchState,
        diagnostics: PopulationMetaDiagnostics,
        progress: TorchOracleRunProgress,
    ) -> None:
        strategy = state.meta_strategy
        if strategy is None:
            raise RuntimeError("cannot summarize iteration without meta-strategy")
        payload = {
            "version": 1,
            "iteration": state.iteration_number,
            "restricted_population": {
                "village": list(state.village_policy_ids),
                "werewolf": list(state.werewolf_policy_ids),
                "fox": list(state.fox_policy_ids),
            },
            "profile_count": len(state.profiles),
            "meta_strategy": _strategy_payload(strategy),
            "restricted_diagnostics": {
                team.value: asdict(diagnostics.for_team(team))
                for team in Team
            },
            "max_restricted_deviation_gain": diagnostics.max_deviation_gain,
            "oracle_policy_ids": list(progress.completed_policy_ids),
            "pool_generation_after": progress.next_pool_generation,
        }
        summary_path = self._summary_path(state.iteration_number)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(summary_path)

    def _save_state(self) -> None:
        state = self._require_state()
        save_torch_population_research_state(state, self.state_path)

    def _require_state(self) -> TorchPopulationResearchState:
        if self.state is None:
            raise RuntimeError("research run has not been started or resumed")
        return self.state

    def _require_record(self, profile: PolicyProfile) -> ProfilePayoff:
        record = self.table.get(profile)
        if record is None:
            raise RuntimeError("profile measurement did not produce a payoff record")
        return record

    def _iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / f"iteration-{iteration:04d}"

    def _meta_path(self, iteration: int) -> Path:
        return self._iteration_dir(iteration) / "meta.json"

    def _oracle_path(self, iteration: int) -> Path:
        return self._iteration_dir(iteration) / "oracle.run.npz"

    def _summary_path(self, iteration: int) -> Path:
        return self._iteration_dir(iteration) / "summary.json"

    @staticmethod
    def _evaluation_seed(state: TorchPopulationResearchState) -> int:
        return state.config.evaluation_seed + state.completed_iterations * 2_000_000_000

    @staticmethod
    def _oracle_seed(state: TorchPopulationResearchState) -> int:
        return state.config.oracle_seed + state.completed_iterations * 1_000_000

    @staticmethod
    def _opponent_seed(state: TorchPopulationResearchState) -> int:
        return state.config.opponent_seed + state.completed_iterations * 100


def save_torch_population_research_state(
    state: TorchPopulationResearchState,
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _RUN_STATE_VERSION,
        "config": {
            **asdict(state.config),
            "ppo_config": asdict(state.config.ppo_config),
        },
        "completed_iterations": state.completed_iterations,
        "phase": state.phase.value,
        "village_policy_ids": list(state.village_policy_ids),
        "werewolf_policy_ids": list(state.werewolf_policy_ids),
        "fox_policy_ids": list(state.fox_policy_ids),
        "iteration_pool_generation": state.iteration_pool_generation,
        "adaptive_games_completed": state.adaptive_games_completed,
        "adaptive_pending": _pending_payload(state.adaptive_pending),
        "meta_strategy": _strategy_payload(state.meta_strategy),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_torch_population_research_state(
    path: str | Path,
) -> TorchPopulationResearchState:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != _RUN_STATE_VERSION:
        raise ValueError("unsupported Transformer population research run-state")
    config_raw = _mapping(raw, "config")
    ppo_raw = _mapping(config_raw, "ppo_config")
    config_args = dict(config_raw)
    config_args.pop("ppo_config", None)
    config = TorchPopulationResearchConfig(
        **config_args,
        ppo_config=TorchPPOConfig(**ppo_raw),
    )
    try:
        phase = TorchPopulationRunPhase(str(raw["phase"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid Transformer population research phase") from exc
    return TorchPopulationResearchState(
        config=config,
        completed_iterations=_required_int(raw, "completed_iterations"),
        phase=phase,
        village_policy_ids=_string_tuple(raw, "village_policy_ids"),
        werewolf_policy_ids=_string_tuple(raw, "werewolf_policy_ids"),
        fox_policy_ids=_string_tuple(raw, "fox_policy_ids"),
        iteration_pool_generation=_optional_int(raw, "iteration_pool_generation"),
        adaptive_games_completed=_required_int(raw, "adaptive_games_completed"),
        adaptive_pending=_pending_from_payload(raw.get("adaptive_pending")),
        meta_strategy=_strategy_from_payload(raw.get("meta_strategy")),
    )


def _profile_seed(base_seed: int, profile: PolicyProfile) -> int:
    payload = f"{profile.village}|{profile.werewolf}|{profile.fox}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return base_seed + int.from_bytes(digest, "big") % 1_000_000_000


def _profile_text(profile: PolicyProfile) -> str:
    return f"{profile.village}/{profile.werewolf}/{profile.fox}"


def _strategy_payload(strategy: PopulationMetaStrategy | None) -> dict[str, Any] | None:
    if strategy is None:
        return None
    return {
        team.value: [
            {"policy_id": item.policy_id, "probability": item.probability}
            for item in strategy.weights(team)
        ]
        for team in Team
    }


def _strategy_from_payload(raw: Any) -> PopulationMetaStrategy | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("invalid research meta-strategy payload")
    by_team: dict[Team, tuple[PolicyWeight, ...]] = {}
    for team in Team:
        items = raw.get(team.value)
        if not isinstance(items, list) or not items:
            raise ValueError("invalid research meta-strategy weights")
        weights: list[PolicyWeight] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("invalid research meta-strategy weight")
            policy_id = item.get("policy_id")
            probability = item.get("probability")
            if not isinstance(policy_id, str) or not isinstance(probability, (int, float)):
                raise ValueError("invalid research meta-strategy weight fields")
            weights.append(PolicyWeight(policy_id, float(probability)))
        by_team[team] = tuple(weights)
    return PopulationMetaStrategy(
        village=by_team[Team.VILLAGE],
        werewolf=by_team[Team.WEREWOLF],
        fox=by_team[Team.FOX],
    )


def _pending_payload(pending: AdaptivePayoffPending | None) -> dict[str, Any] | None:
    if pending is None:
        return None
    return {
        "profile": asdict(pending.profile),
        "games_before": pending.games_before,
        "seed": pending.seed,
    }


def _pending_from_payload(raw: Any) -> AdaptivePayoffPending | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("invalid adaptive pending payload")
    profile_raw = raw.get("profile")
    if not isinstance(profile_raw, dict):
        raise ValueError("adaptive pending profile is invalid")
    try:
        profile = PolicyProfile(
            village=str(profile_raw["village"]),
            werewolf=str(profile_raw["werewolf"]),
            fox=str(profile_raw["fox"]),
        )
    except KeyError as exc:
        raise ValueError("adaptive pending profile is incomplete") from exc
    return AdaptivePayoffPending(
        profile=profile,
        games_before=_required_int(raw, "games_before"),
        seed=_required_int(raw, "seed"),
    )


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"research run-state {key} must be an object")
    return value


def _required_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ValueError(f"research run-state {key} must be an integer")
    return value


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"research run-state {key} must be an integer or null")
    return value


def _string_tuple(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"research run-state {key} must be a string list")
    return tuple(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, Team):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")
