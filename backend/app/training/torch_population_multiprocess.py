"""Multiprocess population rollout with a single accelerator inference owner."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from queue import Empty
from time import perf_counter
from typing import Any, cast

import torch

from app.engine.game import PlayerSpec
from app.engine.roles import Team
from app.training.encoding import EncodedPolicyObservation
from app.training.policy_contract import PolicyLogits
from app.training.population_payoff import (
    PolicyProfile,
    PopulationPayoffResult,
    PopulationPayoffTable,
)
from app.training.torch_policy import TorchTransformerPolicy
from app.training.torch_pool import TorchPolicyPool
from app.training.torch_population import (
    TorchPopulationEvaluationStats,
    TorchProfileEvaluationRequest,
)
from app.training.torch_vectorized import (
    TorchRolloutInferenceStats,
    TorchVectorizedEpisodeCollector,
    _InferenceRequest,
    _PreparedRequest,
)


@dataclass(frozen=True)
class _MultiprocessGame:
    index: int
    profile: PolicyProfile
    seed: int


@dataclass(frozen=True)
class _WorkerJob:
    job_id: int
    games: tuple[_MultiprocessGame, ...]
    max_discussion_ticks: int
    max_inference_batch_size: int | None
    temperature: float


@dataclass(frozen=True)
class _InferenceMessage:
    worker_id: int
    request_id: int
    policy_id: str
    observations: tuple[EncodedPolicyObservation, ...]


@dataclass(frozen=True)
class _InferenceReply:
    request_id: int
    logits: tuple[PolicyLogits, ...]


@dataclass(frozen=True)
class _WorkerGameResult:
    index: int
    profile: PolicyProfile
    winner: Team | None
    is_draw: bool
    days: int


@dataclass(frozen=True)
class _WorkerResult:
    worker_id: int
    job_id: int
    games: tuple[_WorkerGameResult, ...]
    error: str | None = None


class _PolicyProxy:
    """CPU-only identity handle accepted by the existing collector at runtime."""

    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id

    def eval(self) -> _PolicyProxy:
        return self


class _RemoteInferenceCollector(TorchVectorizedEpisodeCollector):
    """Reuse game progression while routing inference through parent queues."""

    def __init__(
        self,
        player_specs: list[PlayerSpec],
        default_policy_id: str,
        *,
        worker_id: int,
        inference_queue: Any,
        response_queue: Any,
        max_discussion_ticks: int,
        max_inference_batch_size: int | None,
        temperature: float,
    ) -> None:
        self._worker_id = worker_id
        self._inference_queue = inference_queue
        self._response_queue = response_queue
        self._next_request_id = 0
        self._proxies: dict[str, _PolicyProxy] = {}
        super().__init__(
            player_specs,
            self.policy_model(default_policy_id),
            max_discussion_ticks=max_discussion_ticks,
            max_inference_batch_size=max_inference_batch_size,
            temperature=temperature,
        )

    def policy_model(self, policy_id: str) -> TorchTransformerPolicy:
        proxy = self._proxies.get(policy_id)
        if proxy is None:
            proxy = _PolicyProxy(policy_id)
            self._proxies[policy_id] = proxy
        return cast(TorchTransformerPolicy, proxy)

    def _infer(
        self,
        requests: list[_InferenceRequest],
    ) -> tuple[_PreparedRequest, ...]:
        if not requests:
            return ()
        self.inference_stats.record_pending(len(requests))

        grouped: dict[str, list[tuple[int, _InferenceRequest]]] = {}
        for request_index, request in enumerate(requests):
            proxy = cast(_PolicyProxy, request.model)
            grouped.setdefault(proxy.policy_id, []).append((request_index, request))

        prepared: list[_PreparedRequest | None] = [None] * len(requests)
        pending: dict[int, list[tuple[int, _InferenceRequest]]] = {}

        # Submit every policy microbatch for this logical decision before waiting.
        # This lets the parent combine matching policies across workers instead of
        # serializing each worker behind a request/reply round trip per policy.
        for policy_id, indexed_requests in grouped.items():
            limit = self.max_inference_batch_size or len(indexed_requests)
            for start in range(0, len(indexed_requests), limit):
                batch = indexed_requests[start : start + limit]
                self.inference_stats.record_inference(len(batch))
                request_id = self._next_request_id
                self._next_request_id += 1
                pending[request_id] = batch
                self._inference_queue.put(
                    _InferenceMessage(
                        worker_id=self._worker_id,
                        request_id=request_id,
                        policy_id=policy_id,
                        observations=tuple(request.encoded for _, request in batch),
                    )
                )

        # Replies can arrive out of request order because the parent regroups all
        # workers' requests by policy before forwarding them through CUDA.
        while pending:
            reply = self._response_queue.get()
            if not isinstance(reply, _InferenceReply):
                raise RuntimeError("multiprocess inference returned an invalid reply")
            batch = pending.pop(reply.request_id, None)
            if batch is None:
                raise RuntimeError("multiprocess inference returned an unexpected reply")
            if len(reply.logits) != len(batch):
                raise RuntimeError("multiprocess inference returned the wrong batch length")
            for (original_index, request), logits in zip(
                batch,
                reply.logits,
                strict=True,
            ):
                prepared[original_index] = _PreparedRequest(request=request, logits=logits)

        if any(item is None for item in prepared):
            raise RuntimeError("multiprocess inference did not prepare every request")
        return tuple(cast(_PreparedRequest, item) for item in prepared)


def evaluate_torch_policy_profiles_multiprocess(
    player_specs: list[PlayerSpec],
    pool: TorchPolicyPool,
    table: PopulationPayoffTable,
    requests: tuple[TorchProfileEvaluationRequest, ...],
    *,
    worker_count: int = 4,
    max_discussion_ticks: int = 8,
    max_parallel_games: int = 32,
    max_inference_batch_size: int | None = None,
    temperature: float = 1.0,
    inference_coalesce_seconds: float = 0.002,
) -> TorchPopulationEvaluationStats:
    """Evaluate CPU game workers while the parent exclusively owns Torch/CUDA.

    Each existing ``max_parallel_games`` chunk is partitioned over spawn-based
    workers. Workers own environments, observation/encoding, legality, sampling,
    and transitions. They send only encoded observations plus immutable policy
    ids to the parent, which loads checkpoints and performs all neural inference.
    Payoff-table mutation remains parent-only.
    """

    if not requests:
        raise ValueError("population evaluation requires at least one request")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if max_parallel_games <= 0:
        raise ValueError("max_parallel_games must be positive")
    if max_inference_batch_size is not None and max_inference_batch_size <= 0:
        raise ValueError("max_inference_batch_size must be positive")
    if inference_coalesce_seconds < 0:
        raise ValueError("inference_coalesce_seconds cannot be negative")

    games = tuple(
        _MultiprocessGame(index, request.profile, seed)
        for index, (request, seed) in enumerate(
            (request, seed)
            for request in requests
            for seed in request.seeds
        )
    )
    chunks = _chunk_unique_seeds(games, max_parallel_games=max_parallel_games)
    workers = min(worker_count, max_parallel_games, len(games))
    context = mp.get_context("spawn")
    inference_queue = context.Queue()
    result_queue = context.Queue()
    job_queues = [context.Queue() for _ in range(workers)]
    response_queues = [context.Queue() for _ in range(workers)]
    processes = [
        context.Process(
            target=_worker_main,
            args=(
                worker_id,
                player_specs,
                job_queues[worker_id],
                inference_queue,
                response_queues[worker_id],
                result_queue,
            ),
            name=f"werewolf-rollout-{worker_id}",
            daemon=True,
        )
        for worker_id in range(workers)
    ]
    for process in processes:
        process.start()

    checkpoint_loads = 0
    inference_stats = TorchRolloutInferenceStats()
    rollout_started = perf_counter()
    try:
        for job_id, chunk in enumerate(chunks):
            active_workers = min(workers, len(chunk))
            assignments = tuple(
                tuple(chunk[offset::active_workers])
                for offset in range(active_workers)
            )
            policy_ids = sorted(
                {
                    policy_id
                    for game in chunk
                    for policy_id in (
                        game.profile.village,
                        game.profile.werewolf,
                        game.profile.fox,
                    )
                }
            )
            model_cache = {
                policy_id: pool.load(policy_id).eval() for policy_id in policy_ids
            }
            checkpoint_loads += len(model_cache)

            for worker_id, assignment in enumerate(assignments):
                job_queues[worker_id].put(
                    _WorkerJob(
                        job_id=job_id,
                        games=assignment,
                        max_discussion_ticks=max_discussion_ticks,
                        max_inference_batch_size=max_inference_batch_size,
                        temperature=temperature,
                    )
                )

            worker_results = _serve_worker_jobs(
                active_workers=active_workers,
                job_id=job_id,
                model_cache=model_cache,
                inference_queue=inference_queue,
                response_queues=response_queues,
                result_queue=result_queue,
                max_inference_batch_size=max_inference_batch_size,
                inference_coalesce_seconds=inference_coalesce_seconds,
                inference_stats=inference_stats,
            )
            by_index = {
                game.index: game
                for worker_result in worker_results
                for game in worker_result.games
            }
            if len(by_index) != len(chunk):
                raise RuntimeError("multiprocess rollout did not return every game")
            table.record_results(
                PopulationPayoffResult(
                    profile=by_index[game.index].profile,
                    winner=by_index[game.index].winner,
                    is_draw=by_index[game.index].is_draw,
                    days=by_index[game.index].days,
                )
                for game in chunk
            )
            del model_cache
    finally:
        _stop_workers(processes, job_queues)

    return TorchPopulationEvaluationStats(
        games=len(games),
        rollout_chunks=len(chunks),
        rollout_seconds=perf_counter() - rollout_started,
        checkpoint_loads=checkpoint_loads,
        inference_calls=inference_stats.inference_calls,
        inference_observations=inference_stats.inference_observations,
        max_pending_inference_requests=inference_stats.max_pending_requests,
        max_inference_batch=inference_stats.max_inference_batch,
    )


def _worker_main(
    worker_id: int,
    player_specs: list[PlayerSpec],
    job_queue: Any,
    inference_queue: Any,
    response_queue: Any,
    result_queue: Any,
) -> None:
    torch.set_num_threads(1)
    while True:
        job = job_queue.get()
        if job is None:
            return
        if not isinstance(job, _WorkerJob):
            result_queue.put(
                _WorkerResult(
                    worker_id=worker_id,
                    job_id=-1,
                    games=(),
                    error="worker received an invalid job",
                )
            )
            continue
        try:
            if not job.games:
                raise RuntimeError("worker job contains no games")
            collector = _RemoteInferenceCollector(
                player_specs,
                job.games[0].profile.village,
                worker_id=worker_id,
                inference_queue=inference_queue,
                response_queue=response_queue,
                max_discussion_ticks=job.max_discussion_ticks,
                max_inference_batch_size=job.max_inference_batch_size,
                temperature=job.temperature,
            )
            team_models = tuple(
                {
                    Team.VILLAGE: collector.policy_model(game.profile.village),
                    Team.WEREWOLF: collector.policy_model(game.profile.werewolf),
                    Team.FOX: collector.policy_model(game.profile.fox),
                }
                for game in job.games
            )
            results = collector.collect(
                tuple(game.seed for game in job.games),
                team_models=team_models,
            )
            result_queue.put(
                _WorkerResult(
                    worker_id=worker_id,
                    job_id=job.job_id,
                    games=tuple(
                        _WorkerGameResult(
                            index=game.index,
                            profile=game.profile,
                            winner=result.winner,
                            is_draw=result.is_draw,
                            days=result.days,
                        )
                        for game, result in zip(job.games, results, strict=True)
                    ),
                )
            )
        except BaseException:
            result_queue.put(
                _WorkerResult(
                    worker_id=worker_id,
                    job_id=job.job_id,
                    games=(),
                    error=traceback.format_exc(),
                )
            )


def _serve_worker_jobs(
    *,
    active_workers: int,
    job_id: int,
    model_cache: dict[str, TorchTransformerPolicy],
    inference_queue: Any,
    response_queues: list[Any],
    result_queue: Any,
    max_inference_batch_size: int | None,
    inference_coalesce_seconds: float,
    inference_stats: TorchRolloutInferenceStats,
) -> tuple[_WorkerResult, ...]:
    completed: dict[int, _WorkerResult] = {}
    while len(completed) < active_workers:
        _drain_worker_results(result_queue, completed, job_id)
        if len(completed) >= active_workers:
            break
        try:
            first = inference_queue.get(timeout=0.01)
        except Empty:
            _assert_workers_not_failed(completed)
            continue

        messages = [first]
        if inference_coalesce_seconds == 0:
            while True:
                try:
                    messages.append(inference_queue.get_nowait())
                except Empty:
                    break
        else:
            deadline = perf_counter() + inference_coalesce_seconds
            while True:
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    break
                try:
                    messages.append(inference_queue.get(timeout=remaining))
                except Empty:
                    break

        _serve_inference_messages(
            messages,
            model_cache=model_cache,
            response_queues=response_queues,
            max_inference_batch_size=max_inference_batch_size,
            inference_stats=inference_stats,
        )
    _assert_workers_not_failed(completed)
    return tuple(completed[index] for index in sorted(completed))


def _drain_worker_results(
    result_queue: Any,
    completed: dict[int, _WorkerResult],
    job_id: int,
) -> None:
    while True:
        try:
            result = result_queue.get_nowait()
        except Empty:
            return
        if not isinstance(result, _WorkerResult):
            raise RuntimeError("multiprocess rollout returned an invalid worker result")
        if result.job_id != job_id:
            raise RuntimeError("multiprocess rollout returned a stale worker result")
        completed[result.worker_id] = result


def _assert_workers_not_failed(completed: dict[int, _WorkerResult]) -> None:
    for result in completed.values():
        if result.error is not None:
            raise RuntimeError(
                f"multiprocess rollout worker {result.worker_id} failed:\n{result.error}"
            )


def _serve_inference_messages(
    messages: list[Any],
    *,
    model_cache: dict[str, TorchTransformerPolicy],
    response_queues: list[Any],
    max_inference_batch_size: int | None,
    inference_stats: TorchRolloutInferenceStats,
) -> None:
    requests: list[_InferenceMessage] = []
    for message in messages:
        if not isinstance(message, _InferenceMessage):
            raise RuntimeError("multiprocess rollout returned an invalid inference request")
        if message.policy_id not in model_cache:
            raise RuntimeError(f"worker requested unloaded policy {message.policy_id}")
        requests.append(message)

    inference_stats.record_pending(sum(len(request.observations) for request in requests))
    grouped: dict[str, list[_InferenceMessage]] = {}
    for request in requests:
        grouped.setdefault(request.policy_id, []).append(request)

    for policy_id, policy_requests in grouped.items():
        model = model_cache[policy_id]
        observations = tuple(
            observation
            for request in policy_requests
            for observation in request.observations
        )
        logits: list[PolicyLogits] = []
        limit = max_inference_batch_size or len(observations)
        for start in range(0, len(observations), limit):
            batch = observations[start : start + limit]
            inference_stats.record_inference(len(batch))
            with torch.no_grad():
                output = model.forward_batch(batch)
            logits.extend(model.policy_logits_batch(output))

        offset = 0
        for request in policy_requests:
            end = offset + len(request.observations)
            response_queues[request.worker_id].put(
                _InferenceReply(
                    request_id=request.request_id,
                    logits=tuple(logits[offset:end]),
                )
            )
            offset = end
        if offset != len(logits):
            raise RuntimeError("multiprocess inference split was inconsistent")


def _stop_workers(processes: list[Any], job_queues: list[Any]) -> None:
    for job_queue in job_queues:
        job_queue.put(None)
    for process in processes:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)


def _chunk_unique_seeds(
    games: tuple[_MultiprocessGame, ...],
    *,
    max_parallel_games: int,
) -> tuple[tuple[_MultiprocessGame, ...], ...]:
    remaining = list(games)
    chunks: list[tuple[_MultiprocessGame, ...]] = []
    while remaining:
        chunk: list[_MultiprocessGame] = []
        deferred: list[_MultiprocessGame] = []
        seen_seeds: set[int] = set()
        for game in remaining:
            if len(chunk) < max_parallel_games and game.seed not in seen_seeds:
                chunk.append(game)
                seen_seeds.add(game.seed)
            else:
                deferred.append(game)
        if not chunk:
            raise RuntimeError("multiprocess evaluation could not schedule a rollout chunk")
        chunks.append(tuple(chunk))
        remaining = deferred
    return tuple(chunks)
