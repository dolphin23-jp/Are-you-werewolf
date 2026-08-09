import pytest

from app.engine.roles import Team

torch_historical_state = pytest.importorskip(
    "app.training.torch_historical_run_state"
)
TorchHistoricalRunProgress = torch_historical_state.TorchHistoricalRunProgress


def test_historical_run_progress_rejects_invalid_boundaries():
    with pytest.raises(ValueError, match="completed_batches cannot be negative"):
        TorchHistoricalRunProgress(
            completed_batches=-1,
            base_seed=1,
            episodes_per_batch=1,
            requested_teams=(Team.VILLAGE,),
        )

    with pytest.raises(ValueError, match="episodes_per_batch must be positive"):
        TorchHistoricalRunProgress(
            completed_batches=0,
            base_seed=1,
            episodes_per_batch=0,
            requested_teams=(Team.VILLAGE,),
        )

    with pytest.raises(ValueError, match="requested_teams cannot be empty"):
        TorchHistoricalRunProgress(
            completed_batches=0,
            base_seed=1,
            episodes_per_batch=1,
            requested_teams=(),
        )

    with pytest.raises(ValueError, match="requested_teams cannot contain duplicates"):
        TorchHistoricalRunProgress(
            completed_batches=0,
            base_seed=1,
            episodes_per_batch=1,
            requested_teams=(Team.VILLAGE, Team.VILLAGE),
        )

    with pytest.raises(ValueError, match="next_pool_generation cannot be negative"):
        TorchHistoricalRunProgress(
            completed_batches=0,
            base_seed=1,
            episodes_per_batch=1,
            requested_teams=(Team.VILLAGE,),
            next_pool_generation=-1,
        )
