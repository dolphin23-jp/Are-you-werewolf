from app.engine.roles import (
    ROLE_DEFINITIONS,
    TOTAL_PLAYERS,
    AlphaWolfTracker,
    RoleAssigner,
    RoleName,
)
from tests.conftest import make_controller


def test_total_players_is_17():
    assert TOTAL_PLAYERS == 17
    assert sum(d.count for d in ROLE_DEFINITIONS.values()) == 17


def test_role_assigner_covers_all_players_exactly_once():
    player_ids = [f"p{i}" for i in range(17)]
    assignment = RoleAssigner(seed=42).assign(player_ids)
    assert set(assignment.keys()) == set(player_ids)
    counts: dict[RoleName, int] = {}
    for role in assignment.values():
        counts[role] = counts.get(role, 0) + 1
    for role, definition in ROLE_DEFINITIONS.items():
        assert counts.get(role, 0) == definition.count


def test_role_assigner_is_seed_reproducible():
    player_ids = [f"p{i}" for i in range(17)]
    a = RoleAssigner(seed=7).assign(player_ids)
    b = RoleAssigner(seed=7).assign(player_ids)
    assert a == b


def test_first_victim_never_wolf_or_fox():
    player_ids = [f"p{i}" for i in range(17)]
    for seed in range(50):
        assigner = RoleAssigner(seed=seed)
        assignment = assigner.assign(player_ids)
        victim = assigner.pick_first_victim(assignment)
        assert assignment[victim] not in (RoleName.WEREWOLF, RoleName.FOX)


def test_alpha_wolf_reassigned_on_death():
    tracker = AlphaWolfTracker(["w1", "w2", "w3"], seed=1)
    alpha = tracker.alpha_id
    assert alpha in ("w1", "w2", "w3")
    tracker.on_wolf_death(alpha, rng=__import__("random").Random(1))
    assert tracker.alpha_id != alpha
    assert tracker.alpha_id in {"w1", "w2", "w3"} - {alpha}


def test_alpha_wolf_no_reassign_when_non_alpha_dies():
    tracker = AlphaWolfTracker(["w1", "w2", "w3"], seed=1)
    alpha = tracker.alpha_id
    other = next(w for w in ("w1", "w2", "w3") if w != alpha)
    tracker.on_wolf_death(other)
    assert tracker.alpha_id == alpha


def test_alpha_wolf_successor_is_reproducible_from_the_seed():
    """Replaying a seed must pick the same successor, or every later attack forks."""
    for seed in range(30):
        successors = []
        for _ in range(2):
            tracker = AlphaWolfTracker(["w1", "w2", "w3"], seed=seed)
            tracker.on_wolf_death(tracker.alpha_id)
            successors.append(tracker.alpha_id)
        assert successors[0] == successors[1], f"seed {seed} diverged: {successors}"


def test_executing_the_alpha_picks_the_same_successor_on_replay():
    def successor(seed: int) -> str:
        controller = make_controller(seed=seed)
        controller.start_game()
        controller.resolve_night()
        controller.start_discussion()
        controller.end_discussion()
        alpha = controller.alpha_wolf_id
        for voter in controller.state.alive_ids():
            if voter != alpha:
                controller.vote(voter, alpha)
        controller.vote(alpha, next(p for p in controller.state.alive_ids() if p != alpha))
        controller.resolve_votes()
        assert not controller.state.players[alpha].alive
        return controller.alpha_wolf_id

    for seed in range(1, 21):
        assert successor(seed) == successor(seed), f"seed {seed} diverged"
