from app.engine.roles import (
    ROLE_DEFINITIONS,
    TOTAL_PLAYERS,
    AlphaWolfTracker,
    RoleAssigner,
    RoleName,
)


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
