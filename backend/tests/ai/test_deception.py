from app.ai.deception import FakeClaimGuard, assign_madman_strategy, assign_wolf_deception


def test_wolf_deception_assigns_fake_roles_only_to_wolves():
    wolves = ["w1", "w2", "w3"]
    assignment = assign_wolf_deception(wolves, seed=1)
    assert set(assignment.fake_role_by_player.keys()) <= set(wolves)
    assert set(assignment.lurking_player_ids) | set(assignment.fake_role_by_player.keys()) == set(
        wolves
    )


def test_wolf_deception_is_seed_reproducible():
    wolves = ["w1", "w2", "w3"]
    a = assign_wolf_deception(wolves, seed=42)
    b = assign_wolf_deception(wolves, seed=42)
    assert a == b


def test_madman_strategy_returns_valid_choice():
    name, fake_role = assign_madman_strategy(seed=1)
    assert name in ("fake_seer", "fake_medium", "lurk")


def test_fake_claim_guard_rejects_teammate_as_black():
    guard = FakeClaimGuard(wolf_team_ids={"w1", "w2"})
    assert guard.can_claim_result("w1", "w2", claimed_is_werewolf=True) is False
    assert guard.can_claim_result("w1", "v1", claimed_is_werewolf=True) is True


def test_fake_claim_guard_rejects_retargeting_same_player():
    guard = FakeClaimGuard(wolf_team_ids={"w1"})
    assert guard.can_claim_result("w1", "v1", claimed_is_werewolf=False) is True
    guard.register_claim("w1", "v1")
    assert guard.can_claim_result("w1", "v1", claimed_is_werewolf=False) is False
