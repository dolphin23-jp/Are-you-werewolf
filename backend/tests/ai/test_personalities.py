from app.ai.personalities import PERSONALITIES, assign_personalities


def test_assignment_is_seed_reproducible():
    ids = [f"p{i}" for i in range(1, 17)]
    a = assign_personalities(ids, seed=1)
    b = assign_personalities(ids, seed=1)
    assert {pid: p.name for pid, p in a.items()} == {pid: p.name for pid, p in b.items()}


def test_assignment_covers_all_players():
    ids = [f"p{i}" for i in range(1, 17)]
    assignment = assign_personalities(ids, seed=3)
    assert set(assignment.keys()) == set(ids)
    assert all(p in PERSONALITIES for p in assignment.values())


def test_prompt_section_mentions_all_axes():
    personality = PERSONALITIES[0]
    section = personality.to_prompt_section()
    assert personality.tone in section
    assert personality.thinking_style in section
    assert personality.discussion_style in section
    assert personality.emotional_tendency in section
