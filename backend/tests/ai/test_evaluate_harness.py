from scripts.evaluate import _make_specs


def test_evaluation_replaces_human_with_seventeenth_ai():
    specs = _make_specs()

    assert len(specs) == 17
    assert [spec.player_id for spec in specs] == [f"p{i}" for i in range(17)]
    assert all(not spec.is_human for spec in specs)
    assert all(spec.name != "観戦席" for spec in specs)
