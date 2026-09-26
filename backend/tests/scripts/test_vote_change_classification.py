"""Seeded mock games must not report explained vote changes as unexplained.

`unexplained_vote_change_count` is a hard gate failure. A mock v2 campaign over
seeds 1-3 used to report 47 of them, most of which had a public cause the
classifier did not look at: a counter-CO after the seat spoke, the first-round
tally before a runoff, or later public arguments.
"""

from __future__ import annotations

import asyncio

from scripts.compare_engines import play


def test_seed_one_has_no_unexplained_vote_changes():
    run = asyncio.run(play("v2", 1))
    quality = run.reasoning_quality

    assert quality["vote_count"] > 0
    assert quality["justified_vote_change_count"] > 0
    assert quality["unexplained_vote_change_count"] == 0
