"""`CreateGameRequest.human_name` regression.

The default used to be the literal string "あなた" (the Japanese second-
person pronoun). That string gets broadcast verbatim into every OTHER
player's prompt (chat log lines, roster labels like "NAME(p0)"), where it
reads as "you" rather than a name -- plausibly why a real evaluation run
showed a wave of AI players claiming to themselves be p0.
"""

from __future__ import annotations

from app.api.schemas import CreateGameRequest


def test_human_name_default_is_not_a_bare_pronoun():
    assert CreateGameRequest().human_name != "あなた"
