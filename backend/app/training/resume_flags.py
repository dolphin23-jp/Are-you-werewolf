"""Say which command-line flags an exact resume will not apply.

On ``--resume`` the saved run-state is authoritative for hyperparameters, seeds
and model shape. Passing, say, ``--learning-rate 1e-4`` alongside ``--resume``
used to be accepted without a word while the old rate kept being used.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from typing import TextIO


def warn_ignored_on_resume(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
    dests: Iterable[str],
    *,
    stream: TextIO | None = None,
) -> tuple[str, ...]:
    """Print one warning naming the given flags that appear in ``argv``."""

    wanted = set(dests)
    passed: list[str] = []
    for action in parser._actions:
        if action.dest not in wanted:
            continue
        if any(
            arg == option or arg.startswith(option + "=")
            for option in action.option_strings
            for arg in argv
        ):
            passed.append(action.option_strings[-1])
    if passed:
        print(
            "warning: --resume keeps the saved run-state values; ignoring "
            + ", ".join(passed),
            file=stream if stream is not None else sys.stderr,
        )
    return tuple(passed)
