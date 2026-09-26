"""The mock claims only what its seat could: its own role or a planned fake.

It used to pick any CO line at random, so mock campaigns ran on a board of
fake freemasons and counter-claims no real model produces.
"""

from __future__ import annotations

import asyncio

from app.ai.provider.mock import MockProvider
from app.ai.schemas import DiscussionOutput


def _claims(system: str, draws: int = 200) -> set[str | None]:
    provider = MockProvider(seed=3)

    async def run() -> set[str | None]:
        seen: set[str | None] = set()
        for _ in range(draws):
            output = await provider.generate_structured(
                system=system, messages=[], response_schema=DiscussionOutput
            )
            assert output is not None
            seen.add(output.public_claim_role)
        return seen

    return asyncio.run(run())


def test_a_villager_never_claims():
    assert _claims("【役職情報】あなたの役職は「村人」です。") == {None}


def test_a_seer_claims_only_seer():
    assert _claims("【役職情報】あなたの役職は「占い師」です。") == {None, "seer"}


def test_a_wolf_claims_only_its_planned_fake():
    system = "【役職情報】あなたの役職は「人狼」です。あなたは「霊媒師」を騙る担当です。"
    assert _claims(system) == {None, "medium"}
