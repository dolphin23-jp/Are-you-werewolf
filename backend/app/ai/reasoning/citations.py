"""Who said whose ballot went where, and whether the record agrees.

A player who says "ダイキは1日目にユイへ投票した" is citing the record, and the
record can be checked. When the citation is wrong, two things follow that a
straight read of `vote_records` cannot produce on its own:

* the listener has picked up a reason built on a ballot that never happened, so
  correcting it later has to move their conclusion;
* the *citation* is keyed on what was said, not on what happened, which is the
  only way a mistaken quote is distinguishable from a correct one.

This is what `misremembered_vote` was always for. Without a producer the
category was a weight with nothing to weigh, and a human saying "I never voted
for him" retracted nothing, because nobody had built anything on the claim.

Correct citations produce nothing. Repeating the record accurately is not
evidence about anybody.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ai.reasoning.belief.corrections import vote_fact_id
from app.ai.reasoning.facts import PublicFactLedger, mentions_player

# "<voter>は … <target>に投票した". Narrow: both sides have to be named in one
# clause, so a sentence that merely mentions two people does not become a claim
# about a ballot.
_CITATION_RE = re.compile(
    r"(?P<voter>[^、，。！？!?\s]{1,20}?)は"
    r"(?:[^、，。！？!?]{0,12}?)"
    r"(?P<target>[^、，。！？!?\s]{1,20}?)(?:へ|に)投票(?:しました|した|している|しています)"
)
_DAY_RE = re.compile(r"(?P<day>\d+)日目")


@dataclass(frozen=True)
class VoteCitation:
    """Someone's account of one ballot, with whether the record agrees."""

    speaker_id: str
    voter_id: str
    cited_target_id: str
    day: int
    recorded_target_id: str | None
    source_message_id: str = ""

    @property
    def is_accurate(self) -> bool:
        return self.recorded_target_id == self.cited_target_id

    @property
    def is_verifiable(self) -> bool:
        return self.recorded_target_id is not None

    @property
    def fact_id(self) -> str:
        """Keyed on the ballot *as cited*, so a misquote is its own fact."""
        return vote_fact_id(self.voter_id, self.day, 1, self.cited_target_id)


def parse_vote_citations(
    text: str, ledger: PublicFactLedger, speaker_id: str, source_message_id: str = ""
) -> tuple[VoteCitation, ...]:
    """Read accounts of other people's ballots out of one public message."""
    day_match = _DAY_RE.search(text)
    day = int(day_match.group("day")) if day_match else ledger.day
    citations: list[VoteCitation] = []
    for match in _CITATION_RE.finditer(text):
        voter = _named(match.group("voter"), ledger)
        target = _named(match.group("target"), ledger)
        if voter is None or target is None or voter == target:
            continue
        recorded = ledger.vote_of(voter, day)
        citations.append(
            VoteCitation(
                speaker_id=speaker_id,
                voter_id=voter,
                cited_target_id=target,
                day=day,
                recorded_target_id=recorded.target_id if recorded else None,
                source_message_id=source_message_id,
            )
        )
    return tuple(citations)


def _named(label: str, ledger: PublicFactLedger) -> str | None:
    for player_id in ledger.known_player_ids():
        if mentions_player(label, player_id, ledger.name_of(player_id)):
            return player_id
    return None


__all__ = ["VoteCitation", "parse_vote_citations"]
