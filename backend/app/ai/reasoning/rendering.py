"""The execution candidate a turn displays, fixed by code and read back from the text.

The belief engine decides who a seat wants executed. The model writes the prose.
Those two used to meet only in the reasoning memo: the memo was overwritten with
the decided target, and the sentence the table actually saw was left alone. So a
model could write "ユイを第一候補にします" while the code had decided on Daiki,
and the table heard Yui.

Two operations, kept deliberately symmetric:

* `enforce_execution_target` removes every sentence in the prose that declares a
  candidate other than the decided one, then states the decided one in fixed
  wording. It is applied as the *last* change to the message, so nothing that
  runs afterwards can quietly drop it.
* `displayed_execution_target` reads the candidate back out of the final text.
  This is what gets recorded as the seat's stated target and what the audit
  reports -- measured from the string the table saw, not asserted from the
  decision it was supposed to match. A check that reads its answer off the
  input cannot catch the case where the two came apart.

Detection is intentionally broad on the removal side. Dropping a rationale
sentence that happened to mention a rival costs a little colour; leaving a
contradicting declaration in place makes the turn say two things.
"""

from __future__ import annotations

import re

from app.ai.reasoning.facts import PublicFactLedger

# A sentence that commits to who should be executed. Covers the forms models
# actually produce, not only the literal "第一候補": "第一処刑候補" is not a
# substring match for it, and "吊りたい" / "投票します" commit just as hard.
_DECLARES_CANDIDATE = re.compile(
    r"第一(?:処刑)?候補"
    r"|本命|最有力"
    r"|(?:吊|つ)(?:り|る)(?:たい|べき|ます|ましょう)"
    r"|吊るべき|吊りたい"
    r"|処刑(?:したい|すべき|します|しましょう|先)"
    r"|投票(?:します|する|先|したい|しましょう)"
    r"|に入れ(?:ます|る|たい)"
)
_SENTENCE = re.compile(r"[^。！？!?\n]*[。！？!?\n]?")
_CANONICAL = re.compile(r"現時点の第一処刑候補は(?P<label>[^。]+?)です。")


def canonical_target_sentence(ledger: PublicFactLedger, target_id: str) -> str:
    return f"現時点の第一処刑候補は{ledger.label_of(target_id)}です。"


def _sentences(message: str) -> list[str]:
    return [part for part in _SENTENCE.findall(message) if part.strip()]


def enforce_execution_target(
    message: str,
    target_id: str | None,
    ledger: PublicFactLedger,
    *,
    speaker_id: str,
) -> str:
    """Strip competing candidate declarations, then state the decided one.

    With no decided target the seat has not narrowed it down, so any sentence
    naming someone as the candidate contradicts that and is removed too -- but
    nothing is added, because there is nothing to state.
    """
    kept: list[str] = []
    for sentence in _sentences(message):
        if _CANONICAL.search(sentence):
            continue  # restated below; never duplicated
        if _DECLARES_CANDIDATE.search(sentence):
            named = {
                pid
                for pid in ledger.mentioned_player_ids(sentence)
                if pid != speaker_id
            }
            if named - ({target_id} if target_id else set()):
                continue
        kept.append(sentence)
    body = "".join(kept).strip()
    if target_id is None:
        return body
    return canonical_target_sentence(ledger, target_id) + body


def displayed_execution_target(message: str, ledger: PublicFactLedger) -> str | None:
    """The candidate the final text actually states, or None if it states none."""
    match = _CANONICAL.search(message)
    if match is None:
        return None
    named = ledger.mentioned_player_ids(match.group("label"))
    return named[0] if len(named) == 1 else None


__all__ = [
    "canonical_target_sentence",
    "displayed_execution_target",
    "enforce_execution_target",
]
