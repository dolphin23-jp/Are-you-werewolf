"""Deterministic rendering of the public-fact portion of daily summaries."""

from __future__ import annotations

from app.ai.reasoning.facts import PublicFactLedger


def render_public_fact_summary(ledger: PublicFactLedger) -> str:
    labels = {item.player_id: f"{item.name}({item.player_id})" for item in ledger.players}
    alive = "、".join(labels[pid] for pid in ledger.alive_ids) or "なし"
    dead = [
        f"{labels[item.player_id]}:{item.death_day}日目/{item.death_cause.value}"
        for item in ledger.players
        if not item.alive and item.death_day is not None and item.death_cause is not None
    ]
    claims = [
        f"{item.day}日目 {labels[item.player_id]}={item.claimed_role.value}CO"
        for item in ledger.claims
    ]
    results = [
        f"{item.day}日目 {labels[item.claimant_id]}の"
        f"{'占い' if item.result_type == 'seer' else '霊媒'}主張:"
        f"{labels[item.target_id]}={'黒' if item.is_werewolf else '白'}"
        for item in ledger.results
    ]
    votes = [
        f"{item.day}日目R{item.round} {labels[item.voter_id]}→{labels[item.target_id]}"
        for item in ledger.votes
    ]
    return "\n".join(
        [
            "【公開事実】",
            f"生存者: {alive}",
            "死亡者: " + ("、".join(dead) or "なし"),
            "公開CO: " + (" / ".join(claims) or "なし"),
            "公開判定: " + (" / ".join(results) or "なし"),
            "投票履歴: " + (" / ".join(votes) or "なし"),
        ]
    )
