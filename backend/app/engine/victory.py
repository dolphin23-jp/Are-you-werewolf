"""Victory condition evaluation.

- Village win: 0 werewolves alive and the Fox is not alive.
- Fox win: the Fox is alive AND either 0 werewolves remain or werewolves
  have reached parity/majority (the Fox "steals" the win either way).
- Werewolf win: werewolves reach parity/majority and the Fox is not alive.
- Draw: handled by the vote manager when runoffs are exhausted; not
  computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.engine.roles import RoleName, Team
from app.engine.state import GameState


@dataclass
class VictoryResult:
    winner: Team
    reason: str


class VictoryChecker:
    def check(self, state: GameState) -> VictoryResult | None:
        alive = state.alive_players()
        wolves_alive = [p for p in alive if p.role == RoleName.WEREWOLF]
        fox_alive = any(p.role == RoleName.FOX for p in alive)
        non_wolves_alive = len(alive) - len(wolves_alive)

        if not wolves_alive:
            if fox_alive:
                return VictoryResult(Team.FOX, "人狼が全滅し、妖狐が生存しているため妖狐陣営の勝利")
            return VictoryResult(Team.VILLAGE, "人狼が全滅したため村人陣営の勝利")

        if len(wolves_alive) >= non_wolves_alive:
            if fox_alive:
                return VictoryResult(
                    Team.FOX,
                    "人狼が村人陣営と同数以上に達したが妖狐が生存しているため妖狐陣営の勝利",
                )
            return VictoryResult(Team.WEREWOLF, "人狼が村人陣営と同数以上に達し人狼陣営の勝利")

        return None

    def player_won(self, state: GameState, player_id: str, winner: Team) -> bool:
        """Individual win/loss by team alignment (Madman/Fox are affiliated
        independently of their apparent village-side CO-composition bucket)."""
        player = state.players[player_id]
        return player.team == winner
