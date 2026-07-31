import { PlayerAvatar } from "../common/PlayerAvatar";
import { useGameStore } from "../../state/gameStore";

export function PlayerListPanel() {
  const view = useGameStore((s) => s.view);
  if (!view) return null;

  const aliveCount = view.players.filter((p) => p.alive).length;

  return (
    <div className="panel player-list-panel">
      <h3>{`プレイヤー (${aliveCount}/${view.players.length} 生存)`}</h3>
      <ul className="player-list">
        {view.players.map((p) => (
          <li key={p.player_id} className={p.alive ? "" : "player-list__item--dead"}>
            <PlayerAvatar name={p.name} alive={p.alive} isYou={p.player_id === view.your_player_id} />
            <span>{p.name}</span>
            {!p.alive && <span className="tag">{deathLabel(p.death_cause)}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

function deathLabel(cause: string | null): string {
  switch (cause) {
    case "executed":
      return "処刑";
    case "attacked":
      return "襲撃";
    case "cursed":
      return "呪殺";
    case "first_victim":
      return "犠牲";
    default:
      return "死亡";
  }
}
