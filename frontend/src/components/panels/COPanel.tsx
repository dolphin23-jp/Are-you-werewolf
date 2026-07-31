import { useState } from "react";
import { declareCo } from "../../api/client";
import { ROLE_LABELS, type RoleName } from "../../api/types";
import { useGameStore } from "../../state/gameStore";

const CLAIMABLE_ROLES: RoleName[] = ["seer", "medium", "hunter", "freemason", "villager"];

export function COPanel() {
  const view = useGameStore((s) => s.view);
  const sessionId = useGameStore((s) => s.sessionId);
  const playerNames = useGameStore((s) => s.playerNames);
  const refreshView = useGameStore((s) => s.refreshView);
  const setError = useGameStore((s) => s.setError);
  const [claiming, setClaiming] = useState(false);

  if (!view || !sessionId) return null;

  const isAlive = view.players.find((p) => p.player_id === view.your_player_id)?.alive ?? false;
  const alreadyClaimed = view.co_declarations.some((c) => c.player_id === view.your_player_id);

  const handleClaim = async (role: RoleName) => {
    setClaiming(true);
    try {
      await declareCo(sessionId, role);
      await refreshView();
    } catch (e) {
      setError(e instanceof Error ? e.message : "CO に失敗しました");
    } finally {
      setClaiming(false);
    }
  };

  return (
    <div className="panel co-panel">
      <h3>カミングアウト (CO)</h3>
      {view.co_declarations.length > 0 ? (
        <ul>
          {view.co_declarations.map((c, i) => (
            <li key={i}>
              {playerNames[c.player_id] ?? c.player_id}: {ROLE_LABELS[c.claimed_role]}CO
            </li>
          ))}
        </ul>
      ) : (
        <p className="co-panel__empty">まだ誰もCOしていません</p>
      )}
      {isAlive && !alreadyClaimed && (
        <div className="co-panel__actions">
          {CLAIMABLE_ROLES.map((role) => (
            <button key={role} className="btn btn--small" disabled={claiming} onClick={() => void handleClaim(role)}>
              {ROLE_LABELS[role]}CO
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
