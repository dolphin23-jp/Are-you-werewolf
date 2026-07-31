import { ROLE_LABELS } from "../../api/types";
import { useGameStore } from "../../state/gameStore";

export function COPanel() {
  const view = useGameStore((s) => s.view);
  const playerNames = useGameStore((s) => s.playerNames);

  if (!view) return null;

  return (
    <div className="panel co-panel">
      <h3>公開情報まとめ</h3>
      <p className="co-panel__hint">議論中に本人が明かしたCOを自動で記録します</p>
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
    </div>
  );
}
