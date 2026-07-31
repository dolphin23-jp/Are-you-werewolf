import { ROLE_LABELS } from "../../api/types";
import { useGameStore } from "../../state/gameStore";

export function RolePanel() {
  const view = useGameStore((s) => s.view);
  const playerNames = useGameStore((s) => s.playerNames);

  if (!view || !view.your_role) return null;

  const nameOf = (id: string) => playerNames[id] ?? id;

  return (
    <div className="panel role-panel">
      <h3>あなた: {ROLE_LABELS[view.your_role]}</h3>
      {view.allies.length > 0 && <p>仲間: {view.allies.map(nameOf).join("、")}</p>}
      {view.your_divine_results.length > 0 && (
        <div>
          <h4>占い結果</h4>
          <ul>
            {view.your_divine_results.map((r) => (
              <li key={`${r.day}-${r.target_id}`}>
                {r.day}日目: {nameOf(r.target_id)} は{r.is_werewolf ? "人狼" : "人狼ではない"}
              </li>
            ))}
          </ul>
        </div>
      )}
      {view.your_medium_results.length > 0 && (
        <div>
          <h4>霊媒結果</h4>
          <ul>
            {view.your_medium_results.map((r) => (
              <li key={`${r.day}-${r.target_id}`}>
                {r.day}日目: {nameOf(r.target_id)} は{r.is_werewolf ? "人狼だった" : "人狼ではなかった"}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
