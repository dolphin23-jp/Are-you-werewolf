import { ROLE_DEFINITIONS_JA, ROLE_LABELS } from "../../api/types";
import { useGameStore } from "../../state/gameStore";

export function RoleRevealScreen() {
  const view = useGameStore((s) => s.view);
  const playerNames = useGameStore((s) => s.playerNames);
  const setScreen = useGameStore((s) => s.setScreen);

  if (!view || !view.your_role) {
    return (
      <div className="screen screen--role-reveal">
        <p>役職を確認しています...</p>
      </div>
    );
  }

  const roleLabel = ROLE_LABELS[view.your_role];
  const description = ROLE_DEFINITIONS_JA[view.your_role];
  const allyNames = view.allies.map((id) => playerNames[id] ?? id);

  return (
    <div className="screen screen--role-reveal">
      <h2>あなたの役職</h2>
      <div className="role-card">
        <div className="role-card__name">{roleLabel}</div>
        <p className="role-card__description">{description}</p>
        {allyNames.length > 0 && (
          <p className="role-card__allies">仲間: {allyNames.join("、")}</p>
        )}
      </div>
      <button className="btn btn--primary" onClick={() => setScreen("main")}>
        はじめる
      </button>
    </div>
  );
}
