import { useState } from "react";
import { resolveNight, submitNightAction } from "../../api/client";
import { useGameStore } from "../../state/gameStore";

const ACTION_LABELS: Record<string, string> = {
  divine: "占う",
  guard: "護衛する",
  attack: "襲撃する",
};

export function NightActionPanel() {
  const view = useGameStore((s) => s.view);
  const sessionId = useGameStore((s) => s.sessionId);
  const refreshView = useGameStore((s) => s.refreshView);
  const setError = useGameStore((s) => s.setError);
  const [target, setTarget] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (!view || !sessionId || view.phase !== "night") return null;

  const isAlive = view.players.find((p) => p.player_id === view.your_player_id)?.alive ?? false;
  if (!isAlive) {
    return (
      <div className="panel night-action-panel">
        <p>あなたは既に脱落しています。AIの夜が進行中です...</p>
      </div>
    );
  }

  const role = view.your_role;
  let actionType: "divine" | "guard" | "attack" | null = null;
  if (role === "seer") actionType = "divine";
  else if (role === "hunter" && view.day > 0) actionType = "guard";
  else if (role === "werewolf" && view.day > 0 && view.is_alpha_wolf) actionType = "attack";

  if (actionType === null) {
    const waitingMessage =
      role === "werewolf"
        ? "アルファ人狼が襲撃先を決めています。人狼チャットで相談しましょう。"
        : "夜が更けています。AIたちが行動中です...";
    return (
      <div className="panel night-action-panel">
        <p>{waitingMessage}</p>
      </div>
    );
  }

  const excludeIds =
    actionType === "attack"
      ? new Set([view.your_player_id, ...view.allies])
      : new Set([view.your_player_id]);
  const candidates = view.players.filter((p) => p.alive && !excludeIds.has(p.player_id));

  const handleSubmit = async () => {
    if (!target || !actionType) return;
    setSubmitting(true);
    try {
      await submitNightAction(sessionId, actionType, target);
      setTarget("");
      await refreshView();
    } catch (e) {
      setError(e instanceof Error ? e.message : "行動の送信に失敗しました");
    } finally {
      setSubmitting(false);
    }
  };

  const handleSkip = async () => {
    setSubmitting(true);
    try {
      await resolveNight(sessionId);
      await refreshView();
    } catch (e) {
      setError(e instanceof Error ? e.message : "処理に失敗しました");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="panel night-action-panel">
      <h3>夜の行動: {ACTION_LABELS[actionType]}</h3>
      <select value={target} onChange={(e) => setTarget(e.target.value)} disabled={submitting}>
        <option value="">対象を選択...</option>
        {candidates.map((p) => (
          <option key={p.player_id} value={p.player_id}>
            {p.name}
          </option>
        ))}
      </select>
      <div className="night-action-panel__actions">
        <button className="btn btn--primary" disabled={!target || submitting} onClick={() => void handleSubmit()}>
          {ACTION_LABELS[actionType]}
        </button>
        {actionType === "divine" && view.day === 0 && (
          <button className="btn" disabled={submitting} onClick={() => void handleSkip()}>
            占わずに進める
          </button>
        )}
      </div>
    </div>
  );
}
