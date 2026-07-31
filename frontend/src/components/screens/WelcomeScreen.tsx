import { useState } from "react";
import { createGame, startGame } from "../../api/client";
import { useGameStore } from "../../state/gameStore";

export function WelcomeScreen() {
  const [name, setName] = useState("ゲスト");
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const setSession = useGameStore((s) => s.setSession);
  const refreshView = useGameStore((s) => s.refreshView);

  const handleStart = async () => {
    setLoading(true);
    setLocalError(null);
    try {
      const created = await createGame(name.trim() || "ゲスト");
      await startGame(created.session_id);
      setSession(created.session_id, created.human_player_id, created.player_names);
      await refreshView();
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : "ゲームの開始に失敗しました");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="screen screen--welcome">
      <h1>Are you werewolf?</h1>
      <p className="lead">AI 16人と挑む、本格チャット人狼。</p>
      <label className="field">
        <span>あなたの名前</span>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          maxLength={16}
          placeholder="あなたの名前"
        />
      </label>
      {localError && <p className="error-text">{localError}</p>}
      <button className="btn btn--primary" disabled={loading} onClick={() => void handleStart()}>
        {loading ? "準備中..." : "ゲームを始める"}
      </button>
    </div>
  );
}
