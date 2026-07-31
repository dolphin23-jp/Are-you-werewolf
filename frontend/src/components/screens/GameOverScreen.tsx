import { ROLE_LABELS } from "../../api/types";
import { downloadTranscript } from "../../api/client";
import { useGameStore } from "../../state/gameStore";
import { useState } from "react";

const WINNER_LABELS: Record<string, string> = {
  village: "村人陣営の勝利",
  werewolf: "人狼陣営の勝利",
  fox: "妖狐陣営の勝利",
};

export function GameOverScreen() {
  const view = useGameStore((s) => s.view);
  const debug = useGameStore((s) => s.debug);
  const debugMode = useGameStore((s) => s.debugMode);
  const toggleDebug = useGameStore((s) => s.toggleDebug);
  const refreshDebug = useGameStore((s) => s.refreshDebug);
  const reset = useGameStore((s) => s.reset);
  const sessionId = useGameStore((s) => s.sessionId);
  const setError = useGameStore((s) => s.setError);
  const [downloading, setDownloading] = useState(false);

  if (!view) return null;

  const resultLabel = view.is_draw
    ? "引き分け"
    : (view.winner && WINNER_LABELS[view.winner]) || "終了";

  const handleShowRoles = () => {
    toggleDebug();
    void refreshDebug();
  };

  const handleDownloadTranscript = async () => {
    if (!sessionId) return;
    setDownloading(true);
    try {
      await downloadTranscript(sessionId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "解析用ログの取得に失敗しました");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="screen screen--game-over">
      <h1>{resultLabel}</h1>
      <p className="lead">{view.victory_reason}</p>

      <button className="btn" onClick={handleShowRoles}>
        {debugMode ? "役職一覧を隠す" : "全員の役職を見る"}
      </button>

      {debugMode && debug && (
        <ul className="role-reveal-list">
          {debug.players.map((p) => (
            <li key={p.player_id}>
              <strong>{p.name}</strong>: {ROLE_LABELS[p.role]}
              {!p.alive && `(${p.death_cause ?? "死亡"})`}
            </li>
          ))}
        </ul>
      )}

      <button className="btn" disabled={downloading} onClick={() => void handleDownloadTranscript()}>
        {downloading ? "ログを準備中..." : "解析用ログをダウンロード"}
      </button>

      <button className="btn btn--primary" onClick={reset}>
        もう一度遊ぶ
      </button>
    </div>
  );
}
