import { endDiscussion, startDiscussion, startNight } from "../../api/client";
import { PhaseBanner } from "../common/PhaseBanner";
import { RolePanel } from "../panels/RolePanel";
import { PlayerListPanel } from "../panels/PlayerListPanel";
import { ChatPanel } from "../panels/ChatPanel";
import { NightActionPanel } from "../panels/NightActionPanel";
import { VotePanel } from "../panels/VotePanel";
import { COPanel } from "../panels/COPanel";
import { DebugPanel } from "../panels/DebugPanel";
import { GameOverScreen } from "./GameOverScreen";
import { useGameStore } from "../../state/gameStore";

export function MainGameScreen() {
  const view = useGameStore((s) => s.view);
  const sessionId = useGameStore((s) => s.sessionId);
  const connected = useGameStore((s) => s.connected);
  const error = useGameStore((s) => s.error);
  const busy = useGameStore((s) => s.busy);
  const setBusy = useGameStore((s) => s.setBusy);
  const refreshView = useGameStore((s) => s.refreshView);
  const setError = useGameStore((s) => s.setError);

  if (!view || !sessionId) {
    return (
      <div className="screen screen--main">
        <p>ゲーム情報を読み込んでいます...</p>
      </div>
    );
  }

  if (view.phase === "game_over") {
    return <GameOverScreen />;
  }

  const runAction = async (action: (id: string) => Promise<void>) => {
    setBusy(true);
    try {
      await action(sessionId);
      await refreshView();
    } catch (e) {
      setError(e instanceof Error ? e.message : "操作に失敗しました");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen screen--main">
      <PhaseBanner phase={view.phase} day={view.day} connected={connected} />
      {error && <p className="error-text">{error}</p>}

      <div className="main-game-layout">
        <div className="main-game-layout__sidebar">
          <RolePanel />
          <PlayerListPanel />
          <COPanel />
        </div>
        <div className="main-game-layout__center">
          <ChatPanel />
          <NightActionPanel />
          <VotePanel />

          {view.phase === "dawn" && (
            <button className="btn btn--primary" disabled={busy} onClick={() => void runAction(startDiscussion)}>
              議論を始める
            </button>
          )}
          {view.phase === "discussion" && (
            <button className="btn btn--primary" disabled={busy} onClick={() => void runAction(endDiscussion)}>
              投票へ進む
            </button>
          )}
          {view.phase === "vote_result" && (
            <button className="btn btn--primary" disabled={busy} onClick={() => void runAction(startNight)}>
              夜へ進む
            </button>
          )}
        </div>
        <div className="main-game-layout__sidebar">
          <DebugPanel />
        </div>
      </div>
    </div>
  );
}
