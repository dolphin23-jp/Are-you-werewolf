import { useEffect } from "react";
import { useGameStore } from "../state/gameStore";

/** Polling fallback/complement to the WebSocket stream: guarantees the view
 * eventually catches up even if a push event is missed. */
export function useGamePolling(intervalMs = 2500, connectedIntervalMs = 10000): void {
  const sessionId = useGameStore((s) => s.sessionId);
  const connected = useGameStore((s) => s.connected);
  const finished = useGameStore((s) => s.view?.phase === "game_over");
  const refreshView = useGameStore((s) => s.refreshView);
  const refreshDebug = useGameStore((s) => s.refreshDebug);

  useEffect(() => {
    // Nothing changes after the game ends, and while the socket is up it
    // already triggers a refresh on every event: polling is only the safety net.
    if (!sessionId || finished) return;
    const id = window.setInterval(
      () => {
        void refreshView();
        void refreshDebug();
      },
      connected ? connectedIntervalMs : intervalMs,
    );
    return () => window.clearInterval(id);
  }, [sessionId, finished, connected, intervalMs, connectedIntervalMs, refreshView, refreshDebug]);
}
