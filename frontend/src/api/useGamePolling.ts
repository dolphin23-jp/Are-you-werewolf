import { useEffect } from "react";
import { useGameStore } from "../state/gameStore";

/** Polling fallback/complement to the WebSocket stream: guarantees the view
 * eventually catches up even if a push event is missed. */
export function useGamePolling(intervalMs = 2500): void {
  const sessionId = useGameStore((s) => s.sessionId);
  const refreshView = useGameStore((s) => s.refreshView);
  const refreshDebug = useGameStore((s) => s.refreshDebug);

  useEffect(() => {
    if (!sessionId) return;
    const id = window.setInterval(() => {
      void refreshView();
      void refreshDebug();
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [sessionId, intervalMs, refreshView, refreshDebug]);
}
