import { useEffect, useRef } from "react";
import { WS_BASE } from "./client";
import { useGameStore } from "../state/gameStore";

/** Low-latency push channel: any event (chat, phase change, death, etc.)
 * simply triggers a fresh view fetch rather than fine-grained patching --
 * simple and robust given the view endpoint is already cheap. Reconnects
 * with exponential backoff. */
export function useGameSocket(): void {
  const sessionId = useGameStore((s) => s.sessionId);
  const humanId = useGameStore((s) => s.humanId);
  const setConnected = useGameStore((s) => s.setConnected);
  const refreshView = useGameStore((s) => s.refreshView);
  const refreshDebug = useGameStore((s) => s.refreshDebug);

  const onEventRef = useRef<() => void>(() => {});
  onEventRef.current = () => {
    void refreshView();
    void refreshDebug();
  };

  useEffect(() => {
    if (!sessionId || !humanId) return;

    let cancelled = false;
    let socket: WebSocket | null = null;
    let retryDelay = 1000;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) return;
      socket = new WebSocket(`${WS_BASE}/ws/${sessionId}/${humanId}`);

      socket.onopen = () => {
        retryDelay = 1000;
        setConnected(true);
      };
      socket.onmessage = () => {
        onEventRef.current();
      };
      socket.onclose = () => {
        setConnected(false);
        if (cancelled) return;
        retryTimer = setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, 15000);
      };
      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      socket?.close();
    };
  }, [sessionId, humanId, setConnected]);
}
