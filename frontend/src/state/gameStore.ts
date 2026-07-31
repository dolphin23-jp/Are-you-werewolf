import { create } from "zustand";
import { getDebugView, getView } from "../api/client";
import type { DebugView, GameView } from "../api/types";

export type Screen = "welcome" | "role-reveal" | "main";

interface GameStoreState {
  screen: Screen;
  sessionId: string | null;
  humanId: string | null;
  playerNames: Record<string, string>;
  view: GameView | null;
  debug: DebugView | null;
  debugMode: boolean;
  connected: boolean;
  error: string | null;
  busy: boolean;
  selectedSpeakerId: string | null;

  setScreen: (screen: Screen) => void;
  setSession: (
    sessionId: string,
    humanId: string,
    playerNames: Record<string, string>,
  ) => void;
  setConnected: (connected: boolean) => void;
  setError: (error: string | null) => void;
  setBusy: (busy: boolean) => void;
  setSelectedSpeakerId: (playerId: string | null) => void;
  toggleDebug: () => void;
  refreshView: () => Promise<void>;
  refreshDebug: () => Promise<void>;
  reset: () => void;
}

export const useGameStore = create<GameStoreState>((set, get) => ({
  screen: "welcome",
  sessionId: null,
  humanId: null,
  playerNames: {},
  view: null,
  debug: null,
  debugMode: false,
  connected: false,
  error: null,
  busy: false,
  selectedSpeakerId: null,

  setScreen: (screen) => set({ screen }),
  setSession: (sessionId, humanId, playerNames) =>
    set({ sessionId, humanId, playerNames, view: null, debug: null, screen: "role-reveal" }),
  setConnected: (connected) => set({ connected }),
  setError: (error) => set({ error }),
  setBusy: (busy) => set({ busy }),
  setSelectedSpeakerId: (selectedSpeakerId) => set({ selectedSpeakerId }),
  toggleDebug: () => set((s) => ({ debugMode: !s.debugMode })),

  refreshView: async () => {
    const { sessionId, humanId } = get();
    if (!sessionId || !humanId) return;
    try {
      const view = await getView(sessionId, humanId);
      set({ view, error: null });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : "通信エラーが発生しました" });
    }
  },

  refreshDebug: async () => {
    const { sessionId, debugMode } = get();
    if (!sessionId || !debugMode) return;
    try {
      const debug = await getDebugView(sessionId);
      set({ debug });
    } catch {
      // best-effort only
    }
  },

  reset: () =>
    set({
      screen: "welcome",
      sessionId: null,
      humanId: null,
      playerNames: {},
      view: null,
      debug: null,
      connected: false,
      error: null,
      busy: false,
      selectedSpeakerId: null,
    }),
}));
