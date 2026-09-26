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
  /** An action the player took failed. Stays until dismissed or replaced. */
  error: string | null;
  /** The latest view refresh failed; cleared by the next one that succeeds. */
  connectionError: string | null;
  /** Shown on the welcome screen after a game was lost (e.g. server restart). */
  notice: string | null;
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
  /** The server no longer has this game: go back to the start with a notice. */
  sessionLost: () => void;
}

export const SESSION_LOST_NOTICE =
  "ゲームが見つかりません。サーバーが再起動した可能性があります。新しいゲームを始めてください。";

/** A 404 for the game itself. Duck-typed rather than `instanceof ApiError`,
 * so the check does not depend on how the client module is loaded. */
function isSessionGone(error: unknown): boolean {
  return typeof error === "object" && error !== null && (error as { status?: unknown }).status === 404;
}

const viewRefresh: { inFlight: Promise<void> | null; again: boolean } = {
  inFlight: null,
  again: false,
};

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
  connectionError: null,
  notice: null,
  busy: false,
  selectedSpeakerId: null,

  setScreen: (screen) => set({ screen }),
  // Every new game starts with hidden information hidden. debugMode is the
  // game-over "show all roles" toggle too; carried over, the next game would
  // open with every role and all wolf chat on screen.
  setSession: (sessionId, humanId, playerNames) =>
    set({
      sessionId,
      humanId,
      playerNames,
      view: null,
      debug: null,
      debugMode: false,
      error: null,
      connectionError: null,
      notice: null,
      screen: "role-reveal",
    }),
  setConnected: (connected) => set({ connected }),
  setError: (error) => set({ error }),
  setBusy: (busy) => set({ busy }),
  setSelectedSpeakerId: (selectedSpeakerId) => set({ selectedSpeakerId }),
  toggleDebug: () => set((s) => ({ debugMode: !s.debugMode })),

  // One request at a time, plus at most one more for whatever asked meanwhile.
  // Every WebSocket event and every poll used to start its own GET: responses
  // overlapped, an older one could land last and roll the phase back, and a
  // poll from the previous game could land after "play again".
  refreshView: () => {
    if (viewRefresh.inFlight) {
      viewRefresh.again = true;
      return viewRefresh.inFlight;
    }
    viewRefresh.inFlight = (async () => {
      do {
        viewRefresh.again = false;
        const { sessionId, humanId } = get();
        if (!sessionId || !humanId) return;
        try {
          const view = await getView(sessionId, humanId);
          if (get().sessionId !== sessionId) return;
          // Only the connection error: an action's error must outlive the next
          // poll, which used to erase it within milliseconds.
          set({ view, connectionError: null });
        } catch (e) {
          if (get().sessionId !== sessionId) return;
          if (isSessionGone(e)) {
            // Polling and reconnecting would go on forever behind a banner.
            get().sessionLost();
            return;
          }
          set({ connectionError: e instanceof Error ? e.message : "通信エラーが発生しました" });
        }
      } while (viewRefresh.again);
    })().finally(() => {
      viewRefresh.inFlight = null;
    });
    return viewRefresh.inFlight;
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
      debugMode: false,
      connected: false,
      error: null,
      connectionError: null,
      notice: null,
      busy: false,
      selectedSpeakerId: null,
    }),

  sessionLost: () => {
    get().reset();
    set({ notice: SESSION_LOST_NOTICE });
  },
}));
