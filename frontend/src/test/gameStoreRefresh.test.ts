import { afterEach, describe, expect, it, vi } from "vitest";
import { getView } from "../api/client";
import type { GameView } from "../api/types";
import { SESSION_LOST_NOTICE, useGameStore } from "../state/gameStore";
import { makeView } from "./fixtures/gameView";

vi.mock("../api/client", () => ({
  getView: vi.fn(),
  getDebugView: vi.fn(),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

describe("refreshView", () => {
  afterEach(() => {
    useGameStore.getState().reset();
    vi.mocked(getView).mockReset();
  });

  it("runs one request at a time and re-fetches once for callers that arrived meanwhile", async () => {
    const first = deferred<GameView>();
    const second = deferred<GameView>();
    vi.mocked(getView).mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    useGameStore.getState().setSession("s1", "p0", {});

    const a = useGameStore.getState().refreshView();
    const b = useGameStore.getState().refreshView();
    const c = useGameStore.getState().refreshView();
    expect(getView).toHaveBeenCalledTimes(1);

    first.resolve(makeView({ phase: "discussion" }));
    second.resolve(makeView({ phase: "voting" }));
    await Promise.all([a, b, c]);

    // Three callers, two requests, and the later answer is the one that stays.
    expect(getView).toHaveBeenCalledTimes(2);
    expect(useGameStore.getState().view?.phase).toBe("voting");
  });

  it("drops a response that belongs to the previous game", async () => {
    const stale = deferred<GameView>();
    vi.mocked(getView).mockReturnValueOnce(stale.promise);
    useGameStore.getState().setSession("s1", "p0", {});
    const pending = useGameStore.getState().refreshView();

    useGameStore.getState().reset();
    useGameStore.getState().setSession("s2", "p0", {});
    stale.resolve(makeView({ session_id: "s1" }));
    await pending;

    expect(useGameStore.getState().view).toBeNull();
  });

  it("keeps an action's error through a successful refresh", async () => {
    vi.mocked(getView).mockResolvedValue(makeView());
    useGameStore.getState().setSession("s1", "p0", {});
    useGameStore.getState().setError("投票に失敗しました");

    await useGameStore.getState().refreshView();

    expect(useGameStore.getState().error).toBe("投票に失敗しました");
  });
});

describe("a game the server no longer has", () => {
  afterEach(() => {
    useGameStore.getState().reset();
    vi.mocked(getView).mockReset();
  });

  it("returns to the welcome screen with a notice instead of polling forever", async () => {
    vi.mocked(getView).mockRejectedValueOnce(
      Object.assign(new Error("session not found"), { status: 404 }),
    );
    useGameStore.getState().setSession("s1", "p0", {});

    await useGameStore.getState().refreshView();

    const state = useGameStore.getState();
    expect(state.screen).toBe("welcome");
    expect(state.sessionId).toBeNull();
    expect(state.notice).toBe(SESSION_LOST_NOTICE);
    expect(state.connectionError).toBeNull();
  });

  it("keeps the game on a transient failure", async () => {
    vi.mocked(getView).mockRejectedValueOnce(
      Object.assign(new Error("HTTP 502"), { status: 502 }),
    );
    useGameStore.getState().setSession("s1", "p0", {});

    await useGameStore.getState().refreshView();

    expect(useGameStore.getState().sessionId).toBe("s1");
    expect(useGameStore.getState().connectionError).toBe("HTTP 502");
  });

  it("clears the notice when a new game starts", () => {
    useGameStore.getState().sessionLost();
    useGameStore.getState().setSession("s2", "p0", {});
    expect(useGameStore.getState().notice).toBeNull();
  });
});
