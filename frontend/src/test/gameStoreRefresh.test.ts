import { afterEach, describe, expect, it, vi } from "vitest";
import { getView } from "../api/client";
import type { GameView } from "../api/types";
import { useGameStore } from "../state/gameStore";
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
