import { beforeEach, describe, expect, it } from "vitest";
import { useGameStore } from "../state/gameStore";

describe("gameStore", () => {
  beforeEach(() => {
    useGameStore.getState().reset();
  });

  it("starts on the welcome screen", () => {
    expect(useGameStore.getState().screen).toBe("welcome");
  });

  it("setSession moves to role-reveal and stores identifiers", () => {
    useGameStore.getState().setSession("sess1", "p0", { p0: "Taro" });
    const state = useGameStore.getState();
    expect(state.screen).toBe("role-reveal");
    expect(state.sessionId).toBe("sess1");
    expect(state.humanId).toBe("p0");
    expect(state.playerNames).toEqual({ p0: "Taro" });
  });

  it("reset returns to a clean welcome state", () => {
    useGameStore.getState().setSession("sess1", "p0", { p0: "Taro" });
    useGameStore.getState().setScreen("main");
    useGameStore.getState().reset();
    const state = useGameStore.getState();
    expect(state.screen).toBe("welcome");
    expect(state.sessionId).toBeNull();
    expect(state.view).toBeNull();
  });

  it("toggleDebug flips debugMode", () => {
    expect(useGameStore.getState().debugMode).toBe(false);
    useGameStore.getState().toggleDebug();
    expect(useGameStore.getState().debugMode).toBe(true);
  });
});
