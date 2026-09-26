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

  it("reset turns off the all-roles view before the next game", () => {
    useGameStore.getState().setSession("sess1", "p0", { p0: "Taro" });
    useGameStore.setState({ debugMode: true });
    useGameStore.getState().reset();
    expect(useGameStore.getState().debugMode).toBe(false);
  });

  it("a new session never inherits the previous game's all-roles view", () => {
    useGameStore.setState({ debugMode: true });
    useGameStore.getState().setSession("sess2", "p0", { p0: "Taro" });
    expect(useGameStore.getState().debugMode).toBe(false);
  });
});
