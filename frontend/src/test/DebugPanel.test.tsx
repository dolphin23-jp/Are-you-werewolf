import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DebugPanel } from "../components/panels/DebugPanel";
import { useGameStore } from "../state/gameStore";
import { makeView } from "./fixtures/gameView";

vi.mock("../api/client", () => ({
  getDebugView: vi.fn().mockResolvedValue(null),
  getView: vi.fn(),
}));

describe("DebugPanel", () => {
  afterEach(() => {
    useGameStore.getState().reset();
  });

  it("is not offered while the server keeps roles hidden", () => {
    useGameStore.setState({ view: makeView({ debug_available: false }), sessionId: "s1" });
    const { container } = render(<DebugPanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it("is offered when the server allows the all-roles view", () => {
    useGameStore.setState({ view: makeView({ debug_available: true }), sessionId: "s1" });
    render(<DebugPanel />);
    expect(screen.getByRole("button", { name: "表示する" })).toBeInTheDocument();
  });
});
