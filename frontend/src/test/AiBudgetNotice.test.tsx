import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AI_BUDGET_EXHAUSTED_MESSAGE, AiBudgetNotice } from "../components/common/AiBudgetNotice";
import { MainGameScreen } from "../components/screens/MainGameScreen";
import { useGameStore } from "../state/gameStore";
import { makeView } from "./fixtures/gameView";

// The game screen's panels call the API on interaction; nothing here should
// reach the network.
vi.mock("../api/client");

describe("AiBudgetNotice", () => {
  afterEach(() => {
    // Unmount first, so resetting the store does not re-render a screen
    // that is about to be thrown away.
    cleanup();
    useGameStore.getState().reset();
  });

  it("renders nothing while the AI still has budget", () => {
    const { container } = render(<AiBudgetNotice exhausted={false} costUsd={3.5} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("explains why the AI stopped and what to do about it", () => {
    render(<AiBudgetNotice exhausted={true} costUsd={null} />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(AI_BUDGET_EXHAUSTED_MESSAGE);
    expect(alert).toHaveTextContent("WEREWOLF_SESSION_MAX_COST_USD");
    expect(alert).toHaveTextContent("新しいゲームを作るか上限を見直してください");
    expect(alert).not.toHaveTextContent("推定費用");
  });

  it("shows the estimated spend when the server reports one", () => {
    render(<AiBudgetNotice exhausted={true} costUsd={30.0412} />);
    expect(screen.getByRole("alert")).toHaveTextContent("このゲームの推定費用: $30.04");
  });

  it("appears on the game screen once the server reports the cap was hit", () => {
    useGameStore.setState({
      view: makeView({ ai_budget_exhausted: true, ai_cost_usd: 30.2 }),
      sessionId: "s1",
      humanId: "p0",
    });
    render(<MainGameScreen />);
    expect(screen.getByRole("alert")).toHaveTextContent(AI_BUDGET_EXHAUSTED_MESSAGE);
  });

  it("is absent from the game screen in an ordinary game", () => {
    useGameStore.setState({ view: makeView(), sessionId: "s1", humanId: "p0" });
    render(<MainGameScreen />);
    expect(screen.queryByText(AI_BUDGET_EXHAUSTED_MESSAGE)).not.toBeInTheDocument();
  });
});
