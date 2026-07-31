import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerListPanel } from "../components/panels/PlayerListPanel";
import { useGameStore } from "../state/gameStore";
import type { GameView } from "../api/types";

function makeView(overrides: Partial<GameView> = {}): GameView {
  return {
    session_id: "s1",
    phase: "discussion",
    day: 1,
    vote_round: 1,
    your_player_id: "p0",
    your_role: "villager",
    allies: [],
    players: [
      { player_id: "p0", name: "Taro", alive: true, death_cause: null, death_day: null },
      { player_id: "p1", name: "Hanako", alive: false, death_cause: "executed", death_day: 1 },
    ],
    public_chat: [],
    private_chat: [],
    your_divine_results: [],
    your_medium_results: [],
    co_declarations: [],
    vote_history: [],
    winner: null,
    victory_reason: "",
    is_draw: false,
    ...overrides,
  };
}

describe("PlayerListPanel", () => {
  afterEach(() => {
    useGameStore.getState().reset();
  });

  it("renders alive and dead players with a death tag", () => {
    useGameStore.setState({ view: makeView() });
    render(<PlayerListPanel />);

    expect(screen.getByText("Taro")).toBeInTheDocument();
    expect(screen.getByText("Hanako")).toBeInTheDocument();
    expect(screen.getByText("処刑")).toBeInTheDocument();
    expect(screen.getByText("プレイヤー (1/2 生存)")).toBeInTheDocument();
  });

  it("renders nothing without a view", () => {
    useGameStore.setState({ view: null });
    const { container } = render(<PlayerListPanel />);
    expect(container).toBeEmptyDOMElement();
  });
});
