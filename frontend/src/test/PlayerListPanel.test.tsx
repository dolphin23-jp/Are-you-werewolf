import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerListPanel } from "../components/panels/PlayerListPanel";
import { useGameStore } from "../state/gameStore";
import { makeView } from "./fixtures/gameView";

describe("PlayerListPanel", () => {
  afterEach(() => {
    useGameStore.getState().reset();
  });

  it("renders alive and dead players with a death tag", () => {
    useGameStore.setState({
      view: makeView({
        players: [
          { player_id: "p0", name: "Taro", alive: true, death_cause: null, death_day: null },
          { player_id: "p1", name: "Hanako", alive: false, death_cause: "executed", death_day: 1 },
        ],
      }),
    });
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
