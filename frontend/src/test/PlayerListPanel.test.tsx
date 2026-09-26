import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PlayerListPanel } from "../components/panels/PlayerListPanel";
import { useGameStore } from "../state/gameStore";
import { nameStanding } from "../state/playerStanding";
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

  it("shows whether a publicly named shared partner has confirmed", () => {
    useGameStore.setState({
      view: makeView({
        freemason_partner_claims: [
          { claimant_id: "p1", partner_id: "p2", day: 1, confirmed: false },
        ],
      }),
    });
    const { rerender } = render(<PlayerListPanel />);
    expect(screen.getByText("相方確認待ち")).toBeInTheDocument();

    useGameStore.setState({
      view: makeView({
        freemason_partner_claims: [
          { claimant_id: "p1", partner_id: "p2", day: 1, confirmed: true },
        ],
      }),
    });
    rerender(<PlayerListPanel />);
    expect(screen.getAllByText("共有相方確認済み")).toHaveLength(2);
  });
});

describe("name standing", () => {
  it("is white only when every published verdict is white", () => {
    const white = {
      claimant_id: "p4",
      result_type: "seer" as const,
      target_id: "p3",
      is_werewolf: false,
      day: 1,
      source_message_id: "m1",
    };
    const black = { ...white, claimant_id: "p9", is_werewolf: true };

    expect(nameStanding(makeView({ public_result_claims: [white] }), "p3")).toBe("white");
    expect(nameStanding(makeView({ public_result_claims: [white, black] }), "p3")).toBe("gray");
  });
});
