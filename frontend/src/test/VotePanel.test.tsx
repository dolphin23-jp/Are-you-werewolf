import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { VotePanel } from "../components/panels/VotePanel";
import type { GameView } from "../api/types";
import { useGameStore } from "../state/gameStore";
import { makeView as makeBaseView } from "./fixtures/gameView";

function makeView(overrides: Partial<GameView> = {}): GameView {
  return makeBaseView({ phase: "voting", ...overrides });
}

function optionNames(): string[] {
  return screen
    .getAllByRole("option")
    .map((option) => option.textContent ?? "")
    .filter((text) => text !== "投票先を選択...");
}

describe("VotePanel", () => {
  afterEach(() => {
    useGameStore.getState().reset();
  });

  it("offers every living player except yourself in an ordinary round", () => {
    useGameStore.setState({ view: makeView(), sessionId: "s1" });
    render(<VotePanel />);
    expect(optionNames()).toEqual(["Hanako", "Jiro"]);
  });

  it("narrows the options to the tied players during a runoff", () => {
    useGameStore.setState({
      view: makeView({ phase: "runoff", vote_round: 2, runoff_candidates: ["p1"] }),
      sessionId: "s1",
    });
    render(<VotePanel />);
    // Jiro was eliminated from the runoff, so offering him would only earn a
    // server-side rejection.
    expect(optionNames()).toEqual(["Hanako"]);
    expect(screen.getByText(/同数だったプレイヤーのみ/)).toBeInTheDocument();
  });

  it("does not exclude a tied player who happens to be you", () => {
    useGameStore.setState({
      view: makeView({ phase: "runoff", vote_round: 2, runoff_candidates: ["p0", "p2"] }),
      sessionId: "s1",
    });
    render(<VotePanel />);
    expect(optionNames()).toEqual(["Jiro"]);
  });

  it("uses the server's current-round vote state instead of prior-day history", () => {
    useGameStore.setState({
      view: makeView({
        day: 2,
        vote_round: 1,
        has_voted_current_round: false,
        vote_history: [{ voter_id: "p0", target_id: "p1", day: 1, round: 1 }],
      }),
      sessionId: "s1",
    });
    render(<VotePanel />);
    expect(screen.getByRole("combobox")).toBeEnabled();
    expect(screen.queryByText(/他のプレイヤーの投票を待っています/)).not.toBeInTheDocument();
  });
});
