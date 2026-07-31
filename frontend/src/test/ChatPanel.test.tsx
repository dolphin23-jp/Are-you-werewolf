import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { ChatPanel } from "../components/panels/ChatPanel";
import { useGameStore } from "../state/gameStore";
import type { GameView } from "../api/types";

function makeView(overrides: Partial<GameView> = {}): GameView {
  return {
    session_id: "s1",
    phase: "discussion",
    day: 1,
    vote_round: 1,
    runoff_candidates: [],
    your_player_id: "p0",
    your_role: "villager",
    allies: [],
    players: [{ player_id: "p0", name: "Taro", alive: true, death_cause: null, death_day: null }],
    public_chat: [{ author_id: "p1", content: "こんにちは", channel: "public", day: 1 }],
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

describe("ChatPanel", () => {
  afterEach(() => {
    useGameStore.getState().reset();
  });

  it("renders public chat messages", () => {
    useGameStore.setState({ view: makeView(), sessionId: "s1", playerNames: { p1: "Hanako" } });
    render(<ChatPanel />);
    expect(screen.getByText(/こんにちは/)).toBeInTheDocument();
  });

  it("disables the input outside the discussion phase", () => {
    useGameStore.setState({ view: makeView({ phase: "voting" }), sessionId: "s1" });
    render(<ChatPanel />);
    expect(screen.getByPlaceholderText("現在は発言できません")).toBeDisabled();
  });

  it("only shows the wolf-chat tab for werewolves", () => {
    useGameStore.setState({ view: makeView({ your_role: "werewolf" }), sessionId: "s1" });
    render(<ChatPanel />);
    expect(screen.getByText("人狼チャット")).toBeInTheDocument();
    expect(screen.queryByText("共有者チャット")).not.toBeInTheDocument();
  });
});
