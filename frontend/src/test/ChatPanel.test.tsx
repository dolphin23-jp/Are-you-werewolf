import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { passDiscussionTurn, sendChat } from "../api/client";
import { ChatPanel } from "../components/panels/ChatPanel";
import { useGameStore } from "../state/gameStore";
import { makeView } from "./fixtures/gameView";

vi.mock("../api/client", () => ({ sendChat: vi.fn(), passDiscussionTurn: vi.fn() }));

describe("ChatPanel", () => {
  afterEach(() => {
    useGameStore.getState().reset();
    vi.clearAllMocks();
    vi.mocked(passDiscussionTurn).mockResolvedValue(undefined);
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

  it("switches between day logs and filters by speaker", () => {
    useGameStore.setState({
      view: makeView({
        day: 2,
        public_chat: [
          {
            message_id: "m1",
            author_id: "p1",
            content: "初日の意見",
            channel: "public",
            day: 1,
            reply_to: null,
            quote: null,
          },
          {
            message_id: "m2",
            author_id: "p2",
            content: "二日目の意見",
            channel: "public",
            day: 2,
            reply_to: null,
            quote: null,
          },
        ],
      }),
      sessionId: "s1",
      playerNames: { p1: "Hanako", p2: "Jiro" },
    });
    render(<ChatPanel />);

    fireEvent.click(screen.getByRole("button", { name: "1日目" }));
    expect(screen.getByText(/初日の意見/)).toBeInTheDocument();
    expect(screen.queryByText(/二日目の意見/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Hanako" }));
    expect(screen.getByRole("button", { name: "Hanakoの発言のみ ×" })).toBeInTheDocument();
  });

  it("does not send on IME Enter and keeps Shift+Enter for a newline", () => {
    useGameStore.setState({ view: makeView(), sessionId: "s1" });
    render(<ChatPanel />);
    const textarea = screen.getByPlaceholderText("発言を入力...");
    fireEvent.change(textarea, { target: { value: "変換中" } });

    fireEvent.keyDown(textarea, { key: "Enter", isComposing: true, keyCode: 229 });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });

    expect(sendChat).not.toHaveBeenCalled();
    expect(textarea).toHaveValue("変換中");
  });

  it("shows a reply preview and uses message ids as reply targets", async () => {
    useGameStore.setState({ view: makeView(), sessionId: "s1", playerNames: { p1: "Hanako" } });
    render(<ChatPanel />);

    fireEvent.click(screen.getByRole("button", { name: "m1に返信" }));

    expect(screen.getByText(/返信先 \[m1\] Hanako/)).toBeInTheDocument();
  });

  it("shows waiting state and passes manually", async () => {
    useGameStore.setState({
      view: makeView({ awaiting_your_speech: true, speech_wait_remaining_seconds: 45 }),
      sessionId: "s1",
    });
    render(<ChatPanel />);

    expect(screen.getByText(/あなたの発言を待っています/)).toHaveTextContent("残り 45 秒");
    fireEvent.click(screen.getByRole("button", { name: "パス" }));
    await waitFor(() => expect(passDiscussionTurn).toHaveBeenCalledWith("s1"));
  });

  it("automatically passes when the countdown reaches zero", async () => {
    useGameStore.setState({
      view: makeView({ awaiting_your_speech: true, speech_wait_remaining_seconds: 0 }),
      sessionId: "s1",
    });
    render(<ChatPanel />);

    await waitFor(() => expect(passDiscussionTurn).toHaveBeenCalledWith("s1"));
  });

  it("turns a pending question into a reply selection", () => {
    useGameStore.setState({
      view: makeView({
        pending_questions: [
          { asker: "p1", target: "p0", question: "理由は?", source_message_id: "m1", day: 1 },
        ],
      }),
      sessionId: "s1",
      playerNames: { p1: "Hanako" },
    });
    render(<ChatPanel />);

    fireEvent.click(screen.getByRole("button", { name: /\[m1\] Hanako: 「理由は\?」/ }));

    expect(screen.getByText(/返信先 \[m1\] Hanako/)).toBeInTheDocument();
  });

  it("renders a quote with its source author", () => {
    useGameStore.setState({
      view: makeView({
        public_chat: [
          ...makeView().public_chat,
          { message_id: "m2", author_id: "p2", content: "回答", channel: "public", day: 1, reply_to: "m1", quote: "こんにちは" },
        ],
      }),
      sessionId: "s1",
      playerNames: { p1: "Hanako", p2: "Jiro" },
    });
    render(<ChatPanel />);

    expect(screen.getByRole("button", { name: /Hanako こんにちは/ })).toBeInTheDocument();
  });

  it("scrolls to the newest message while already near the bottom", () => {
    const scrollTo = vi.fn();
    Element.prototype.scrollTo = scrollTo;
    useGameStore.setState({ view: makeView(), sessionId: "s1" });

    render(<ChatPanel />);

    expect(scrollTo).toHaveBeenCalled();
  });
});
