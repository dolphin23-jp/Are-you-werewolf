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
      view: makeView({
        awaiting_your_speech: true,
        speech_wait_remaining_seconds: 0,
        speech_wait_token: "1:100",
      }),
      sessionId: "s1",
    });
    render(<ChatPanel />);

    await waitFor(() => expect(passDiscussionTurn).toHaveBeenCalledWith("s1"));
  });

  it("auto-passes only once while the same wait is still pending", async () => {
    // The 2.5s poll replaces `view` wholesale. Without per-wait deduping, a wait
    // that lingers at zero would fire a pass request on every refresh.
    const waiting = {
      awaiting_your_speech: true,
      speech_wait_remaining_seconds: 0,
      speech_wait_token: "1:100",
    };
    useGameStore.setState({ view: makeView(waiting), sessionId: "s1" });
    const { rerender } = render(<ChatPanel />);
    await waitFor(() => expect(passDiscussionTurn).toHaveBeenCalledTimes(1));

    useGameStore.setState({ view: makeView(waiting) });
    rerender(<ChatPanel />);
    useGameStore.setState({ view: makeView(waiting) });
    rerender(<ChatPanel />);

    expect(passDiscussionTurn).toHaveBeenCalledTimes(1);
  });

  it("auto-passes again once a new wait begins", async () => {
    useGameStore.setState({
      view: makeView({
        awaiting_your_speech: true,
        speech_wait_remaining_seconds: 0,
        speech_wait_token: "1:100",
      }),
      sessionId: "s1",
    });
    const { rerender } = render(<ChatPanel />);
    await waitFor(() => expect(passDiscussionTurn).toHaveBeenCalledTimes(1));

    useGameStore.setState({
      view: makeView({
        awaiting_your_speech: true,
        speech_wait_remaining_seconds: 0,
        speech_wait_token: "1:200",
      }),
    });
    rerender(<ChatPanel />);

    await waitFor(() => expect(passDiscussionTurn).toHaveBeenCalledTimes(2));
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

  it("shows discussion progress to living players, not only after death", () => {
    useGameStore.setState({
      view: makeView({ discussion_progress: { spoken: 3, total: 12 } }),
      sessionId: "s1",
    });
    render(<ChatPanel />);
    expect(screen.getByText("AI議論進行: 3/12")).toBeInTheDocument();
  });

  it("hides discussion progress before a round has started", () => {
    useGameStore.setState({ view: makeView(), sessionId: "s1" });
    render(<ChatPanel />);
    expect(screen.queryByText(/AI議論進行/)).not.toBeInTheDocument();
  });

  it("keeps the send button usable while a pass is in flight", async () => {
    // Pass and send are independent actions; they used to share one `sending`
    // flag, so passing disabled sending and vice versa.
    let releasePass: () => void = () => {};
    vi.mocked(passDiscussionTurn).mockReturnValue(
      new Promise<void>((resolve) => {
        releasePass = resolve;
      }),
    );
    useGameStore.setState({
      view: makeView({ awaiting_your_speech: true, speech_wait_remaining_seconds: 30 }),
      sessionId: "s1",
    });
    render(<ChatPanel />);

    fireEvent.change(screen.getByPlaceholderText("発言を入力..."), {
      target: { value: "考えを述べます" },
    });
    fireEvent.click(screen.getByRole("button", { name: "パス" }));

    await waitFor(() => expect(passDiscussionTurn).toHaveBeenCalled());
    expect(screen.getByRole("button", { name: "パス" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "送信" })).not.toBeDisabled();
    releasePass();
  });

  it("keeps a your-message body wrapper on grouped follow-up lines", () => {
    // The "this is yours" accent hangs off .chat-message--you .chat-message__content
    // because grouped lines omit the author button. jsdom cannot evaluate the
    // stylesheet, so pin the structure the selector needs instead.
    useGameStore.setState({
      view: makeView({
        your_player_id: "p1",
        public_chat: [
          { message_id: "m1", author_id: "p1", content: "一言目", channel: "public", day: 1, reply_to: null, quote: null },
          { message_id: "m2", author_id: "p1", content: "続けて二言目", channel: "public", day: 1, reply_to: null, quote: null },
        ],
      }),
      sessionId: "s1",
      playerNames: { p1: "Hanako" },
    });
    const { container } = render(<ChatPanel />);

    const grouped = container.querySelector("#chat-m2");
    expect(grouped?.className).toContain("chat-message--you");
    expect(grouped?.className).toContain("chat-message--grouped");
    expect(grouped?.querySelector(".chat-message__content")).not.toBeNull();
    // The author button really is absent on the grouped line -- that is why the
    // accent had to move off it.
    expect(grouped?.querySelector(".chat-message__author")).toBeNull();
  });

  it("scrolls to the newest message while already near the bottom", () => {
    const scrollTo = vi.fn();
    Element.prototype.scrollTo = scrollTo;
    useGameStore.setState({ view: makeView(), sessionId: "s1" });

    render(<ChatPanel />);

    expect(scrollTo).toHaveBeenCalled();
  });
});
