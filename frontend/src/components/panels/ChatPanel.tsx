import { useState } from "react";
import { sendChat } from "../../api/client";
import type { ChatChannel } from "../../api/types";
import { useGameStore } from "../../state/gameStore";

export function ChatPanel() {
  const view = useGameStore((s) => s.view);
  const sessionId = useGameStore((s) => s.sessionId);
  const playerNames = useGameStore((s) => s.playerNames);
  const refreshView = useGameStore((s) => s.refreshView);
  const setError = useGameStore((s) => s.setError);
  const selectedSpeakerId = useGameStore((s) => s.selectedSpeakerId);
  const setSelectedSpeakerId = useGameStore((s) => s.setSelectedSpeakerId);

  const [tab, setTab] = useState<ChatChannel>("public");
  const [selectedDay, setSelectedDay] = useState<number | "all">("all");
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);

  if (!view || !sessionId) return null;

  const isAlive = view.players.find((p) => p.player_id === view.your_player_id)?.alive ?? false;
  const canUseWolfChat = view.your_role === "werewolf";
  const canUseFreemasonChat = view.your_role === "freemason";
  const canSendPublic = isAlive && view.phase === "discussion";
  const canSendPrivate =
    isAlive && (tab === "wolf" ? canUseWolfChat : tab === "freemason" ? canUseFreemasonChat : false);
  const canSend = tab === "public" ? canSendPublic : canSendPrivate;

  const channelMessages =
    tab === "public" ? view.public_chat : view.private_chat.filter((m) => m.channel === tab);
  const availableDays = Array.from(
    new Set([...channelMessages.map((message) => message.day), ...view.vote_history.map((vote) => vote.day)]),
  ).sort((a, b) => a - b);
  const messages = channelMessages.filter(
    (message) =>
      (selectedDay === "all" || message.day === selectedDay) &&
      (tab !== "public" || selectedSpeakerId === null || message.author_id === selectedSpeakerId),
  );
  const visibleVotes = view.vote_history.filter(
    (vote) => selectedDay === "all" || vote.day === selectedDay,
  );

  const handleSend = async () => {
    const content = draft.trim();
    if (!content || !canSend) return;
    setSending(true);
    try {
      await sendChat(sessionId, content, tab);
      setDraft("");
      await refreshView();
    } catch (e) {
      setError(e instanceof Error ? e.message : "発言に失敗しました");
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="panel chat-panel">
      <div className="chat-panel__tabs">
        <button
          className={tab === "public" ? "tab tab--active" : "tab"}
          onClick={() => setTab("public")}
        >
          全体チャット
        </button>
        {canUseWolfChat && (
          <button className={tab === "wolf" ? "tab tab--active" : "tab"} onClick={() => setTab("wolf")}>
            人狼チャット
          </button>
        )}
        {canUseFreemasonChat && (
          <button
            className={tab === "freemason" ? "tab tab--active" : "tab"}
            onClick={() => setTab("freemason")}
          >
            共有者チャット
          </button>
        )}
      </div>

      <div className="chat-panel__days" aria-label="日付別ログ">
        <button className={selectedDay === "all" ? "tab tab--active" : "tab"} onClick={() => setSelectedDay("all")}>全日程</button>
        {availableDays.map((day) => (
          <button key={day} className={selectedDay === day ? "tab tab--active" : "tab"} onClick={() => setSelectedDay(day)}>
            {day}日目
          </button>
        ))}
      </div>

      {tab === "public" && selectedSpeakerId && (
        <button className="speaker-filter" onClick={() => setSelectedSpeakerId(null)}>
          {playerNames[selectedSpeakerId] ?? selectedSpeakerId}の発言のみ ×
        </button>
      )}

      <div className="chat-panel__log">
        {messages.length === 0 && <p className="chat-panel__empty">まだ発言はありません</p>}
        {messages.map((m, i) => (
          <div
            key={i}
            className={
              m.author_id === view.your_player_id ? "chat-message chat-message--you" : "chat-message"
            }
          >
            <button className="chat-message__author" onClick={() => setSelectedSpeakerId(m.author_id)}>
              {playerNames[m.author_id] ?? m.author_id}
            </button>: {m.content}
          </div>
        ))}
        {tab === "public" && visibleVotes.length > 0 && (
          <section className="vote-history">
            <h4>投票記録</h4>
            {visibleVotes.map((vote, index) => (
              <p key={`${vote.day}-${vote.round}-${vote.voter_id}-${index}`}>
                {vote.day}日目 R{vote.round}: {playerNames[vote.voter_id] ?? vote.voter_id} → {playerNames[vote.target_id] ?? vote.target_id}
              </p>
            ))}
          </section>
        )}
      </div>

      <div className="chat-panel__input">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void handleSend();
          }}
          disabled={!canSend || sending}
          maxLength={500}
          placeholder={canSend ? "発言を入力..." : "現在は発言できません"}
        />
        <button className="btn" disabled={!canSend || sending || !draft.trim()} onClick={() => void handleSend()}>
          送信
        </button>
      </div>
    </div>
  );
}
